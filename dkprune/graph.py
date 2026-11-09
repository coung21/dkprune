"""Infer network structure from a cfg, and read/write conv parameters in Darknet order.

build(cfg)               -> Graph: per-layer in/out channels, sizes, channel origins, groups, locks
read_params(g, weights)  -> {conv index: ConvParams}, checks the file is consumed exactly
write_params(g, params, header) -> Weights
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dkprune.io import weights as wt
from dkprune.io.cfg import Cfg
from dkprune.layers import registry
from dkprune.layers.base import IMAGE, Channel, ConvSpec, GraphError, Input, opt_int


# ---------------------------------------------------------------- structure

@dataclass(frozen=True)
class LayerInfo:
    index: int
    type: str
    sources: tuple[int, ...]            # -1 = input image
    in_channels: int                    # for route/shortcut: channels of the previous layer
    out_channels: int
    in_hw: tuple[int, int]
    out_hw: tuple[int, int]
    origin: tuple[Channel, ...]         # origin of each OUTPUT channel
    in_origin: tuple[Channel, ...]      # origin of each INPUT channel (used by conv)
    conv: ConvSpec | None
    flops: float
    group: int | None                   # shared pruning group index (conv only)
    lock_reason: str | None             # None if prunable (conv only)

    @property
    def is_conv(self) -> bool:
        return self.conv is not None

    @property
    def prunable(self) -> bool:
        return self.is_conv and self.lock_reason is None


@dataclass(frozen=True)
class Graph:
    input_hw: tuple[int, int]
    input_channels: int
    layers: tuple[LayerInfo, ...]
    groups: tuple[tuple[int, ...], ...]     # each group: convs that must prune the same set of filters

    def convs(self) -> list[LayerInfo]:
        return [l for l in self.layers if l.is_conv]

    @property
    def param_count(self) -> int:
        return sum(l.conv.param_count for l in self.convs() if l.conv is not None)

    @property
    def bflops(self) -> float:
        return sum(l.flops for l in self.layers) / 1e9

    def consumers(self, conv: int) -> dict[int, list[tuple[int, int]]]:
        """Which convs consume the output of `conv`, and at which column (input channel).

        Returns {consumer conv: [(filter of `conv`, column in the consumer's weight), ...]}
        """
        out: dict[int, list[tuple[int, int]]] = {}
        for l in self.convs():
            for col, (src, f) in enumerate(l.in_origin):
                if src == conv:
                    out.setdefault(l.index, []).append((f, col))
        return out

    def table(self) -> str:
        rows = [f"{'#':>3} {'type':<14}{'sources':<18}{'in':>5}{'out':>5}{'grid (h x w)':>20}"
                f"{'params':>10}  prune"]
        for l in self.layers:
            src = ",".join("image" if s == IMAGE else str(s) for s in l.sources)
            grid = f"{l.out_hw[0]}x{l.out_hw[1]}"
            params = l.conv.param_count if l.conv is not None else ""
            if not l.is_conv:
                state = ""
            elif l.lock_reason:
                state = f"LOCKED: {l.lock_reason}"
            else:
                assert l.group is not None
                size = len(self.groups[l.group])
                state = "yes" + (f" (group {l.group}, {size} convs)" if size > 1 else "")
            rows.append(f"{l.index:>3} {l.type:<14}{src:<18}{l.in_channels:>5}{l.out_channels:>5}"
                        f"{grid:>20}{params:>10}  {state}")
        rows.append(f"\nTotal: {self.param_count:,} params, {self.bflops:.3f} BFLOPs, "
                    f"{sum(l.prunable for l in self.layers)}/{len(self.convs())} convs prunable")
        return "\n".join(rows)


class _UnionFind:
    def __init__(self):
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def build(cfg: Cfg) -> Graph:
    net = cfg.net
    channels = opt_int(net, -1, "channels", 3)
    hw = (opt_int(net, -1, "height"), opt_int(net, -1, "width"))
    image = Input(IMAGE, channels, hw, tuple((IMAGE, j) for j in range(channels)))

    done: list[Input] = []                  # output info of the layers processed so far
    inferred = []

    def layer_at(j: int) -> Input:
        return done[j]

    for i, section in enumerate(cfg.layers):
        prev = done[-1] if done else image
        inf = registry.get(section.type, i).infer(i, section, prev, layer_at)
        inferred.append((section.type, prev, inf))
        done.append(Input(i, inf.out_channels, inf.out_hw, inf.origin))

    # build groups (shortcut) and locks; if one conv is locked, its whole group is locked
    uf = _UnionFind()
    conv_ids = [i for i, (_, _, inf) in enumerate(inferred) if inf.conv is not None]
    for i in conv_ids:
        uf.find(i)
    locks: dict[int, str] = {}
    for _, _, inf in inferred:
        for a, b in inf.ties:
            uf.union(a, b)
        for c, why in inf.locks:
            locks.setdefault(c, why)

    roots = sorted({uf.find(i) for i in conv_ids})
    group_of_root = {r: g for g, r in enumerate(roots)}
    groups: list[list[int]] = [[] for _ in roots]
    for i in conv_ids:
        groups[group_of_root[uf.find(i)]].append(i)
    group_lock: dict[int, str] = {}
    for g, members in enumerate(groups):
        for m in members:
            if m in locks:
                group_lock[g] = locks[m] if len(members) == 1 else f"same group as conv {m} ({locks[m]})"
                break

    layers = []
    for i, (ltype, prev, inf) in enumerate(inferred):
        g = group_of_root[uf.find(i)] if inf.conv is not None else None
        reason = (locks.get(i) or group_lock.get(g)) if g is not None else None
        layers.append(LayerInfo(
            index=i, type=ltype, sources=inf.sources,
            in_channels=prev.channels, out_channels=inf.out_channels,
            in_hw=prev.hw, out_hw=inf.out_hw,
            origin=inf.origin, in_origin=prev.origin,
            conv=inf.conv, flops=inf.flops, group=g, lock_reason=reason,
        ))
    return Graph(hw, channels, tuple(layers), tuple(tuple(m) for m in groups))


# ---------------------------------------------------------------- parameters

@dataclass(frozen=True)
class ConvParams:
    """Parameters of one conv, exactly as stored in .weights.

    With BN   : bias = BN beta; scale, mean, var belong to BN.
    Without BN: bias is the conv bias; scale, mean, var = None.
    """
    bias: np.ndarray
    weight: np.ndarray                  # [filters, in_channels/groups, size, size]
    scale: np.ndarray | None = None
    mean: np.ndarray | None = None
    var: np.ndarray | None = None

    @property
    def has_bn(self) -> bool:
        return self.scale is not None


def read_params(graph: Graph, weights: wt.Weights) -> dict[int, ConvParams]:
    r = wt.WeightsReader(weights)
    params: dict[int, ConvParams] = {}
    for l in graph.convs():
        spec = l.conv
        assert spec is not None
        n = spec.filters
        tag = f"layer {l.index}"
        if spec.batch_normalize:
            bias = r.take(n, f"{tag} BN bias")
            scale = r.take(n, f"{tag} BN scale")
            mean = r.take(n, f"{tag} BN mean")
            var = r.take(n, f"{tag} BN var")
            weight = r.take(spec.weight_shape, f"{tag} weight")
            params[l.index] = ConvParams(bias, weight, scale, mean, var)
        else:
            bias = r.take(n, f"{tag} bias")
            weight = r.take(spec.weight_shape, f"{tag} weight")
            params[l.index] = ConvParams(bias, weight)
    r.finish()
    return params


def write_params(graph: Graph, params: dict[int, ConvParams], header: wt.Header) -> wt.Weights:
    w = wt.WeightsWriter(header)
    for l in graph.convs():
        p = params[l.index]
        spec = l.conv
        assert spec is not None
        if p.weight.shape != spec.weight_shape:
            raise GraphError(f"Layer {l.index}: weight has shape {p.weight.shape}, cfg expects {spec.weight_shape}")
        if p.has_bn != spec.batch_normalize:
            raise GraphError(f"Layer {l.index}: BN parameters do not match batch_normalize in cfg")
        parts = [p.bias, p.scale, p.mean, p.var] if p.has_bn else [p.bias]
        for a in parts:
            if a.shape != (spec.filters,):
                raise GraphError(f"Layer {l.index}: BN/bias vector has shape {a.shape}, expected ({spec.filters},)")
            w.put(a)
        w.put(p.weight)
    return w.build()