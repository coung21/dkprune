from __future__ import annotations
 
from dataclasses import dataclass
 
import numpy as np
 
from dkprune import graph as gr
from dkprune.graph import ConvParams, Graph
from dkprune.io import cfg as cfgio
from dkprune.io.weights import Header
from dkprune.layers.base import GraphError

_CHECK_HEADER = Header(0, 2, 5, 0)  # just to check shape

@dataclass(frozen=True)
class Pruned:
    cfg: cfgio.Cfg
    graph: Graph
    params: dict[int, ConvParams]


def _take(a: np.ndarray | None, idx: np.ndarray, axis: int = 0) -> np.ndarray | None:
    return None if a is None else np.take(a, idx, axis=axis)


def _keep_per_conv(graph: Graph, keep_by_group: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """{group: filter to keep} -> {conv: filter indices to keep}"""

    keep: dict[int, np.ndarray] = {}
    for g, idx in keep_by_group.items():
        idx = np.sort(np.asarray(idx, dtype=np.int64))
        for c in graph.groups[g]:
            info = graph.layers[c]
            if not info.prunable:
                raise GraphError(f"conv {c} in group {g} is not prunable for reason: {info.lock_reason}")

            if info.conv is None:
                raise GraphError(f"conv {c} in group {g} has no convolution parameters")
            n = info.conv.filters

            if idx.size == 0 or idx[0] < 0 or idx[-1] >= n or np.unique(idx).size != idx.size:
                raise GraphError(f"conv {c} in group {g}: invalid filter indices to keep: {idx}")

            keep[c] = idx

    return keep


def apply(cfg: cfgio.Cfg, graph: Graph, params: dict[int, ConvParams], keep_by_group: dict[int, np.ndarray]) -> Pruned:
    """Apply pruning to the given graph and parameters."""

    keep = _keep_per_conv(graph, keep_by_group)
    kept_sets = {c: set(idx.tolist()) for c, idx in keep.items()}

    new_params: dict[int, ConvParams] = {}

    for l in graph.convs():
        p = params[l.index]

        weight, bias, scale, mean, var = p.weight, p.bias, p.scale, p.mean, p.var

        #1. prune row (filter) of this conv
        rows = keep.get(l.index)
        if rows is not None:
            weight = _take(weight, rows, axis=0)
            bias, scale, mean, var = (_take(a, rows) for a in (bias, scale, mean, var))

        #2. prune column (input channel) comes from the previous filter pruned in the previous conv
        cols = [col for col, (src, f) in enumerate(l.in_origin)
                if src not in kept_sets or f in kept_sets[src]]
        if len(cols) != len(l.in_origin):
            if l.conv is not None and l.conv.groups > 1:
                raise GraphError(f"conv {l.index} is grouped, cannot prune input channels")

            weight = _take(weight, np.asarray(cols, dtype=np.int64), axis=1)

        if bias is None:
            raise GraphError(f"conv {l.index} has no bias parameters")

        new_params[l.index] = ConvParams(bias, np.ascontiguousarray(weight), scale, mean, var)

        # 3. edit cfg, rebuild graph, and return new params
    edits = [cfgio.Edit(c, "filters", int(idx.size)) for c, idx in keep.items()]
    new_cfg = cfgio.set_values(cfg, edits) if edits else cfg
    new_graph = gr.build(new_cfg)
    gr.write_params(new_graph, new_params, header=_CHECK_HEADER)
    return Pruned(new_cfg, new_graph, new_params)