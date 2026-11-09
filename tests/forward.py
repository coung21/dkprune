
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from dkprune.criteria import BN_EPS
from dkprune.graph import ConvParams, Graph
from dkprune.io.cfg import Cfg

ACTIVATIONS = {
    "linear": lambda x: x,
    "leaky": lambda x: np.where(x > 0, x, 0.1 * x),
    "relu": lambda x: np.maximum(x, 0),
    "logistic": lambda x: 1 / (1 + np.exp(-x)),
    "mish": lambda x: x * np.tanh(np.logaddexp(0, x)),
    "swish": lambda x: x / (1 + np.exp(-x)),
}


def activation(cfg: Cfg, i: int, default: str):
    opt = cfg.layers[i].find("activation")
    name = opt.value if opt else default
    if name not in ACTIVATIONS:
        raise NotImplementedError(f"activation '{name}' is not supported by the test forward")
    return name, ACTIVATIONS[name]


def _windows(xp: np.ndarray, k: int, stride: int, out_h: int, out_w: int) -> np.ndarray:
    """k x k windows over the last two axes: [c, out_h, out_w, k, k]."""
    # window (1, k, k) over all axes instead of axis=(1, 2), which numpy's type stubs reject
    win = sliding_window_view(xp, (1, k, k))[:, ::stride, ::stride, 0]
    return win[:, :out_h, :out_w]


def conv2d(x: np.ndarray, w: np.ndarray, stride: int, pad: int, groups: int) -> np.ndarray:
    c, h, ww = x.shape
    k = w.shape[2]
    out_h, out_w = (h + 2 * pad - k) // stride + 1, (ww + 2 * pad - k) // stride + 1
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    win = _windows(xp, k, stride, out_h, out_w)
    cg, og = c // groups, w.shape[0] // groups
    outs = [np.einsum("chwij,ocij->ohw", win[g * cg:(g + 1) * cg], w[g * og:(g + 1) * og], optimize=True)
            for g in range(groups)]
    return np.concatenate(outs, axis=0)


def maxpool(x: np.ndarray, size: int, stride: int, padding: int) -> np.ndarray:
    _, h, w = x.shape
    out_h, out_w = (h + padding - size) // stride + 1, (w + padding - size) // stride + 1
    left = padding // 2                                         # Darknet: offset = -pad / 2
    right_h = max(0, (out_h - 1) * stride + size - (h + left))
    right_w = max(0, (out_w - 1) * stride + size - (w + left))
    xp = np.pad(x, ((0, 0), (left, right_h), (left, right_w)), constant_values=-np.inf)
    win = _windows(xp, size, stride, out_h, out_w)
    return win.max(axis=(-1, -2))


def forward(cfg: Cfg, graph: Graph, params: dict[int, ConvParams], x: np.ndarray) -> list[np.ndarray]:
    outs: list[np.ndarray] = []
    yolo: list[np.ndarray] = []
    for l in graph.layers:
        s = cfg.layers[l.index]
        prev = outs[-1] if outs else x

        def get(key: str, default: int) -> int:
            opt = s.find(key)
            return int(opt.value) if opt is not None else default

        if l.conv is not None:
            p, spec = params[l.index], l.conv
            y = conv2d(prev, p.weight.astype(np.float64), spec.stride, spec.padding, spec.groups)
            if p.has_bn:
                assert p.scale is not None and p.mean is not None and p.var is not None
                y = (y - p.mean[:, None, None]) / np.sqrt(p.var[:, None, None] + BN_EPS)
                y = y * p.scale[:, None, None]
            y = y + p.bias[:, None, None]
            y = activation(cfg, l.index, "logistic")[1](y)
        elif l.type in ("maxpool", "max"):
            size = get("size", 1)
            y = maxpool(prev, size, get("stride", 1), get("padding", size - 1))
        elif l.type == "upsample":
            st = get("stride", 2)
            y = prev.repeat(st, axis=1).repeat(st, axis=2)
        elif l.type == "route":
            groups, gid = get("groups", 1), get("group_id", 0)
            parts = []
            for j in l.sources:
                o = outs[j]
                n = o.shape[0] // groups
                parts.append(o[gid * n:(gid + 1) * n])
            y = np.concatenate(parts, axis=0)
        elif l.type == "shortcut":
            y = prev + sum(outs[j] for j in l.sources[1:])
            y = activation(cfg, l.index, "linear")[1](y)
        elif l.type == "yolo":
            y = prev
            yolo.append(prev)
        else:
            raise NotImplementedError(l.type)
        outs.append(y)
    return yolo