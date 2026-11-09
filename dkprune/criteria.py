from __future__ import annotations
from typing import Callable
 
import numpy as np
 
from dkprune.graph import ConvParams

BN_EPS = 1e-5

Criterion = Callable[[list[ConvParams], np.random.Generator], np.ndarray]  # returns a 1D array of scores for each output channel

def _effective_weight(p: ConvParams) -> np.ndarray:
    """Weight [filters, -1] after folding batchnorm into conv: w * gamma / sqrt(var + eps)"""

    w = p.weight.reshape(p.weight.shape[0], -1).astype(np.float64)

    if p.has_bn:
        assert p.scale is not None and p.var is not None
        w = w * (p.scale / np.sqrt(p.var + BN_EPS))[:, None]

    return w

def l2(members: list[ConvParams], rng: np.random.Generator) -> np.ndarray:
    """L2 norm of each output channel's effective weight (BN folded)."""
    scores = np.zeros(members[0].weight.shape[0], dtype=np.float64)
    for p in members:
        scores += np.sqrt((_effective_weight(p) ** 2).sum(axis=1))
    return scores


def l1(members: list[ConvParams], rng: np.random.Generator) -> np.ndarray:
    """L1 norm of each output channel's effective weight (BN folded)."""
    scores = np.zeros(members[0].weight.shape[0], dtype=np.float64)
    for p in members:
        scores += np.abs(_effective_weight(p)).sum(axis=1)
    return scores

def l2_raw(members: list[ConvParams], rng: np.random.Generator) -> np.ndarray:
    """L2 norm of each output channel's raw weight (BN not folded)."""
    scores = np.zeros(members[0].weight.shape[0], dtype=np.float64)
    for p in members:
        scores += np.sqrt(
            (p.weight.reshape(p.weight.shape[0], -1).astype(np.float64) ** 2).sum(axis=1)
        )
    return scores

def random(members: list[ConvParams], rng: np.random.Generator) -> np.ndarray:
    """Random score for each output channel (use for testing)."""
    return rng.random(members[0].weight.shape[0])


CRITERIA: dict[str, Criterion] = {"l2": l2, "l1": l1, "random": random, "l2_raw": l2_raw}

def get(name: str) -> Criterion:
    """Get a criterion by name."""
    try:
        return CRITERIA[name]
    except KeyError:
        raise ValueError(f"Unknown criterion {name!r}, must be one of {list(CRITERIA.keys())}") from None

    