from __future__ import annotations
 
from dataclasses import dataclass, field
 
import numpy as np
 
from dkprune import criteria
from dkprune.graph import ConvParams, Graph


@dataclass(frozen=True)
class PruneConfig:
    ratio: float
    criterion: str = "l2"
    round_to: int = 8
    min_keep: int = 8
    skip: tuple[int, ...] = field(default=())
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.ratio < 1:
            raise ValueError(f"ratio must be in [0, 1), got {self.ratio}")
        if self.round_to < 1 or self.min_keep < 1:
            raise ValueError(f"round_to and min_keep must be >= 1, got {self.round_to} and {self.min_keep}")

def target_keep(n: int, cfg: PruneConfig) -> int:
    """Compute the number of channels to keep after pruning."""
    keep = n * (1 - cfg.ratio)
    keep = int(np.round(keep / cfg.round_to) * cfg.round_to)

    keep = max(keep, cfg.min_keep, cfg.round_to)
    keep = min(keep, n)
    return keep


def plan(graph: Graph, params: dict[int, ConvParams], cfg: PruneConfig) -> dict[int, np.ndarray]:
    """Compute a pruning plan for the given graph and parameters."""

    score_fn = criteria.get(cfg.criterion)
    rng = np.random.default_rng(cfg.seed)
    keep: dict[int, np.ndarray] = {}

    for g, members in enumerate(graph.groups):

        infos = [graph.layers[m] for m in members]
        if not infos[0].prunable or any(m in cfg.skip for m in members):
            # skip this group
            continue

        conv = infos[0].conv
        if conv is None:
            continue
        n = conv.filters
        k = target_keep(n, cfg)

        if k >= n:
            # nothing to prune
            continue

        scores = score_fn([params[m] for m in members], rng)

        order = np.argsort(-scores, kind="stable") 
        keep[g] = np.sort(order[:k])
    return keep