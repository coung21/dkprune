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
    ratios: dict[int, float] = field(default_factory=dict)

    def __post_init__(self):
        if not 0 <= self.ratio < 1:
            raise ValueError(f"ratio must be in [0, 1), received {self.ratio}")
        if self.round_to < 1 or self.min_keep < 1:
            raise ValueError("round_to and min_keep must be >= 1")
        for c, r in self.ratios.items():
            if not 0 <= r < 1:
                raise ValueError(f"ratios[{c}] must be in [0, 1), received {r}")

    def ratio_for(self, leader: int) -> float:
        return self.ratios.get(leader, self.ratio)


def target_keep(n: int, cfg: PruneConfig, ratio: float | None = None) -> int:
    keep = n * (1 - (cfg.ratio if ratio is None else ratio))
    keep = int(round(keep / cfg.round_to)) * cfg.round_to      
    keep = max(keep, cfg.min_keep, cfg.round_to)
    return min(keep, n)                                        


def plan(graph: Graph, params: dict[int, ConvParams], cfg: PruneConfig,
         only: set[int] | None = None) -> dict[int, np.ndarray]:
    score_fn = criteria.get(cfg.criterion)
    rng = np.random.default_rng(cfg.seed)
    keep: dict[int, np.ndarray] = {}

    for g, members in enumerate(graph.groups):
        if only is not None and g not in only:
            continue
        infos = [graph.layers[m] for m in members]
        if not infos[0].prunable or any(m in cfg.skip for m in members):
            continue
        conv = infos[0].conv
        if conv is None:
            continue
        n = conv.filters
        k = target_keep(n, cfg, cfg.ratio_for(members[0]))
        if k >= n:
            continue
        scores = score_fn([params[m] for m in members], rng)
        order = np.argsort(-scores, kind="stable")
        keep[g] = np.sort(order[:k])
    return keep