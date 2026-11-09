from __future__ import annotations
 
from dataclasses import dataclass
from pathlib import Path
 
import numpy as np
 
from dkprune import graph as gr
from dkprune.io import cfg as cfgio
from dkprune.io import weights as wt
 
 
@dataclass(frozen=True)
class Report:
    param_count: int
    convs: int
    bflops: float
    nonfinite: int                  # number of NaN/Inf in weights
    bad_variance: list[int]         # conv has BN var <= 0
 
    @property
    def ok(self) -> bool:
        return self.nonfinite == 0 and not self.bad_variance
 
    def __str__(self) -> str:
        lines = [f"{self.convs} conv, {self.param_count:,} parameters, {self.bflops:.3f} BFLOPs: cfg and weights match"]
        if self.nonfinite:
            lines.append(f"WARNING: {self.nonfinite} NaN/Inf values in weights")
        if self.bad_variance:
            lines.append(f"WARNING: BN var <= 0 in conv {self.bad_variance}")
        return "\n".join(lines)
 
 
def check(cfg: cfgio.Cfg, weights: wt.Weights) -> Report:
    """Check if the given cfg and weights match; raise an error if they don't, otherwise return a Report."""
    graph = gr.build(cfg)
    params = gr.read_params(graph, weights)          
    bad_var = [
        c for c, p in params.items()
        if p.has_bn and p.var is not None and not (p.var > 0).all()
    ]
    return Report(
        param_count=graph.param_count,
        convs=len(params),
        bflops=graph.bflops,
        nonfinite=int((~np.isfinite(weights.data)).sum()),
        bad_variance=bad_var,
    )
 
 
def check_files(cfg_path: str | Path, weights_path: str | Path) -> Report:
    return check(cfgio.load(cfg_path), wt.load(weights_path))