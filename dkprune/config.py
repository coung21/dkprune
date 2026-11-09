

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from dkprune.policy import PruneConfig


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class FinetuneConfig:
    max_batches: int = 2000
    last_max_batches: int | None = None
    learning_rate: float = 0.001
    burn_in: int = 200


@dataclass(frozen=True)
class RunConfig:
    data: Path                              
    cfg: Path                                
    weights: Path                            
    run_dir: Path                           
    darknet_cmd: str = "darknet"
    workdir: Path | None = None              
    eval_data: Path | None = None           
    rounds: int = 4
    max_map_drop: float = 0.02              
    prune: PruneConfig = field(default_factory=lambda: PruneConfig(ratio=0.16))
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)

    @property
    def darknet_workdir(self) -> Path:
        return self.workdir or self.data.parent


def _path(base: Path, value) -> Path:
    p = Path(str(value)).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def load(path: str | Path) -> RunConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.parent.resolve()                 

    try:
        p = raw.get("prune", {}) or {}
        f = raw.get("finetune", {}) or {}
        cfg = RunConfig(
            data=_path(base, raw["data"]),
            cfg=_path(base, raw["cfg"]),
            weights=_path(base, raw["weights"]),
            run_dir=_path(base, raw["run_dir"]),
            darknet_cmd=str(raw.get("darknet_cmd", "darknet")),
            workdir=_path(base, raw["workdir"]) if raw.get("workdir") else None,
            eval_data=_path(base, raw["eval_data"]) if raw.get("eval_data") else None,
            rounds=int(raw.get("rounds", 4)),
            max_map_drop=float(raw.get("max_map_drop", 0.02)),
            prune=PruneConfig(
                ratio=float(p.get("ratio", 0.16)),
                criterion=str(p.get("criterion", "l2")),
                round_to=int(p.get("round_to", 8)),
                min_keep=int(p.get("min_keep", 8)),
                skip=tuple(int(x) for x in p.get("skip", []) or []),
                seed=int(p.get("seed", 0)),
            ),
            finetune=FinetuneConfig(
                max_batches=int(f.get("max_batches", 2000)),
                last_max_batches=int(f["last_max_batches"]) if f.get("last_max_batches") else None,
                learning_rate=float(f.get("learning_rate", 0.001)),
                burn_in=int(f.get("burn_in", 200)),
            ),
        )
    except KeyError as e:
        raise ConfigError(f"{path}: required key {e} is missing") from None
    except (TypeError, ValueError) as e:
        raise ConfigError(f"{path}: invalid value: {e}") from None

    for name in ("data", "cfg", "weights"):
        if not getattr(cfg, name).exists():
            raise ConfigError(f"{path}: cannot find file {name} = {getattr(cfg, name)}")
    if cfg.eval_data and not cfg.eval_data.exists():
        raise ConfigError(f"{path}: cannot find file eval_data = {cfg.eval_data}")
    if cfg.rounds < 1:
        raise ConfigError(f"{path}: rounds must be >= 1")
    if not 0 <= cfg.max_map_drop < 1:
        raise ConfigError(f"{path}: max_map_drop must be in [0, 1)")
    if cfg.finetune.max_batches < 1 or cfg.finetune.burn_in < 0:
        raise ConfigError(f"{path}: finetune.max_batches must be >= 1, burn_in >= 0")
    return cfg