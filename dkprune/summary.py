

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RoundResult:
    round: int                    
    params: int
    bflops: float
    map: float                    # 0..1
    cfg: str                      # cfg path of the pruned model (after fine-tune) of this round
    weights: str                  # weights path of the pruned model (after fine-tune) of this round

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "RoundResult":
        return RoundResult(**json.loads(path.read_text(encoding="utf-8")))


def choose(results: list[RoundResult], max_map_drop: float) -> RoundResult:
    base = results[0].map
    ok = [r for r in results if base - r.map <= max_map_drop + 1e-12]
    return max(ok, key=lambda r: r.round)


def table(results: list[RoundResult], max_map_drop: float) -> str:
    base = results[0]
    best = choose(results, max_map_drop)
    rows = ["| Round | Parameters | % origin | BFLOPs | mAP | Δ mAP | |",
            "|---:|---:|---:|---:|---:|---:|:---|"]
    for r in results:
        mark = "**choose**" if r.round == best.round else ("gone over" if base.map - r.map > max_map_drop else "")
        rows.append(f"| {r.round} | {r.params:,} | {100 * r.params / base.params:.1f}% | {r.bflops:.3f} | "
                    f"{100 * r.map:.2f}% | {100 * (r.map - base.map):+.2f} | {mark} |")
    return "\n".join(rows)


def write(run_dir: Path, results: list[RoundResult], max_map_drop: float) -> Path:
    best = choose(results, max_map_drop)
    text = (f"# Prune results\n\n{table(results, max_map_drop)}\n\n"
            f"Threshold: mAP should not drop more than {100 * max_map_drop:.2f} points from the original model.\n\n"
            f"Selected model (round {best.round}):\n\n- cfg: `{best.cfg}`\n- weights: `{best.weights}`\n")
    out = run_dir / "summary.md"
    out.write_text(text, encoding="utf-8")
    return out