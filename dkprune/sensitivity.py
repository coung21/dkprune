

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from dkprune import graph as gr
from dkprune import policy, surgery
from dkprune.darknet import Darknet, write_data_file
from dkprune.io import cfg as cfgio
from dkprune.io import weights as wt

Log = Callable[[str], None]


@dataclass(frozen=True)
class SensitivityConfig:
    ratios: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5)  
    tolerance: float = 0.01       
    max_ratio: float = 0.3      

    def __post_init__(self):
        if not self.ratios or any(not 0 < r < 1 for r in self.ratios):
            raise ValueError("sensitivity.ratios not in the list of (0, 1)")
        if not 0 <= self.tolerance < 1 or not 0 <= self.max_ratio < 1:
            raise ValueError("sensitivity.tolerance and max_ratio must be in [0, 1)")


def _key(leader: int, ratio: float) -> str:
    return f"{leader}:{ratio:g}"


def analyze(cfg_path: Path, weights_path: Path, data: Path, prune: policy.PruneConfig,
            sc: SensitivityConfig, dk: Darknet, out_dir: Path, baseline_map: float,
            log: Log = print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    store = out_dir / "results.json"
    results = json.loads(store.read_text()) if store.exists() else {"baseline": baseline_map, "entries": {}}
    entries: dict[str, dict] = results["entries"]

    c = cfgio.load(cfg_path)
    w = wt.load(weights_path)
    g = gr.build(c)
    params = gr.read_params(g, w)
    eval_data = out_dir / "eval.data"
    write_data_file(data, eval_data, out_dir / "backup")
    tmp_cfg, tmp_weights = out_dir / "trial.cfg", out_dir / "trial.weights"

    groups = [gid for gid, m in enumerate(g.groups)
              if g.layers[m[0]].prunable and not any(x in prune.skip for x in m)]
    total = len(groups) * len(sc.ratios)
    n_done = sum(_key(g.groups[gid][0], r) in entries for gid in groups for r in sc.ratios)
    log(f"[sensitivity] {len(groups)} groups x {len(sc.ratios)} ratios = {total} evaluations, {n_done} done")

    for gid in groups:
        leader = g.groups[gid][0]
        conv = g.layers[leader].conv
        if conv is None:
            raise ValueError(f"prunable layer {leader} has no convolution")
        n = conv.filters
        for r in sc.ratios:
            key = _key(leader, r)
            if key in entries:
                continue
            trial = policy.PruneConfig(ratio=r, criterion=prune.criterion, round_to=prune.round_to,
                                       min_keep=prune.min_keep, seed=prune.seed)
            keep = policy.plan(g, params, trial, only={gid})
            if gid not in keep:                       
                m, kept = baseline_map, n
            else:
                out = surgery.apply(c, g, params, keep)
                cfgio.save(out.cfg, tmp_cfg)
                wt.save(gr.write_params(out.graph, out.params, w.header), tmp_weights)
                m = dk.map(eval_data, tmp_cfg, tmp_weights, out_dir / "logs" / f"{leader}_{r:g}.log")
                kept = int(keep[gid].size)
            entries[key] = {"leader": leader, "ratio": r, "filters": n, "kept": kept, "map": m}
            store.write_text(json.dumps(results, indent=1))          # lưu ngay: chạy tiếp được
            n_done += 1
            log(f"[sensitivity {n_done}/{total}] conv {leader}: {n} -> {kept} filter, "
                f"mAP {100 * m:.2f}% ({100 * (m - baseline_map):+.2f})")

    for p in (tmp_cfg, tmp_weights):
        p.unlink(missing_ok=True)
    return results


def derive_ratios(results: dict, sc: SensitivityConfig) -> dict[int, float]:
    base = results["baseline"]
    by_leader: dict[int, list[dict]] = {}
    for e in results["entries"].values():
        by_leader.setdefault(e["leader"], []).append(e)
    out = {}
    for leader, es in by_leader.items():
        ok = [e["ratio"] for e in es
              if e["ratio"] <= sc.max_ratio + 1e-12 and base - e["map"] <= sc.tolerance + 1e-12]
        out[leader] = max(ok) if ok else 0.0
    return out


def table(results: dict, sc: SensitivityConfig) -> str:
    base = results["baseline"]
    chosen = derive_ratios(results, sc)
    ratios = sorted(sc.ratios)
    head = "| Conv | Filter | " + " | ".join(f"cut {100 * r:g}%" for r in ratios) + " | Percentage |"
    rows = [head, "|---:|---:|" + "---:|" * len(ratios) + "---:|"]
    by_leader: dict[int, dict[float, dict]] = {}
    for e in results["entries"].values():
        by_leader.setdefault(e["leader"], {})[e["ratio"]] = e
    for leader in sorted(by_leader):
        es = by_leader[leader]
        n = next(iter(es.values()))["filters"]
        cells = []
        for r in ratios:
            e = es.get(r)
            cells.append("" if e is None else
                         ("Not changed" if e["kept"] == n else f"{100 * (e['map'] - base):+.2f}"))
        rows.append(f"| {leader} | {n} | " + " | ".join(cells) + f" | {100 * chosen.get(leader, 0):g}% |")
    return "\n".join(rows)


def write_report(out_dir: Path, results: dict, sc: SensitivityConfig) -> Path:
    text = (f"# Sensitivity Analysis\n\norgin mAP: {100 * results['baseline']:.2f}%. "
            f"Each cell represents the drop in mAP (in points) when only this group is pruned, without fine-tuning.\n"
            f"Percentage: the highest pruning ratio (not exceeding {100 * sc.max_ratio:g}%) that results in a mAP drop not exceeding "
            f"{100 * sc.tolerance:g} points.\n\n{table(results, sc)}\n")
    p = out_dir / "sensitivity.md"
    p.write_text(text, encoding="utf-8")
    return p