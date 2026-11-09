"""Iterative pruning loop: prune -> verify -> fine-tune (Darknet) -> mesure -> loop.

Every round k (1..rounds) will produce:
    round_0.cfg, round_0.weights  
    round_k.cfg, round_k.weights                 
    round_k.data                             
    backup_k/                                    
    round_k_best.weights                          
    round_k.json                                  
    logs/                                         
    summary.md
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

from dkprune import graph as gr
from dkprune import policy, summary, surgery, verify
from dkprune.config import RunConfig
from dkprune.darknet import Darknet, find_weights, read_data_file, write_data_file
from dkprune.io import cfg as cfgio
from dkprune.io import weights as wt
from dkprune.summary import RoundResult

Log = Callable[[str], None]


def net_edits(max_batches: int, learning_rate: float, burn_in: int) -> list[cfgio.Edit]:
    s1, s2 = int(max_batches * 0.8), int(max_batches * 0.9)
    return [
        cfgio.Edit(None, "max_batches", max_batches, insert=True),
        cfgio.Edit(None, "learning_rate", learning_rate, insert=True),
        cfgio.Edit(None, "burn_in", burn_in, insert=True),
        cfgio.Edit(None, "policy", "steps", insert=True),
        cfgio.Edit(None, "steps", f"{s1},{s2}", insert=True),
        cfgio.Edit(None, "scales", ".1,.1", insert=True),
    ]


def _measure(rc: RunConfig, dk: Darknet, k: int, cfg_path: Path, weights_path: Path) -> RoundResult:
    run = rc.run_dir
    data = rc.eval_data or rc.data
    eval_copy = run / f"eval_{k}.data"                
    write_data_file(data, eval_copy, run / f"backup_{k}")
    m = dk.map(eval_copy, cfg_path, weights_path, run / "logs" / f"round_{k}_map.log")
    g = gr.build(cfgio.load(cfg_path))
    return RoundResult(k, g.param_count, g.bflops, m, str(cfg_path), str(weights_path))


def _check_data_has_valid(rc: RunConfig) -> None:
    if "valid" not in read_data_file(rc.data):
        raise ValueError(f"{rc.data} doesnt  'valid=': Darknet need a valid= line to evaluate mAP (or set eval_data in config)")


def run(rc: RunConfig, dk: Darknet | None = None, log: Log = print) -> list[RoundResult]:
    dk = dk or Darknet.from_string(rc.darknet_cmd, rc.darknet_workdir)
    _check_data_has_valid(rc)
    run_dir = rc.run_dir
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    results: list[RoundResult] = []

    # round 0: original model
    done0 = run_dir / "round_0.json"
    if done0.exists():
        results.append(RoundResult.load(done0))
        log(f"[round 0]: mAP {100 * results[0].map:.2f}%")
    else:
        shutil.copyfile(rc.cfg, run_dir / "round_0.cfg")
        shutil.copyfile(rc.weights, run_dir / "round_0.weights")
        log(f"[round 0] {verify.check_files(run_dir / 'round_0.cfg', run_dir / 'round_0.weights')}")
        r0 = _measure(rc, dk, 0, run_dir / "round_0.cfg", run_dir / "round_0.weights")
        r0.save(done0)
        results.append(r0)
        log(f"[round 0] original mAP {100 * r0.map:.2f}%")

    # loop rounds 1..rc.rounds
    for k in range(1, rc.rounds + 1):
        done = run_dir / f"round_{k}.json"
        if done.exists():
            r = RoundResult.load(done)
            results.append(r)
            log(f"[round {k}]: mAP {100 * r.map:.2f}%")
        else:
            prev = results[-1]
            r = _prune_round(rc, dk, k, Path(prev.cfg), Path(prev.weights), log)
            r.save(done)
            results.append(r)

        drop = results[0].map - r.map
        log(f"[round {k}] parameters {r.params:,} ({100 * r.params / results[0].params:.1f}% of original), "
            f"mAP {100 * r.map:.2f}% (drop {100 * drop:.2f} points)")
        if drop > rc.max_map_drop:
            log(f"[round {k}] mAP drop exceeds threshold {100 * rc.max_map_drop:.2f} points: stopping")
            break

    out = summary.write(run_dir, results, rc.max_map_drop)
    best = summary.choose(results, rc.max_map_drop)
    log(f"Done. Selected round {best.round}: {best.cfg} + {best.weights}\nSummary: {out}")
    return results


def _prune_round(rc: RunConfig, dk: Darknet, k: int, cfg_path: Path, weights_path: Path, log: Log) -> RoundResult:
    run = rc.run_dir
    c = cfgio.load(cfg_path)
    w = wt.load(weights_path)
    g = gr.build(c)
    params = gr.read_params(g, w)

    # 1. prune
    keep = policy.plan(g, params, rc.prune)
    if not keep:
        raise RuntimeError(f"Round {k}: policy.plan returned empty keep list: cannot prune further")
    out = surgery.apply(c, g, params, keep)

    # 2. [net] for fine-tune: set max_batches, learning_rate, burn_in
    ft = rc.finetune
    mb = ft.last_max_batches if (k == rc.rounds and ft.last_max_batches) else ft.max_batches
    new_cfg = cfgio.set_values(out.cfg, net_edits(mb, ft.learning_rate, ft.burn_in))
    cfg_k, weights_k = run / f"round_{k}.cfg", run / f"round_{k}.weights"
    cfgio.save(new_cfg, cfg_k)
    wt.save(gr.write_params(out.graph, out.params, w.header), weights_k)
    log(f"[round {k}] pruned {len(keep)} groups: {g.param_count:,} -> {out.graph.param_count:,} parameters; "
        f"{verify.check_files(cfg_k, weights_k)}")

    # 3. fine-tune Darknet
    backup = run / f"backup_{k}"
    if backup.exists():                       # còn sót từ lần chạy dở trước: xóa để không lấy nhầm weights cũ
        shutil.rmtree(backup)
    backup.mkdir()
    data_k = run / f"round_{k}.data"
    write_data_file(rc.data, data_k, backup)
    log(f"[round {k}] fine-tune {mb} rounds... (log: {run / 'logs' / f'round_{k}_train.log'})")
    dk.train(data_k, cfg_k, weights_k, run / "logs" / f"round_{k}_train.log", charts_to=run / "logs" / f"charts_{k}")

    # 4. best weigts, verify, measure
    best = run / f"round_{k}_best.weights"
    shutil.copyfile(find_weights(backup, cfg_k.stem), best)
    rep = verify.check_files(cfg_k, best)
    if not rep.ok:
        raise RuntimeError(f"Round {k}: weights after fine-tune have issues:\n{rep}")
    return _measure(rc, dk, k, cfg_k, best)