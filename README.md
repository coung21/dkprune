# dkprune

Iterative **filter pruning** for **Darknet / YOLO** models.

dkprune reads a Darknet `.cfg` + `.weights` pair, removes the least important
convolution filters, writes a smaller `.cfg` + `.weights` pair that Darknet loads
like any other model, and fine-tunes it with Darknet after each round. It repeats
until accuracy drops below a threshold you set, then tells you which round to keep.


```
round 0: 6,053,095 params  11.07 BFLOPs  mAP 80.00%
round 1: 4,266,455 params   7.85 BFLOPs  mAP 79.50%  (-0.50)
round 2: 3,085,535 params   6.03 BFLOPs  mAP 78.50%  (-1.50)   <- chosen
round 3: 2,133,943 params   4.51 BFLOPs  mAP 70.00%  (-10.00)  over threshold, stop
```

## Features

- **Architecture-agnostic.** Rules are written per layer type, not per model.
  YOLOv3, YOLOv4, YOLOv4-tiny, YOLOv7-tiny and custom cfgs built from the
  supported layers work without code changes.
- **Correct channel bookkeeping.** Every channel is tracked back to the filter that
  produced it, through `route`, `maxpool`, `upsample` and `shortcut`. Removing a filter
  removes exactly the right input columns in every consuming conv.
- **Minimal cfg edits.** Only `filters=` (and the `[net]` training values) are changed.
  Comments, anchors, `new_coords`, line endings and every other option are kept byte for byte.
- **Safe by default.** Unknown layers or options stop with a clear error instead of
  producing a broken model. Every cfg/weights pair is verified before and after fine-tuning.
- **Resumable.** All state lives on disk. Re-run the same command after an interruption
  and finished rounds are skipped.
- **Light.** The core needs only `numpy`. Training and evaluation are done by Darknet.

## Requirements

- Python 3.10+ with `numpy` and `pyyaml`
- [Darknet](https://codeberg.org/CCodeRun/darknet) (the Hank.ai / CCodeRun version),
  built with GPU support, reachable as a command (directly or through `docker run`)
- A trained model (`.cfg` + `.weights`) and its dataset in Darknet format,
  with a **validation set** (`valid=` in the `.data` file)

## Install

```bash
git clone <this repo> dkprune
cd dkprune
pip install -r requirements.txt
```

A suggested layout, with dkprune next to Darknet and the data outside both repos:

```
workspace/
├── darknet/     # Darknet sources: build and install only
├── dkprune/     # this repo
└── data/        # dataset, baseline model, config, runs
```

## Quick start

**1. Check that dkprune understands your model**

```bash
python run.py inspect model.cfg model.weights
```


**2. Write a config**

```bash
cp config.example.yaml config.yaml     # then edit the paths
```

Minimal example:

```yaml
data: /data/lego/mine.data            # must contain valid=
cfg: /data/baseline/LegoGears.cfg
weights: /data/baseline/LegoGears_best.weights
run_dir: /data/runs/lego_l2_r16
darknet_cmd: darknet

rounds: 4
max_map_drop: 0.02                    # stop when mAP drops more than 2 points

prune:
  ratio: 0.16                         # remove 16% of the filters per round
  skip: [0]                           # never prune the first conv

finetune:
  max_batches: 1000
  last_max_batches: 2000              # train the last round longer
  learning_rate: 0.001
  burn_in: 100
```

**3. Run**

```bash
python run.py prune config.yaml
```

**4. Use the result**

Open `run_dir/summary.md`. It shows every round and the chosen model
(`round_k.cfg` + `round_k_best.weights`). Measure latency on your target device:
the speed-up depends heavily on the hardware.

## Commands

| Command | What it does |
|---|---|
| `python run.py inspect <cfg> [weights]` | Print the network table; also verify the weights if given |
| `python run.py verify <cfg> <weights>` | Check that the pair matches and the weights are sane (exit 0 = ok) |
| `python run.py prune <config.yaml>` | Run iterative pruning |

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `data` | required | `.data` used for fine-tuning. Must have `valid=` so Darknet computes mAP and saves `_best.weights` |
| `cfg`, `weights` | required | Baseline model |
| `run_dir` | required | Where all results are written. Use one per experiment |
| `darknet_cmd` | `darknet` | Command used to call Darknet (see Docker below) |
| `workdir` | folder of `data` | Folder Darknet runs in. Relative paths in `.data` and image lists are resolved from here |
| `eval_data` | `data` | `.data` used to measure mAP each round, e.g. pointing `valid=` to a test set |
| `rounds` | `4` | Maximum number of prune + fine-tune rounds |
| `max_map_drop` | `0.02` | Stop when mAP is more than this below the baseline (0.02 = 2 points) |
| `prune.ratio` | `0.16` | Fraction of filters removed from each group per round, relative to the current count |
| `prune.criterion` | `l2` | `l2`, `l1`, `l2_raw` or `random` |
| `prune.round_to` | `8` | Kept filter counts are multiples of this (better for GPUs and NPUs) |
| `prune.min_keep` | `8` | Never keep fewer filters than this |
| `prune.skip` | `[]` | Conv layer indices that are never pruned |
| `finetune.max_batches` | `2000` | Fine-tuning iterations per round |
| `finetune.last_max_batches` | same | Iterations for the last round |
| `finetune.learning_rate` | `0.001` | Learning rate written to `[net]` |
| `finetune.burn_in` | `200` | Burn-in written to `[net]` |


## What a run produces

```
run_dir/
├── config.yaml                 copy of the config used
├── round_0.cfg / .weights      baseline (copied)
├── round_k.cfg / .weights      after pruning, before fine-tuning
├── round_k.data                .data with backup= backup_k/
├── backup_k/                   Darknet's weights for round k
├── round_k_best.weights        after fine-tuning
├── round_k.json                results of round k (a round is done when this exists)
├── logs/
│   ├── round_k_train.log       full Darknet output
│   ├── round_k_map.log
│   └── charts_k/chart*.png     Darknet training charts
└── summary.md                  table of all rounds and the chosen model
```

## How it works

Each round:

1. **Score** every filter with the L2 norm of its weights, after folding in the
   BatchNorm factor `gamma / sqrt(var + eps)` (the actual scale of the filter's output).
2. **Plan**: in every prunable group, keep the highest-scoring filters, rounded to a
   multiple of `round_to`.
3. **Cut**: remove the filter rows (and the four BatchNorm vectors), remove the
   matching input columns in every consumer, and change `filters=` in the cfg.
4. **Verify** the new cfg/weights pair.
5. **Fine-tune** with `darknet detector train -map -dont_show -clear`, using a fresh
   backup folder and `[net]` set for fine-tuning (`steps` at 80% and 90% of `max_batches`).
6. **Measure** mAP with `darknet detector map` and stop if the drop exceeds the threshold.


## Supported layers

`[convolutional]`, `[maxpool]`, `[upsample]`, `[route]` (including `groups`/`group_id`),
`[shortcut]` (without weights), `[yolo]`.


## Project layout

```
dkprune/
├── io/cfg.py, io/weights.py    read/write Darknet files, byte-exact
├── layers/                     one rule per layer type + registry
├── graph.py                    channels, origins, groups, locks; read/write conv params
├── criteria.py                 filter scores
├── policy.py                   which filters to keep
├── surgery.py                  cut rows, columns, BatchNorm; edit cfg
├── verify.py                   check a cfg/weights pair
├── darknet.py                  the only place that runs Darknet
├── config.py                   YAML config
├── loop.py                     prune -> verify -> fine-tune -> measure
└── summary.py                  per-round results and summary.md
run.py                          command line
```
