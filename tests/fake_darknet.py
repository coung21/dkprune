"""Stand-in for the darknet binary, for testing dkprune.darknet and dkprune.loop.

    python fake_darknet.py detector train -map -dont_show -nocolor <data> <cfg> <weights> -clear
    python fake_darknet.py detector map -nocolor <data> <cfg> <weights>

Environment:
    FAKE_STATE       file that counts calls per mode (required)
    FAKE_MAPS        comma-separated mAP values (0..1) returned by the 1st, 2nd, ... map call;
                     the last value repeats (required for map)
    FAKE_FAIL_TRAIN  number of the train call that should fail, e.g. "2"
"""

import os
import shutil
import sys
from pathlib import Path


def env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"fake darknet: environment variable {name} is not set")
    return value


def counter(name: str) -> int:
    """Count calls of `name` across processes; returns 1 for the first call."""
    state = Path(env("FAKE_STATE"))
    counts: dict[str, int] = {}
    if state.exists():
        for line in state.read_text().splitlines():
            k, v = line.split("=")
            counts[k] = int(v)
    counts[name] = counts.get(name, 0) + 1
    state.write_text("".join(f"{k}={v}\n" for k, v in counts.items()))
    return counts[name]


def read_backup(data: Path) -> Path | None:
    for line in data.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            key, _, value = s.partition("=")
            if key.strip() == "backup":
                return Path(value.strip())
    return None


def train(flags: set[str], data: Path, cfg: Path, weights: Path) -> int:
    n = counter("train")
    if not {"-map", "-dont_show", "-clear"} <= flags:
        print("missing -map / -dont_show / -clear")
        return 3
    if os.environ.get("FAKE_FAIL_TRAIN") == str(n):
        print("CUDA Error: out of memory (fake)")
        return 1
    backup = read_backup(data)
    if backup is None:
        print(f"no backup= in {data}")
        return 1
    if not weights.exists():
        print(f"Couldn't open file: {weights}")
        return 1
    # "training" leaves the weights unchanged
    shutil.copyfile(weights, backup / f"{cfg.stem}_last.weights")
    shutil.copyfile(weights, backup / f"{cfg.stem}_best.weights")
    Path("chart.png").write_bytes(b"png")
    print("Training complete.")
    return 0


def map_(n: int) -> int:
    maps = [float(x) for x in env("FAKE_MAPS").split(",")]
    value = maps[min(n, len(maps)) - 1]
    print("detections_count=123, unique_truth_count=45")
    print(f"-> mean average precision (mAP@0.50)=\x1b[1;34m{100 * value:.2f}%\x1b[0m")
    return 0


def main() -> int:
    args = sys.argv[1:]
    if len(args) < 2 or args[0] != "detector" or args[1] not in ("train", "map"):
        print("usage: detector train|map [flags] <data> <cfg> <weights>")
        return 1
    mode, rest = args[1], args[2:]
    flags = {a for a in rest if a.startswith("-")}
    files = [a for a in rest if not a.startswith("-")]
    if len(files) < 3:
        print(f"expected <data> <cfg> <weights>, got {files}")
        return 1
    data, cfg, weights = (Path(f) for f in files[:3])
    print(f"fake darknet: {mode} data={data} cfg={cfg} weights={weights}")

    if mode == "train":
        return train(flags, data, cfg, weights)
    return map_(counter("map"))


if __name__ == "__main__":
    sys.exit(main())
