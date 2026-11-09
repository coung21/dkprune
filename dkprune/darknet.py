"""
darknet detector train -map -dont_show -nocolor <data> <cfg> <weights> -clear
darknet detector map -nocolor <data> <cfg> <weights>
"""

from __future__ import annotations
 
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


class DarknetError(RuntimeError):
    """Raised when a darknet command fails."""


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_MAP = re.compile(r"mean average precision \(mAP@([\d.]+)\)\s*=\s*([\d.]+)\s*%")


def parse_map(text: str) -> float:
    """Parse the mAP from the output of a darknet command."""
    found = _MAP.findall(_ANSI.sub("", text))
    if not found:
        raise DarknetError("Could not find mAP in darknet output")
    return float(found[-1][1]) / 100.0

def read_data_file(path: Path) -> dict[str, str]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k, _, v = s.partition("=")
            out[k.strip()] = v.strip()
    return out


def write_data_file(src: Path, dst: Path, backup: Path, valid: Path | None = None) -> None:
    """Copy file .data, change backup= (and valid= if exists) to new values."""
    new = {"backup": str(backup)}
    if valid is not None:
        new["valid"] = str(valid)
 
    lines, done = [], set()
    for line in Path(src).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        key = s.split("=", 1)[0].strip() if "=" in s and not s.startswith("#") else None
        if key in new:
            line = f"{key} = {new[key]}"
            done.add(key)
        lines.append(line)
    lines += [f"{k} = {v}" for k, v in new.items() if k not in done]
    Path(dst).write_text("\n".join(lines) + "\n", encoding="utf-8")



def find_weights(backup: Path, cfg_stem: str) -> Path:
    """Best weights file is <cfg_stem>_best.weights, if not found then <cfg_stem>_final.weights, if not found then <cfg_stem>_last.weights."""
    for suffix in ("_best", "_final", "_last"):
        p = Path(backup) / f"{cfg_stem}{suffix}.weights"
        if p.exists():
            return p
    raise DarknetError(f"Không thấy {cfg_stem}_best/_final/_last.weights trong {backup}. "
                       f"Train có lỗi không? File .data có valid= để Darknet tính mAP và lưu _best không?")


@dataclass(frozen=True)
class Darknet:
    cmd: tuple[str, ...]            # eg. ("darknet",) or ("docker", "run", ..., "darknet")
    workdir: Path                   # working directory for darknet commands
 
    @staticmethod
    def from_string(cmd: str, workdir: str | Path) -> "Darknet":
        parts = tuple(shlex.split(cmd))
        if not parts:
            raise ValueError("darknet_cmd rỗng")
        return Darknet(parts, Path(workdir))
 
    def _run(self, args: list[str], log: Path) -> str:
        log.parent.mkdir(parents=True, exist_ok=True)
        full = [*self.cmd, *args]
        with open(log, "w", encoding="utf-8", errors="replace") as f:
            f.write("$ " + shlex.join(full) + "\n\n")
            f.flush()
            try:
                proc = subprocess.run(full, cwd=self.workdir, stdout=f, stderr=subprocess.STDOUT)
            except FileNotFoundError:
                raise DarknetError(f"Cannot run '{self.cmd[0]}'. Check darknet_cmd in config.") from None
        text = log.read_text(encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            tail = "\n".join(text.splitlines()[-15:])
            raise DarknetError(f"Darknet exit with code {proc.returncode}. See {log}\n...\n{tail}")
        return text
 
    def train(self, data: Path, cfg: Path, weights: Path, log: Path, charts_to: Path | None = None) -> None:
        start = time.time()
        self._run(["detector", "train", "-map", "-dont_show", "-nocolor",
                   str(data), str(cfg), str(weights), "-clear"], log)
        if charts_to is not None:                    
            charts_to.mkdir(parents=True, exist_ok=True)
            for png in self.workdir.glob("chart*.png"):
                if png.stat().st_mtime >= start:
                    shutil.move(str(png), charts_to / png.name)
 
    def map(self, data: Path, cfg: Path, weights: Path, log: Path) -> float:
        return parse_map(self._run(["detector", "map", "-nocolor", str(data), str(cfg), str(weights)], log))