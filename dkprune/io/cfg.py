from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class CfgError(ValueError):
    """Raised when a configuration file is invalid."""


def read_text(path: str | Path) -> str:
    with open(path, "r", encoding="utf-8", errors="surrogateescape", newline="") as f:
        return f.read()


def split_lines(text: str) -> list[str]:
    parts = text.split("\n")
    lines = [p + "\n" for p in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines

def write_text(path: str | Path, lines: list[str] | tuple[str, ...]) -> None:
    with open(path, "w", encoding="utf-8", errors="surrogateescape", newline="") as f:
        f.write("".join(lines))


@dataclass(frozen=True)
class Option:
    key: str
    value: str
    line: int


@dataclass(frozen=True)
class Section:
    type: str # e.g. "convolutional", "route", "shortcut", etc.
    header_line: int
    end_line: int
    options: tuple[Option, ...]

    def find(self, key: str) -> Option | None:
        for option in self.options:
            if option.key == key:
                return option
        return None


@dataclass(frozen=True)
class Cfg:
    lines: tuple[str, ...]
    net: Section # [net] section
    layers: tuple[Section, ...] 


def _normalize(line: str) -> str:
    return "".join(line.split())


def parse(text: str) -> Cfg:

    lines = split_lines(text)
    sections: list[Section] = []

    cur_type: str | None = None
    cur_header = -1
    cur_opts: list[Option] = []

    def close(end: int) -> None:
        if cur_type is not None:
            sections.append(Section(cur_type, cur_header, end, tuple(cur_opts)))

    for i, raw in enumerate(lines):
        s = _normalize(raw)
        if not s or s[0] in ("#", ";"):
            continue

        if s[0] == "[":
            if not s.endswith("]"):
                raise CfgError(f"Invalid section header at line {i + 1}: {raw.strip()!r}")
            
            close(i)
            cur_type, cur_header, cur_opts = s[1:-1], i, []
            continue

        if cur_type is None:
            raise CfgError(f"Option outside of section at line {i + 1}: {raw.strip()!r}")

        key, sep, value = s.partition("=")
        if not sep or not key or not value:
            raise CfgError(f"Invalid option at line {i + 1}: {raw.strip()!r}")

        if any(opt.key == key for opt in cur_opts):
            raise CfgError(f"Duplicate option '{key}' at line {i + 1}: {raw.strip()!r}")

        cur_opts.append(Option(key, value, i))

    close(len(lines))

    if not sections or sections[0].type != "net":
        raise CfgError("Missing [net] section")

    return Cfg(tuple(lines), sections[0], tuple(sections[1:]))

def load(path: str | Path) -> Cfg:
    return parse(read_text(path))

def save(cfg: Cfg, path: str | Path) -> None:
    write_text(path, cfg.lines)