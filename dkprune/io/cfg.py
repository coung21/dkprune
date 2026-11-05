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


# EDIT VALUES

@dataclass(frozen=True)
class Edit:
    """One edit: set `key=value` in layer `layer` (None for [net] section)."""
    layer: int | None
    key: str
    value: str | int | float
    insert: bool = False  # if True, insert the key if it doesn't exist


def _split_eol(line: str) -> tuple[str, str]:
    """filters=64\\r\\n -> ("filters=64", "\\r\\n")"""
    body = line.rstrip("\r\n")
    return body, line[len(body):]

def _replace_value(line: str, value: str) -> str:
    body, eol = _split_eol(line)
    left, _, right = body.partition("=")
    lead = right[: len(right) - len(right.lstrip())]
    trail = right[len(right.rstrip()):]
    return f"{left}={lead}{value}{trail}{eol}"

def _check_value(edit: Edit) -> str:
    value = str(edit.value)
    if not _normalize(value):
        raise ValueError(f"empty value for {edit.key} in layer {edit.layer}")
    if "\n" in value or "\r" in value:
        raise ValueError(f"newlines not allowed in value for {edit.key} in layer {edit.layer}")
    if not edit.key or _normalize(edit.key) != edit.key or "=" in edit.key:
        raise ValueError(f"invalid key {edit.key!r} in layer {edit.layer}")
    return value

def set_values(cfg: Cfg, edits: list[Edit]) -> Cfg:
    lines = list(cfg.lines)
    inserts: list[tuple[int, str]] = []
    seen: set[tuple[int| None, str]] = set()

    for edit in edits:
        value = _check_value(edit)
        if (edit.layer, edit.key) in seen:
            raise ValueError(f"duplicate edit for {edit.key} in layer {edit.layer}")
        seen.add((edit.layer, edit.key))

        if edit.layer is None:
            section = cfg.net
        elif 0 <= edit.layer < len(cfg.layers):
            section = cfg.layers[edit.layer]
        else:
            raise ValueError(f"invalid layer {edit.layer} for {edit.key}")

        opt = section.find(edit.key)
        if opt is not None:
            lines[opt.line] = _replace_value(lines[opt.line], value)
        elif edit.insert:
            anchor = section.options[-1].line if section.options else section.header_line
            inserts.append((anchor, f"{edit.key}={value}"))
        else:
            where = "[net]" if edit.layer is None else f"layer {edit.layer} [{section.type}]"
            raise ValueError(f"option {edit.key} not found in {where}")

    if inserts:
        default_eol = _split_eol(lines[0])[1] or "\n"
        for anchor, text in sorted(inserts, key=lambda x: x[0], reverse=True):
            body, eol = _split_eol(lines[anchor])
            if not eol:
                lines[anchor] = body + default_eol
                eol = default_eol
            lines.insert(anchor + 1, text+eol)

    return parse("".join(lines))


def load(path: str | Path) -> Cfg:
    return parse(read_text(path))

def save(cfg: Cfg, path: str | Path) -> None:
    write_text(path, cfg.lines)