from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from dkprune.io.cfg import Section

Channel = tuple[int, int]         
IMAGE = -1


class GraphError(ValueError):
    """Raised when a graph (cfg) is invalid."""



@dataclass(frozen=True)
class Input:
    index: int
    channels: int
    hw: tuple[int, int]
    origin: tuple[Channel, ...]

@dataclass(frozen=True)
class ConvSpec:
    filters: int
    in_channels: int
    size: int
    stride: int
    padding: int
    groups: int
    batch_normalize: bool

    @property
    def weight_shape(self) -> tuple[int, int, int, int]:
        return (self.filters, self.in_channels // self.groups, self.size, self.size)

    @property
    def param_count(self) -> int:
        n_w = self.filters * (self.in_channels // self.groups) * self.size * self.size
        return n_w + (4 if self.batch_normalize else 1) * self.filters

@dataclass(frozen=True)
class Inferred:
    sources: tuple[int, ...]
    out_channels: int
    out_hw: tuple[int, int]
    origin: tuple[Channel, ...]
    conv: ConvSpec | None = None
    flops: float = 0.0

    locks: tuple[tuple[int, str], ...] = field(default=())
    ties: tuple[tuple[int, int], ...] = field(default=())

LayerAt = Callable[[int], Input]

class LayerRule(Protocol):
    names: tuple[str, ...]

    def infer(self, index: int, section: Section, prev: Input, layer_at: LayerAt) -> Inferred:
        ...


def opt_int(section: Section, index: int, key: str, default: int | None = None) -> int:
    opt = section.find(key)
    if opt is None:
        if default is None:
            raise GraphError(f"Layer {index} [{section.type}] missing required key {key}") 
        return default
    try:
        return int(opt.value)
    except ValueError:
        raise GraphError(f"invalid {key}={opt.value!r} in layer {index} at line {opt.line + 1}") from None

def reject(section: Section, index: int, keys: dict[str, str]) -> None:
    """Raises GraphError if any of the keys are present in the section, with a message indicating the expected value."""
    for key, allowed in keys.items():
        opt = section.find(key)
        if opt is not None:
            raise GraphError(f"unexpected {key}={opt.value!r} in layer {index} at line {opt.line + 1}, expected {allowed}") from None

def resolve(index: int, raw: str, section: Section) -> int:

    try:
        v = int(raw)
    except ValueError:
        raise GraphError(f"invalid layer {index} {section.type}={raw!r}, expected integer") from None

    j = index + v if v < 0 else v
    if not 0 <= j < index:
        raise GraphError(f"Layer {index} [{section.type}] has invalid 'from={raw}' (resolved to {j}), must be in [0, {index})")
    return j

def producers(origin: tuple[Channel, ...]) -> list[int]:
    """All the `conv` layers that produced the channels in `origin`, in order of appearance, except for IMAGE channels."""
    seen: dict[int, None] = {}
    for conv, _ in origin:
        if conv != IMAGE:
            seen.setdefault(conv)

    return list(seen)