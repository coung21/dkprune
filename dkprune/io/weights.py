from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

F32 = np.dtype("<f4")
I32 = np.dtype("<i4")
U32 = np.dtype("<u4")
U64 = np.dtype("<u8")

class WeightsError(ValueError):
    """Raised when a weights file is invalid."""

# HEADER
@dataclass(frozen=True)
class Header:
    major: int
    minor: int
    revision: int
    seen: int

    @property
    def seen_is_64bit(self) -> bool:
        return self.major * 10 + self.minor >= 2

    @property
    def nbytes(self) -> int:
        return 3 * 4 + (8 if self.seen_is_64bit else 4)

    def to_bytes(self) -> bytes:
        head = np.array([self.major, self.minor, self.revision], dtype=I32).tobytes()
        seen = np.array([self.seen], dtype=U64 if self.seen_is_64bit else U32).tobytes()
        return head + seen

    @staticmethod
    def from_bytes(buf: bytes) -> "Header":
        if len(buf) < 12:
            raise WeightsError(f"Header is too short: {len(buf)} bytes")
        
        major, minor, revision = (int(v) for v in np.frombuffer(buf, dtype=I32, count=3))
        is64 = major * 10 + minor >= 2
        need = 20 if is64 else 16
        if len(buf) < need:
            raise WeightsError(f"Header is too short: {len(buf)} bytes, need {need} bytes")

        seen = int(np.frombuffer(buf, dtype=U64 if is64 else U32, count=1, offset=12)[0])

        return Header(major, minor, revision, seen)


# WEIGHTS
@dataclass(frozen=True)
class Weights:
    header: Header
    data: np.ndarray

    @property
    def count(self) -> int:
        return self.data.size


def _readonly(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a

def from_bytes(buf: bytes) -> Weights:
    header = Header.from_bytes(buf)
    body = len(buf) - header.nbytes
    if body % 4 != 0:
        raise WeightsError(f"Body size {body} is not a multiple of 4 bytes")
    data = np.frombuffer(buf, dtype=F32, offset=header.nbytes).copy()
    return Weights(header, _readonly(data))


def to_bytes(w: Weights) -> bytes:
    return w.header.to_bytes() + np.ascontiguousarray(w.data, dtype=F32).tobytes()

def load(path: str | Path) -> Weights:
    return from_bytes(Path(path).read_bytes())

def save(w: Weights, path: str | Path) -> None:
    Path(path).write_bytes(to_bytes(w))



# READER & WRITER

class WeightsReader:

    def __init__(self, w: Weights) -> None:
        self._data = w
        self._pos = 0

    @property
    def position(self) -> int:
        return self._pos

    @property
    def remaining(self) -> int:
        return self._data.count - self._pos

    def take(self, shape: int | tuple[int, ...], what = "") -> np.ndarray:
        """take next `shape` elements from the weights data, advancing the position"""
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        n = int(np.prod(shape))

        if n > self.remaining:
            raise WeightsError(f"Not enough weights remaining: {self.remaining} < {n}")

        out = self._data.data[self._pos:self._pos+n].reshape(shape).copy()
        self._pos += n

        return _readonly(out)

    def finish(self) -> None:
        """finish reading weights, raising an error if any remain"""
        if self.remaining > 0:
            raise WeightsError(f"Not all weights were consumed: {self.remaining} remaining")


class WeightsWriter:
    def __init__(self, header: Header) -> None:
        self._header = header
        self._parts: list[np.ndarray] = []

    def put(self, a: np.ndarray) -> None:
        a = np.asanyarray(a)
        if not np.issubdtype(a.dtype, np.floating):
            raise WeightsError(f"Cannot write non-floating array of dtype {a.dtype}")

        self._parts.append(np.ascontiguousarray(a, dtype=F32).ravel())


    def build(self) -> Weights:
        data = np.concatenate(self._parts) if self._parts else np.empty(0, dtype=F32)
        return Weights(self._header, _readonly(data))