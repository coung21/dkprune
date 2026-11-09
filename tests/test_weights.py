import os
from pathlib import Path

import numpy as np
import pytest

from dkprune.io import weights as wt


REAL = os.environ.get("DKPRUNE_WEIGHTS")

REAL_COUNT = os.environ.get("DKPRUNE_WEIGHTS_COUNT")


def make_bytes(major=0, minor=2, revision=5, seen=123456789, values=(0.5, -1.0, 3.25)):
    h = wt.Header(major, minor, revision, seen)
    return h.to_bytes() + np.array(values, dtype="<f4").tobytes()



def test_header_new_format_is_20_bytes():
    buf = make_bytes()
    w = wt.from_bytes(buf)
    assert w.header == wt.Header(0, 2, 5, 123456789)
    assert w.header.nbytes == 20
    assert w.data.tolist() == [0.5, -1.0, 3.25]


def test_header_old_format_seen_is_32bit():
    buf = make_bytes(major=0, minor=1, revision=0, seen=1000)
    w = wt.from_bytes(buf)
    assert w.header.nbytes == 16
    assert w.header.seen == 1000
    assert w.count == 3


def test_seen_larger_than_32bit():
    big = 2**40 + 7
    assert wt.from_bytes(make_bytes(seen=big)).header.seen == big



@pytest.mark.parametrize("minor", [1, 2])
def test_roundtrip_byte_for_byte(minor, tmp_path):
    buf = make_bytes(minor=minor, values=np.random.default_rng(0).standard_normal(1000))
    src = tmp_path / "a.weights"
    src.write_bytes(buf)
    out = tmp_path / "b.weights"
    wt.save(wt.load(src), out)
    assert out.read_bytes() == buf


def test_data_is_float32_and_readonly():
    w = wt.from_bytes(make_bytes())
    assert w.data.dtype == np.dtype("<f4")
    with pytest.raises(ValueError):
        w.data[0] = 9.0


@pytest.mark.parametrize("buf, msg", [
    (b"\x00" * 8, "too short"),
    (np.array([0, 2, 5], "<i4").tobytes() + b"\x00" * 4, "too short"),
    (make_bytes() + b"\x01\x02", "not a multiple of 4"),
])
def test_broken_files(buf, msg):
    with pytest.raises(wt.WeightsError, match=msg):
        wt.from_bytes(buf)



def test_reader_takes_in_order_with_shape():
    w = wt.from_bytes(make_bytes(values=np.arange(10, dtype="<f4")))
    r = wt.WeightsReader(w)
    assert r.take(4).tolist() == [0, 1, 2, 3]
    block = r.take((2, 3))
    assert block.shape == (2, 3) and block.tolist() == [[4, 5, 6], [7, 8, 9]]
    r.finish()


def test_reader_not_enough_data():
    r = wt.WeightsReader(wt.from_bytes(make_bytes()))
    with pytest.raises(wt.WeightsError, match="remaining: 3 < 4"):
        r.take(4, "conv 0")


def test_reader_leftover_data():
    r = wt.WeightsReader(wt.from_bytes(make_bytes()))
    r.take(2)
    with pytest.raises(wt.WeightsError, match="1 remaining"):
        r.finish()


def test_reader_returns_copy():
    w = wt.from_bytes(make_bytes())
    a = wt.WeightsReader(w).take(3)
    assert not np.shares_memory(a, w.data)


def test_writer_roundtrip():
    src = wt.from_bytes(make_bytes(values=np.arange(10, dtype="<f4")))
    r, wr = wt.WeightsReader(src), wt.WeightsWriter(src.header)
    for shape in (4, (2, 3)):
        wr.put(r.take(shape))
    r.finish()
    assert wt.to_bytes(wr.build()) == wt.to_bytes(src)


def test_writer_casts_float64_to_float32():
    wr = wt.WeightsWriter(wt.Header(0, 2, 5, 0))
    wr.put(np.array([1.5, 2.5], dtype=np.float64))    
    out = wt.to_bytes(wr.build())
    assert len(out) == 20 + 2 * 4                       


def test_writer_rejects_integers():
    with pytest.raises(wt.WeightsError, match="non-floating"):
        wt.WeightsWriter(wt.Header(0, 2, 5, 0)).put(np.array([1, 2]))



@pytest.mark.skipif(not REAL, reason="set DKPRUNE_WEIGHTS to run this test")
def test_real_file_roundtrip(tmp_path):
    assert REAL is not None
    src = Path(REAL)
    w = wt.load(src)
    out = tmp_path / "copy.weights"
    wt.save(w, out)
    assert out.read_bytes() == src.read_bytes()
    assert np.isfinite(w.data).all(), "weights có NaN/Inf: quá trình train có vấn đề"
    if REAL_COUNT:
        assert w.count == int(REAL_COUNT)