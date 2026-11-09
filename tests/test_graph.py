from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dkprune import graph as gr
from dkprune.io import cfg
from dkprune.io import weights as wt
from dkprune.layers.base import IMAGE, GraphError

HERE = Path(__file__).parent
CFG = HERE / "cfgs" / "yolov7-tiny.cfg"
# Same network retrained for 10 classes; *.weights is gitignored, so it may be missing.
REAL = HERE / "weights" / "yolov7-tiny-gc10_best.weights"

HEADER = wt.Header(0, 2, 5, 0)
YOLO_HEADS = (89, 93, 97)       # 1x1 convs feeding the [yolo] layers 90, 94, 98


@pytest.fixture(scope="module")
def g() -> gr.Graph:
    return gr.build(cfg.load(CFG))


def arange_weights(graph: gr.Graph, extra: int = 0) -> wt.Weights:
    """Weights whose values are 0, 1, 2, ... so the read order is easy to check."""
    return wt.Weights(HEADER, np.arange(graph.param_count + extra, dtype=np.float32))


# ---------------------------------------------------------------- structure

def test_input(g):
    assert g.input_hw == (416, 416)
    assert g.input_channels == 3
    assert len(g.layers) == 99
    assert len(g.convs()) == 58


def test_first_conv(g):
    l = g.layers[0]
    assert l.sources == (IMAGE,)
    assert (l.in_channels, l.out_channels) == (3, 32)
    assert (l.in_hw, l.out_hw) == ((416, 416), (208, 208))      # stride 2, pad 1
    assert l.in_origin == tuple((IMAGE, j) for j in range(3))
    assert l.origin == tuple((0, j) for j in range(32))
    assert l.conv is not None
    assert l.conv.weight_shape == (32, 3, 3, 3)
    assert l.conv.batch_normalize
    assert l.conv.param_count == 32 * 3 * 3 * 3 + 4 * 32
    assert l.flops == 2.0 * 32 * 3 * 3 * 3 * 208 * 208


def test_relative_route(g):
    # layer 3: "layers=-2" -> layer 1
    l = g.layers[3]
    assert l.type == "route"
    assert l.sources == (1,)
    assert l.out_channels == 64
    assert l.origin == g.layers[1].origin


def test_concat_route_keeps_channel_origin(g):
    # layer 7: "layers = -5,-3,-2,-1" -> 2, 4, 5, 6, each 32 channels
    l = g.layers[7]
    assert l.sources == (2, 4, 5, 6)
    assert l.out_channels == 128
    assert l.origin == tuple((c, j) for c in (2, 4, 5, 6) for j in range(32))
    assert g.layers[8].in_origin == l.origin
    assert g.layers[8].conv is not None
    assert g.layers[8].conv.weight_shape == (64, 128, 1, 1)


def test_absolute_route(g):
    # layer 47: "layers = 24"
    assert g.layers[47].sources == (24,)
    assert g.layers[47].out_hw == (26, 26)


def test_spatial_sizes(g):
    assert g.layers[9].out_hw == (52, 52)           # maxpool stride 2
    assert g.layers[36].out_hw == (13, 13)          # SPP maxpool stride 1 keeps the size
    assert g.layers[40].out_hw == (13, 13)
    assert g.layers[46].out_hw == (26, 26)          # upsample x2
    assert [g.layers[i].out_hw for i in (90, 94, 98)] == [(52, 52), (26, 26), (13, 13)]


def test_maxpool_and_upsample_pass_origin_through(g):
    assert g.layers[9].origin == g.layers[8].origin
    assert g.layers[46].origin == g.layers[45].origin


def test_spp_route_repeats_same_conv(g):
    # layer 41: maxpools 40, 38, 36 of conv 35, then conv 35 itself
    l = g.layers[41]
    assert l.sources == (40, 38, 36, 35)
    assert l.out_channels == 1024
    assert l.origin == tuple((35, j) for _ in range(4) for j in range(256))


def test_yolo_heads_are_locked(g):
    for i in YOLO_HEADS:
        l = g.layers[i]
        assert l.out_channels == 255
        assert not l.prunable
        assert l.lock_reason == f"is a source of a yolo layer (layer {i + 1})"


def test_other_convs_are_prunable(g):
    prunable = [l.index for l in g.convs() if l.prunable]
    assert len(prunable) == 55
    assert not set(prunable) & set(YOLO_HEADS)


def test_non_conv_layers(g):
    for l in g.layers:
        if not l.is_conv:
            assert l.conv is None
            assert l.group is None
            assert l.lock_reason is None
            assert not l.prunable
            assert l.flops == 0


def test_no_shortcut_means_one_conv_per_group(g):
    assert len(g.groups) == 58
    assert all(len(members) == 1 for members in g.groups)
    for l in g.convs():
        assert l.group is not None
        assert g.groups[l.group] == (l.index,)


def test_totals(g):
    assert g.param_count == sum(l.conv.param_count for l in g.convs() if l.conv is not None)
    assert g.param_count == 6_241_885
    assert g.bflops == pytest.approx(sum(l.flops for l in g.layers) / 1e9)


# ---------------------------------------------------------------- consumers

def test_consumers_through_concat(g):
    # conv 2 is the first 32 channels of route 7, which feeds conv 8
    assert g.consumers(2) == {8: [(f, f) for f in range(32)]}
    # conv 4 is channels 32..63 of route 7
    assert g.consumers(4)[8] == [(f, 32 + f) for f in range(32)]


def test_consumers_through_route_and_maxpool(g):
    # conv 24 -> maxpool 25 -> conv 26, route 27 -> conv 28, and route 47 -> conv 48
    assert set(g.consumers(24)) == {26, 28, 48}


def test_consumers_repeated_columns(g):
    # conv 35 appears 4 times in the input of conv 42
    cols = g.consumers(35)[42]
    assert len(cols) == 4 * 256
    assert cols[:2] == [(0, 0), (1, 1)]
    assert cols[256] == (0, 256)


def test_yolo_head_has_no_conv_consumer(g):
    for i in YOLO_HEADS:
        assert g.consumers(i) == {}


# ---------------------------------------------------------------- table

def test_table(g):
    text = g.table()
    assert len(text.splitlines()) == 1 + 99 + 2
    assert "LOCKED: is a source of a yolo layer (layer 90)" in text
    assert "55/58 convs prunable" in text
    assert "6,241,885 params" in text


# ---------------------------------------------------------------- read / write params

def test_read_params_order_with_bn(g):
    params = gr.read_params(g, arange_weights(g))
    assert set(params) == {l.index for l in g.convs()}
    p = params[0]
    assert p.has_bn
    assert p.bias.tolist() == list(range(0, 32))         # darknet order: bias, scale, mean, var, weight
    assert p.scale is not None and p.scale.tolist() == list(range(32, 64))
    assert p.mean is not None and p.mean.tolist() == list(range(64, 96))
    assert p.var is not None and p.var.tolist() == list(range(96, 128))
    assert p.weight.shape == (32, 3, 3, 3)
    assert p.weight.ravel().tolist() == list(range(128, 992))
    assert params[1].bias[0] == 992                      # next conv starts right after


def test_read_params_without_bn(g):
    params = gr.read_params(g, arange_weights(g))
    p = params[97]                                       # last layer: no batch_normalize
    assert not p.has_bn
    assert p.scale is None and p.mean is None and p.var is None
    assert p.weight.shape == (255, 512, 1, 1)
    end = g.param_count
    assert p.weight.ravel()[-1] == end - 1
    assert p.bias[0] == end - 255 * 512 - 255


def test_read_params_too_short(g):
    data = np.zeros(g.param_count - 1, dtype=np.float32)
    with pytest.raises(wt.WeightsError, match="Not enough weights"):
        gr.read_params(g, wt.Weights(HEADER, data))


def test_read_params_too_long(g):
    with pytest.raises(wt.WeightsError, match="3 remaining"):
        gr.read_params(g, arange_weights(g, extra=3))


def test_write_params_roundtrip(g):
    src = arange_weights(g)
    out = gr.write_params(g, gr.read_params(g, src), HEADER)
    assert out.header == HEADER
    assert np.array_equal(out.data, src.data)


def test_write_rejects_wrong_weight_shape(g):
    params = gr.read_params(g, arange_weights(g))
    params[8] = replace(params[8], weight=params[8].weight[:, :100])
    with pytest.raises(GraphError, match="Layer 8: weight has shape"):
        gr.write_params(g, params, HEADER)


def test_write_rejects_bn_mismatch(g):
    params = gr.read_params(g, arange_weights(g))
    params[0] = replace(params[0], scale=None, mean=None, var=None)
    with pytest.raises(GraphError, match="Layer 0: BN parameters"):
        gr.write_params(g, params, HEADER)


def test_write_rejects_wrong_vector_length(g):
    params = gr.read_params(g, arange_weights(g))
    assert params[0].var is not None
    params[0] = replace(params[0], var=params[0].var[:-1])
    with pytest.raises(GraphError, match="Layer 0: BN/bias vector has shape"):
        gr.write_params(g, params, HEADER)


def test_write_rejects_missing_conv(g):
    params = gr.read_params(g, arange_weights(g))
    del params[5]
    with pytest.raises(KeyError):
        gr.write_params(g, params, HEADER)

def test_default_valued_options_are_accepted():
    text = ("[net]\nwidth=32\nheight=32\nchannels=3\n"
            "[convolutional]\nfilters=8\nsize=3\npad=1\ndilation=1\nbinary=0\n"
            "[convolutional]\nfilters=8\nsize=1\n"
            "[shortcut]\nfrom=-2\nweights_type=none\n")
    gr.build(cfg.parse(text))

# ---------------------------------------------------------------- real weights

def ten_class_cfg() -> cfg.Cfg:
    """yolov7-tiny.cfg changed to 10 classes: heads get (10 + 5) * 3 = 45 filters."""
    c = cfg.load(CFG)
    edits = [cfg.Edit(i, "filters", 45) for i in YOLO_HEADS]
    edits += [cfg.Edit(i + 1, "classes", 10) for i in YOLO_HEADS]
    return cfg.set_values(c, edits)


@pytest.mark.skipif(not REAL.exists(), reason=f"{REAL.name} not found")
def test_real_weights_roundtrip_byte_for_byte(tmp_path):
    g10 = gr.build(ten_class_cfg())
    w = wt.load(REAL)
    assert w.count == g10.param_count
    params = gr.read_params(g10, w)
    assert params[89].weight.shape == (45, 128, 1, 1)
    out = tmp_path / "out.weights"
    wt.save(gr.write_params(g10, params, w.header), out)
    assert out.read_bytes() == REAL.read_bytes()
