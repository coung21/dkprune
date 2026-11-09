import os
from pathlib import Path

import numpy as np
import pytest

from dkprune import criteria, graph, policy, surgery, verify
from dkprune.graph import ConvParams
from dkprune.io import cfg as cfgio
from dkprune.io import weights as wt
from dkprune.layers.base import GraphError
from tests.forward import activation, forward

CFGS = Path(__file__).parent / "cfgs"
REAL_CFG = os.environ.get("DKPRUNE_CFG")
REAL_WEIGHTS = os.environ.get("DKPRUNE_WEIGHTS")


# ---------------------------------------------------------------- tiện ích

def load(name: str, size: tuple[int, int] | None = None) -> cfgio.Cfg:
    """Đọc cfg; nếu có size=(rộng, cao) thì đổi kích thước ảnh để forward cho nhanh."""
    c = cfgio.load(CFGS / name if not os.path.isabs(name) else name)
    if size:
        c = cfgio.set_values(c, [cfgio.Edit(None, "width", size[0]), cfgio.Edit(None, "height", size[1])])
    return c


def random_params(g: graph.Graph, seed=0) -> dict[int, ConvParams]:
    rng = np.random.default_rng(seed)
    out = {}
    for l in g.convs():
        s = l.conv
        assert s is not None
        fan_in = s.weight_shape[1] * s.size * s.size
        w = (rng.standard_normal(s.weight_shape) * np.sqrt(2 / fan_in)).astype("<f4")
        b = (0.1 * rng.standard_normal(s.filters)).astype("<f4")
        if s.batch_normalize:
            out[l.index] = ConvParams(
                b, w,
                rng.uniform(0.5, 1.5, s.filters).astype("<f4"),
                (0.1 * rng.standard_normal(s.filters)).astype("<f4"),
                rng.uniform(0.5, 1.5, s.filters).astype("<f4"))
        else:
            out[l.index] = ConvParams(b, w)
    return out


def group_of(g: graph.Graph, conv: int) -> int:
    gid = g.layers[conv].group
    assert gid is not None, f"layer {conv} is not a conv"
    return gid


def filters_of(c: cfgio.Cfg, layer: int) -> str:
    opt = c.layers[layer].find("filters")
    assert opt is not None
    return opt.value


def kill(g: graph.Graph, params, keep_by_group) -> dict[int, ConvParams]:
    """Làm đầu ra của các filter sắp bị cắt luôn bằng 0 (BN: scale = bias = 0)."""
    out = dict(params)
    for gid, idx in keep_by_group.items():
        for c in g.groups[gid]:
            p = params[c]
            dead = np.setdiff1d(np.arange(p.weight.shape[0]), idx)
            scale = None if p.scale is None else p.scale.copy()
            bias, weight = p.bias.copy(), p.weight.copy()
            bias[dead] = 0
            if scale is not None:
                scale[dead] = 0
            else:
                weight[dead] = 0
            out[c] = ConvParams(bias, weight, scale, p.mean, p.var)
    return out


# ---------------------------------------------------------------- criteria

def test_l2_folds_batchnorm():
    w = np.array([[[[3.0]]], [[[4.0]]]], dtype="<f4")           # 2 filter, 1x1, 1 kênh
    p = ConvParams(np.zeros(2, "<f4"), w, np.array([1.0, 0.5], "<f4"),
                   np.zeros(2, "<f4"), np.array([1.0, 4.0], "<f4"))
    s = criteria.l2([p], np.random.default_rng(0))
    expect = [3.0 * 1.0 / np.sqrt(1 + 1e-5), 4.0 * 0.5 / np.sqrt(4 + 1e-5)]
    assert np.allclose(s, expect)
    assert criteria.l2_raw([p], np.random.default_rng(0)).tolist() == [3.0, 4.0]


def test_group_score_is_sum_of_members():
    g = graph.build(load("yolov4.cfg"))
    p = random_params(g)
    members = [p[4], p[6]]
    rng = np.random.default_rng(0)
    assert np.allclose(criteria.l2(members, rng), criteria.l2([p[4]], rng) + criteria.l2([p[6]], rng))


def test_unknown_criterion():
    with pytest.raises(ValueError, match="Unknown criterion"):
        criteria.get("taylor")


# ---------------------------------------------------------------- policy

@pytest.mark.parametrize("n, ratio, expect", [
    (32, 0.16, 24),      # 26.9 -> làm tròn về 24
    (64, 0.16, 56),      # 53.8 -> 56
    (128, 0.16, 104),    # 107.5 -> 104
    (256, 0.5, 128),
    (8, 0.5, 8),         # min_keep = 8: không cắt
    (12, 0.5, 8),
])
def test_target_keep(n, ratio, expect):
    assert policy.target_keep(n, policy.PruneConfig(ratio=ratio)) == expect


def test_plan_respects_locks_and_skip():
    c = load("yolov7-tiny-gc10.cfg")
    g = graph.build(c)
    keep = policy.plan(g, random_params(g), policy.PruneConfig(ratio=0.3, skip=(0,)))
    pruned_convs = {g.groups[gid][0] for gid in keep}
    assert not pruned_convs & {0, 89, 93, 97}
    assert all(len(idx) % 8 == 0 for idx in keep.values())


def test_plan_keeps_highest_scores():
    c = load("yolov7-tiny-gc10.cfg")
    g = graph.build(c)
    p = random_params(g)
    keep = policy.plan(g, p, policy.PruneConfig(ratio=0.5))
    gid = group_of(g, 1)
    scores = criteria.l2([p[1]], np.random.default_rng(0))
    dropped = np.setdiff1d(np.arange(64), keep[gid])
    assert scores[keep[gid]].min() >= scores[dropped].max()


def test_bad_ratio():
    with pytest.raises(ValueError):
        policy.PruneConfig(ratio=1.0)


# ---------------------------------------------------------------- surgery

def test_surgery_shapes_and_cfg():
    c = load("yolov7-tiny-gc10.cfg")
    g = graph.build(c)
    p = random_params(g)
    keep = {group_of(g, 4): np.arange(0, 32, 2)[:16]}       # conv 4: 32 -> 16 filter
    out = surgery.apply(c, g, p, keep)
    assert out.params[4].weight.shape == (16, 64, 1, 1)        # conv 4 là 1x1
    scale = out.params[4].scale
    assert scale is not None and scale.shape == (16,)
    assert out.params[5].weight.shape == (32, 16, 3, 3)        # cột của conv 5
    assert out.params[8].weight.shape == (64, 112, 1, 1)       # route 7: 128 -> 112
    assert out.graph.layers[7].out_channels == 112
    assert filters_of(out.cfg, 4) == "16"
    changed = [i for i, (a, b) in enumerate(zip(c.lines, out.cfg.lines)) if a != b]
    assert len(changed) == 1                                    # chỉ đúng một dòng filters=
    assert filters_of(c, 4) == "32"            # cfg gốc không bị sửa


def test_surgery_keeps_right_columns():
    c = load("yolov7-tiny-gc10.cfg")
    g = graph.build(c)
    p = random_params(g)
    kept = np.array([1, 5, 9, 30] + list(range(10, 22)))        # 16 filter bất kỳ
    out = surgery.apply(c, g, p, {group_of(g, 4): kept})
    kept = np.sort(kept)
    assert np.array_equal(out.params[8].weight[:, 32:48], p[8].weight[:, 32 + kept])


def test_surgery_ties_shortcut_group():
    c = load("yolov4.cfg")
    g = graph.build(c)
    gid = group_of(g, 4)
    out = surgery.apply(c, g, random_params(g), {gid: np.arange(32)})
    assert out.graph.layers[4].out_channels == out.graph.layers[6].out_channels == 32


def test_surgery_refuses_locked():
    c = load("yolov7-tiny-gc10.cfg")
    g = graph.build(c)
    with pytest.raises(GraphError, match="is not prunable"):
        surgery.apply(c, g, random_params(g), {group_of(g, 89): np.arange(40)})


@pytest.mark.parametrize("idx", [[], [0, 0, 1], [0, 99]])
def test_surgery_rejects_bad_keep(idx):
    c = load("yolov7-tiny-gc10.cfg")
    g = graph.build(c)
    with pytest.raises(GraphError, match="invalid filter indices"):
        surgery.apply(c, g, random_params(g), {group_of(g, 4): np.array(idx, dtype=int)})


def test_pruned_files_verify(tmp_path):
    c = load("yolov7-tiny-gc10.cfg")
    g = graph.build(c)
    p = random_params(g)
    out = surgery.apply(c, g, p, policy.plan(g, p, policy.PruneConfig(ratio=0.3)))
    cfgio.save(out.cfg, tmp_path / "p.cfg")
    wt.save(graph.write_params(out.graph, out.params, wt.Header(0, 2, 5, 0)), tmp_path / "p.weights")
    rep = verify.check_files(tmp_path / "p.cfg", tmp_path / "p.weights")
    assert rep.ok and rep.param_count == out.graph.param_count < g.param_count


# ---------------------------------------------------------------- phép thử tương đương

def equivalence(c: cfgio.Cfg, params=None, ratio=0.4, seed=0):
    g = graph.build(c)
    p = params if params is not None else random_params(g, seed)
    keep = policy.plan(g, p, policy.PruneConfig(ratio=ratio, criterion="random", seed=seed))
    assert keep, "không có nhóm nào bị cắt"
    for gid in keep:                                            # f(0) phải bằng 0 thì mới "tắt" được kênh
        for m in g.groups[gid]:
            name, fn = activation(c, m, "logistic")
            assert abs(fn(np.zeros(1))[0]) < 1e-12, f"conv {m} dùng activation {name}"
    out = surgery.apply(c, g, p, keep)
    x = np.random.default_rng(seed + 1).random((g.input_channels, *g.input_hw))
    before = forward(c, g, kill(g, p, keep), x)
    after = forward(out.cfg, out.graph, out.params, x)
    assert len(before) == len(after) > 0
    for a, b in zip(before, after):
        assert a.shape == b.shape
        assert np.allclose(a, b, rtol=1e-5, atol=1e-5), f"lệch tối đa {np.abs(a - b).max()}"
    return g, out.graph


@pytest.mark.parametrize("name, size", [
    ("yolov7-tiny-gc10.cfg", (128, 64)),     # route nhiều nguồn, maxpool giữa hai conv, SPP
    ("LegoGears.cfg", (64, 64)),
    ("yolov4-tiny.cfg", (64, 64)),           # route có groups (bị khóa)
    ("yolov3-tiny.cfg", (64, 64)),           # maxpool stride 1
    ("yolov4.cfg", (64, 64)),                # shortcut: cắt theo nhóm
])
def test_equivalence(name, size):
    before, after = equivalence(load(name, size))
    assert after.param_count < before.param_count


@pytest.mark.skipif(not (REAL_CFG and REAL_WEIGHTS), reason="đặt DKPRUNE_CFG và DKPRUNE_WEIGHTS")
def test_equivalence_real_weights():
    assert REAL_CFG and REAL_WEIGHTS
    c = load(REAL_CFG, (64, 64))
    g = graph.build(c)
    p = graph.read_params(g, wt.load(REAL_WEIGHTS))
    equivalence(c, p)