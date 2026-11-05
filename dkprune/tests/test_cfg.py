from collections import Counter
from pathlib import Path

import pytest  # type: ignore[import-not-found]

from dkprune.io import cfg

CFGS = Path(__file__).parent / "cfgs"
USER = CFGS / "yolov7-tiny.cfg"


@pytest.mark.parametrize("path", sorted(CFGS.glob("*.cfg")), ids=lambda p: p.name)
def test_roundtrip_byte_for_byte(path, tmp_path):
    out = tmp_path / "out.cfg"
    cfg.save(cfg.load(path), out)
    assert out.read_bytes() == path.read_bytes()


@pytest.mark.parametrize("text", ["", "a", "a\n", "a\r\nb", "\n\n", "x=1\r\n"])
def test_split_then_join_is_identity(text):
    assert "".join(cfg.split_lines(text)) == text


def test_user_cfg_structure():
    c = cfg.load(USER)
    assert c.net.type == "net"
    assert len(c.layers) == 99
    assert Counter(s.type for s in c.layers) == {
        "convolutional": 58, "route": 30, "maxpool": 6, "upsample": 2, "yolo": 3,
    }


def test_user_cfg_values():
    c = cfg.load(USER)
    width = c.net.find("width")
    assert width is not None
    assert width.value == "416"
    saturation = c.net.find("saturation")
    assert saturation is not None
    assert saturation.value == "1.5"          # line containing whitespace
    filters = c.layers[0].find("filters")
    assert filters is not None
    assert filters.value == "32"
    assert c.layers[3].type == "route"
    layers = c.layers[3].find("layers")
    assert layers is not None
    assert layers.value == "-2"
    layers = c.layers[7].find("layers")
    assert layers is not None
    assert layers.value == "-5,-3,-2,-1"  # removed whitespace from value
    assert c.layers[0].find("groups") is None


def test_comment_is_not_value():
    # The conv layer immediately preceding the first [yolo] block has "#activation=linear" followed by "activation=logistic"
    c = cfg.load(USER)
    first_yolo = next(i for i, s in enumerate(c.layers) if s.type == "yolo")
    activation = c.layers[first_yolo - 1].find("activation")
    assert activation is not None
    assert activation.value == "logistic"


def test_option_line_points_to_original_line():
    c = cfg.load(USER)
    opt = c.layers[8].find("filters")
    assert opt is not None
    assert c.lines[opt.line] == "filters=64\n"


@pytest.mark.parametrize("text, msg", [
    ("[net]\nwidth\n", "Invalid option"),
    ("[net]\n=5\n", "Invalid option"),
    ("[net]\nfilters=\n", "Invalid option"),
    ("width=416\n[net]\n", "outside of section"),
    ("[convolutional]\nfilters=1\n", "[net]"),
    ("[net]\n[convolutional\n", "Invalid section header"),
    ("[net]\n[convolutional]\nsize=3\nsize=1\n", "Duplicate option"),
])
def test_errors(text, msg):
    with pytest.raises(cfg.CfgError, match=msg.replace("[", r"\[").replace("]", r"\]")):
        cfg.parse(text)


def test_error_reports_line_number():
    with pytest.raises(cfg.CfgError, match="line 3"):
        cfg.parse("[net]\nwidth=1\noops\n")



def _diff(a: cfg.Cfg, b: cfg.Cfg):
    """Các dòng khác nhau giữa hai cfg có cùng số dòng: [(số dòng từ 1, cũ, mới)]."""
    assert len(a.lines) == len(b.lines)
    return [(i + 1, x, y) for i, (x, y) in enumerate(zip(a.lines, b.lines)) if x != y]
 
 
def test_set_filters_changes_only_that_line():
    old = cfg.load(USER)
    new = cfg.set_values(old, [cfg.Edit(8, "filters", 48)])
    assert _diff(old, new) == [(85, "filters=64\n", "filters=48\n")]
    filters = new.layers[8].find("filters")
    assert filters is not None
    assert filters.value == "48"
 
 
def test_original_is_not_modified():
    old = cfg.load(USER)
    cfg.set_values(old, [cfg.Edit(8, "filters", 48)])
    filters = old.layers[8].find("filters")
    assert filters is not None
    assert filters.value == "64"
 
 
def test_whitespace_around_value_is_kept():
    old = cfg.load(USER)
    new = cfg.set_values(old, [cfg.Edit(None, "saturation", "1.2")])
    assert _diff(old, new) == [(14, "saturation = 1.5\n", "saturation = 1.2\n")]
 
 
def test_many_edits_at_once():
    old = cfg.load(USER)
    new = cfg.set_values(old, [
        cfg.Edit(0, "filters", 24),
        cfg.Edit(8, "filters", 48),
        cfg.Edit(None, "max_batches", 2000),
        cfg.Edit(None, "steps", "1600,1800"),
    ])
    assert [line for line, _, _ in _diff(old, new)] == [21, 23, 29, 85]
    steps = new.net.find("steps")
    assert steps is not None
    assert steps.value == "1600,1800"
 
 
def test_missing_key_is_an_error_by_default():
    with pytest.raises(ValueError, match="option groups not found"):
        cfg.set_values(cfg.load(USER), [cfg.Edit(0, "groups", 2)])
 
 
def test_insert_into_net_keeps_crlf():
    old = cfg.parse("[net]\r\nwidth=416\r\n\r\n[convolutional]\r\nfilters=8\r\n")
    new = cfg.set_values(old, [cfg.Edit(None, "burn_in", 100, insert=True)])
    assert new.lines[:3] == ("[net]\r\n", "width=416\r\n", "burn_in=100\r\n")
    burn_in = new.net.find("burn_in")
    assert burn_in is not None
    assert burn_in.value == "100"
    filters = new.layers[0].find("filters")
    assert filters is not None
    assert filters.value == "8"     # khối sau không bị ảnh hưởng
 
 
def test_insert_into_empty_section_and_at_eof_without_newline():
    old = cfg.parse("[net]\nwidth=416\n[yolo]")
    new = cfg.set_values(old, [cfg.Edit(0, "classes", 20, insert=True)])
    assert "".join(new.lines) == "[net]\nwidth=416\n[yolo]\nclasses=20\n"
 
 
def test_insert_existing_key_just_replaces():
    old = cfg.load(USER)
    new = cfg.set_values(old, [cfg.Edit(None, "burn_in", 50, insert=True)])
    assert len(new.lines) == len(old.lines)
    burn_in = new.net.find("burn_in")
    assert burn_in is not None
    assert burn_in.value == "50"
 
 
@pytest.mark.parametrize("edit, msg", [
    (cfg.Edit(99, "filters", 8), "invalid layer"),
    (cfg.Edit(0, "filters", ""), "empty value"),
    (cfg.Edit(0, "filters", "8\n"), "newlines not allowed"),
    (cfg.Edit(0, "fil ters", 8), "invalid key"),
])
def test_invalid_edits(edit, msg):
    with pytest.raises(ValueError, match=msg):
        cfg.set_values(cfg.load(USER), [edit])
 
 
def test_same_key_twice_is_an_error():
    with pytest.raises(ValueError, match="duplicate edit"):
        cfg.set_values(cfg.load(USER), [cfg.Edit(0, "filters", 8), cfg.Edit(0, "filters", 16)])
 
 
def test_save_edited_cfg(tmp_path):
    new = cfg.set_values(cfg.load(USER), [cfg.Edit(8, "filters", 48)])
    out = tmp_path / "pruned.cfg"
    cfg.save(new, out)
    filters = cfg.load(out).layers[8].find("filters")
    assert filters is not None
    assert filters.value == "48"
    assert out.read_bytes().count(b"\r\n") == USER.read_bytes().count(b"\r\n")