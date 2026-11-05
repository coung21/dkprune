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