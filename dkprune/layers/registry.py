

from typing import cast

from dkprune.layers.base import GraphError, LayerRule
from dkprune.layers.convolutional import Convolutional
from dkprune.layers.maxpool import Maxpool
from dkprune.layers.route import Route
from dkprune.layers.shortcut import Shortcut
from dkprune.layers.upsample import Upsample
from dkprune.layers.yolo import Yolo

_RULES: dict[str, LayerRule] = {}
for _rule in (Convolutional(), Maxpool(), Upsample(), Route(), Shortcut(), Yolo()):
    for _name in _rule.names:
        _RULES[_name] = cast(LayerRule, _rule)


def get(layer_type: str, index: int) -> LayerRule:
    try:
        return _RULES[layer_type]
    except KeyError:
        raise GraphError(f"Layer {index}: type [{layer_type}] is not supported. "
                         f"Supported: {', '.join(sorted(_RULES))}") from None


def is_conv(layer_type: str) -> bool:
    return layer_type in Convolutional.names