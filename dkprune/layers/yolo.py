from dkprune.io.cfg import Section
from dkprune.layers.base import Inferred, Input, LayerAt, producers


class Yolo:
    """Channels of yolo always = (classes + 5) x anchors, so they are not pruned."""
    names = ("yolo",)

    def infer(self, index: int, section: Section, prev: Input, layer_at: LayerAt) -> Inferred:
        locks = tuple((c, f"is a source of a yolo layer (layer {index})") for c in producers(prev.origin))
        return Inferred((prev.index,), prev.channels, prev.hw, prev.origin, locks=locks)

    