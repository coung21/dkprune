from dkprune.layers.base import *
from dkprune.io.cfg import Section


class Upsample:
    names = ("upsample",)
 
    def infer(self, index: int, section: Section, prev: Input, layer_at: LayerAt) -> Inferred:
        stride = opt_int(section, index, "stride", 2)
        if stride <= 0:
            raise GraphError(f"Layer {index} [{section.type}] has invalid stride: {stride}")
        h, w = prev.hw
        out_hw = (h * stride, w * stride)
        return Inferred(
            sources=(prev.index,),
            out_channels=prev.channels,
            out_hw=out_hw,
            origin=prev.origin,
        )