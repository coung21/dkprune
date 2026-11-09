from dkprune.layers.base import *
from dkprune.io.cfg import Section

class Maxpool:
    names = ("maxpool", "max")
    def infer(self, index: int, section: Section, prev: Input, layer_at: LayerAt) -> Inferred:
        reject(section, index, {"stride_x": "", "stride_y": "", "maxpool_depth": "0", "antialiasing": "0"})
        size = opt_int(section, index, "size", 1)
        stride = opt_int(section, index, "stride", 1)
        padding = opt_int(section, index, "padding", size - 1)

        h, w = prev.hw
        out_hw = ((h + padding - size) // stride + 1, (w + padding - size) // stride + 1)

        if min(out_hw) <= 0:
            raise GraphError(f"Layer {index} [{section.type}] has invalid output dimensions: {out_hw}") 

        return Inferred(
            sources=(prev.index,),
            out_channels=prev.channels,
            out_hw=out_hw,
            origin=prev.origin,
        )