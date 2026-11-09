from dkprune.layers.base import *
from dkprune.io.cfg import Section

_UNSUPPORTED = {
    "share_index": "",      
    "binary": "0", "xnor": "0", "deform": "0",
    "stride_x": "", "stride_y": "", "dilation": "1", "antialiasing": "0",
}

class Convolutional:
    names = ("convolutional", "conv")

    def infer(self, index: int, section: Section, prev: Input, layer_at: LayerAt) -> Inferred:
        reject(section, index, _UNSUPPORTED)
        filters = opt_int(section, index, "filters")
        size = opt_int(section, index, "size", 1)
        stride = opt_int(section, index, "stride", 1)
        groups = opt_int(section, index, "groups", 1)
        bn = opt_int(section, index, "batch_normalize", 0) == 1
        padding = size // 2 if opt_int(section, index, "pad", 0) else opt_int(section, index, "padding", 0)

        if filters <= 0 or size <= 0 or stride <= 0 or groups <= 0:
            raise GraphError(f"Layer {index} [{section.type}] has invalid parameters: filters={filters}, size={size}, stride={stride}, groups={groups}")

        if prev.channels % groups or filters % groups:
            raise GraphError(f"Layer {index} [{section.type}] has incompatible channels and groups: in_channels={prev.channels}, filters={filters}, groups={groups}")

        h, w = prev.hw
        out_hw = ((h + 2 * padding - size) // stride + 1, (w + 2 * padding - size) // stride + 1)

        if min(out_hw) <= 0:
            raise GraphError(f"Layer {index} [{section.type}] has invalid output dimensions: {out_hw}")

        spec = ConvSpec(filters, prev.channels, size, stride, padding, groups, bn)
        flops = 2.0 * spec.filters * (spec.in_channels // spec.groups) * spec.size * spec.size * out_hw[0] * out_hw[1]

        locks = ((index, f"groups={groups} > 1"),) if groups > 1 else ()
        return Inferred(
            sources=(prev.index,),
            out_channels=filters,
            out_hw=out_hw,
            origin=tuple((index, j) for j in range(filters)),
            conv=spec,
            flops=flops,
            locks=locks,
        )