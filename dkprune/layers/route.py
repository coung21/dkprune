from dkprune.layers.base import *
from dkprune.io.cfg import Section

class Route:
    names = ("route",)

    def infer(self, index: int, section: Section, prev: Input, layer_at: LayerAt) -> Inferred:

        opt = section.find("layers")
        if opt is None:
            raise GraphError(f"Layer {index} [{section.type}] missing required key layers")
        srcs = [layer_at(resolve(index, raw, section)) for raw in opt.value.split(",")]
        groups = opt_int(section, index, "groups", 1)
        group_id = opt_int(section, index, "group_id", 0)
        if groups <= 0 or not 0 <= group_id < groups:
            raise GraphError(f"Layer {index} [{section.type}] has invalid groups={groups} or group_id={group_id}")

        hw = srcs[0].hw
        for s in srcs:
            if s.hw != hw:
                raise GraphError(f"Layer {index} [{section.type}] has incompatible input dimensions: {[s.hw for s in srcs]}")

        origin = []
        locks = []
        for s in srcs:
            if s.channels % groups:
                raise GraphError(f"Layer {index} [{section.type}] has incompatible input channels and groups: in_channels={s.channels}, groups={groups}")
            part = s.channels // groups
            origin.extend(s.origin[group_id * part:(group_id + 1) * part])
            if groups > 1:
                locks += [(c, f"is a source of a grouped route (layer {index})") for c in producers(s.origin)]

        return Inferred(
            sources=tuple(s.index for s in srcs),
            out_channels=len(origin),
            out_hw=hw,
            origin=tuple(origin),
            locks=tuple(locks),
        )