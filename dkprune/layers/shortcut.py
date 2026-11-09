from dkprune.layers.base import *
from dkprune.io.cfg import Section

def _single_conv(origin: tuple[Channel, ...]) -> int | None:
    """If channels are 0..N-1 of a single conv, return its index; otherwise return None."""
    if not origin:
        return None
    conv = origin[0][0]
    if conv == IMAGE or any(o != (conv, j) for j, o in enumerate(origin)):
        return None
    return conv
 
 
class Shortcut:
    """Add elements: channel k of two inputs must always go together when cutting."""
    names = ("shortcut",)
 
    def infer(self, index: int, section: Section, prev: Input, layer_at: LayerAt) -> Inferred:
        reject(section, index, {"weights_type": "none"})       
        opt = section.find("from")
        if opt is None:
            raise GraphError(f"Layer {index} [{section.type}] missing required key from")
        others = [layer_at(resolve(index, raw, section)) for raw in opt.value.split(",")]
 
        ties, locks = [], []
        base = _single_conv(prev.origin)
        for o in others:
            if o.channels != prev.channels or o.hw != prev.hw:
                raise GraphError(f"Layer {index} [{section.type}]: layer {o.index} ({o.channels} kênh, {o.hw}) "
                                 f"different from previous layer {prev.index} ({prev.channels} kênh, {prev.hw})")
            other = _single_conv(o.origin)
            if base is not None and other is not None:
                ties.append((base, other))          # prune both convs together
            else:
                # inputs mix several layers: channels can't be matched one to one, so lock them
                why = f"is a source of a shortcut (layer {index})"
                locks += [(c, why) for c in producers(prev.origin) + producers(o.origin)]
 
        sources = (prev.index,) + tuple(o.index for o in others)
        return Inferred(sources, prev.channels, prev.hw, prev.origin, ties=tuple(ties), locks=tuple(locks))
 