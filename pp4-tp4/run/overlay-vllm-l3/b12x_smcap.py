"""Reserve SMs from b12x cooperative (grid-barrier) MoE kernels.

b12x W4A16 fused MoE launches sms*blocks_per_sm CTAs and crosses software
all-CTA grid barriers. Under PP micro-batching, NCCL PP kernels (sampled-token
broadcast recv, hidden-state send/recv) sit resident on a few SMs for tens of
ms waiting for their peer; the grid barrier then cannot complete until those
NCCL CTAs exit, which serializes every stage behind the pipeline round trip
(nsys: graph stalls exactly for the Broadcast_RING duration).

B12X_SM_RESERVE=N makes torch.cuda.get_device_properties report
multi_processor_count - N to callers inside the b12x package only, so the
cooperative grid always fits beside the NCCL CTAs.
"""
import os
import sys

_RESERVE = int(os.environ.get("B12X_SM_RESERVE", "0") or 0)


class _Props:
    __slots__ = ("_p", "multi_processor_count")

    def __init__(self, p, reserve):
        self._p = p
        self.multi_processor_count = max(1, int(p.multi_processor_count) - reserve)

    def __getattr__(self, name):
        return getattr(self._p, name)


def _install():
    import torch

    orig = torch.cuda.get_device_properties
    if getattr(orig, "_b12x_smcap", False):
        return

    def get_device_properties(device=None):
        p = orig(device)
        f = sys._getframe(1)
        mod = f.f_globals.get("__name__", "")
        if mod.startswith("b12x"):
            return _Props(p, _RESERVE)
        return p

    get_device_properties._b12x_smcap = True
    torch.cuda.get_device_properties = get_device_properties


if _RESERVE > 0:
    import importlib.abc
    import importlib.machinery

    class _Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == "b12x":
                sys.meta_path.remove(self)
                _install()
            return None

    if "torch" in sys.modules or True:
        sys.meta_path.insert(0, _Hook())
