#!/usr/bin/env python3
"""Route in-place TP all-reduces through the PCIe custom/DMA all-reduce.

Marker: kimi-k3-inplace-ar-dma.

`all_reduce_in_place` went straight to PyNCCL. Kimi's row-parallel prefill
projections (attention output, MoE final output; >= 1,024 rows) use that
entry, so they stayed on the NCCL ring while the functional entry already
used the b12x CE-driven ring. Callers treat the source tensor as dead after
the collective and consume the return value, so returning the custom
all-reduce's output is equivalent.
"""
import sys, py_compile
from pathlib import Path
path = Path(sys.argv[1]); src = path.read_text()
MARKER = "kimi-k3-inplace-ar-dma"
if MARKER in src:
    print("already patched"); sys.exit(0)
old = '''        pynccl_comm = self.pynccl_comm
        if pynccl_comm is None or pynccl_comm.disabled:
            torch.distributed.all_reduce(input_, group=self.device_group)
            return input_
        out = pynccl_comm.all_reduce(input_, out_tensor=input_)
'''
new = f'''        # {MARKER}: eligible prefill-size tensors take the same PCIe
        # custom/DMA all-reduce as the functional entry; the caller consumes
        # the returned tensor and treats the source as dead.
        ca_comm = self.ca_comm
        if (
            ca_comm is not None
            and not ca_comm.disabled
            and ca_comm.should_custom_ar(input_)
        ):
            out = ca_comm.custom_all_reduce(input_)
            if out is not None:
                return out
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is None or pynccl_comm.disabled:
            torch.distributed.all_reduce(input_, group=self.device_group)
            return input_
        out = pynccl_comm.all_reduce(input_, out_tensor=input_)
'''
assert src.count(old) == 1, "anchor"
path.write_text(src.replace(old, new)); py_compile.compile(str(path), doraise=True); print("patched + compiled:", path)
