#!/usr/bin/env python3
"""Zero-copy frames on the synchronous draft RPC path (marker: kimi-k3-draft-zero-copy-sync).

The synchronous proposal (a step that sampled a token) serialized 132 MiB of
aux rows with `.numpy().tobytes()` (a Python-level copy, ~40 ms) before the
ZMQ send. The pinned staging buffers are not rewritten until the next
proposal, and `_rpc` waits for the reply, so the frames can reference the
staging memory directly and be sent with `copy=False`.
"""
import sys, py_compile
from pathlib import Path
path = Path(sys.argv[1]); src = path.read_text()
MARKER = "kimi-k3-draft-zero-copy-sync"
if MARKER in src:
    print("already patched"); sys.exit(0)
def replace_once(old, new):
    global src
    n = src.count(old); assert n == 1, f"anchor count {n} for {old[:70]!r}"
    src = src.replace(old, new)
replace_once(
'''        else:
            positions_frame = positions_staging.numpy().tobytes()
            # NumPy has inconsistent bfloat16 support; preserve its exact bits as u16.
            context_frame = context_staging.view(torch.uint16).numpy().tobytes()
''',
f'''        else:  # {MARKER}: staging memory stays valid until the reply arrives
            positions_frame = memoryview(positions_staging.numpy())
            # NumPy has inconsistent bfloat16 support; preserve its exact bits as u16.
            context_frame = memoryview(context_staging.view(torch.uint16).numpy())
''')
replace_once(
'''        with self._async_lock:
            try:
                self._socket.send_multipart(frames)
                response_frames = self._socket.recv_multipart()
''',
f'''        with self._async_lock:
            try:
                self._socket.send_multipart(frames, copy=False)  # {MARKER}
                response_frames = self._socket.recv_multipart()
''')
path.write_text(src); py_compile.compile(str(path), doraise=True); print("patched + compiled:", path)
