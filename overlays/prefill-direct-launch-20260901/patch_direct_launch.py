#!/usr/bin/env python3
"""Call the dynamic MoE launch function directly in eager mode.

Marker: kimi-k3-direct-dynamic-launch.

`b12x::tp_moe_dynamic_launch` is a `torch.library.custom_op` with about sixty
arguments and `mutates_args="unknown"`; each eager call pays the Python
dispatcher wrapper (`fill_defaults`, ADInplaceOrView version bumps), about
1-2 ms per MoE layer. Outside torch.compile tracing the raw Python function
launches the same kernels, so eager prefill calls it directly.
`B12X_DIRECT_DYNAMIC_LAUNCH=0` restores the dispatcher path.
"""
import sys, py_compile
from pathlib import Path
path = Path(sys.argv[1]); src = path.read_text()
MARKER = "kimi-k3-direct-dynamic-launch"
if MARKER in src:
    print("already patched"); sys.exit(0)
def replace_once(old, new):
    global src
    n = src.count(old); assert n == 1, f"anchor count {n} for {old[:70]!r}"
    src = src.replace(old, new)
replace_once(
    "    torch.ops.b12x.tp_moe_dynamic_launch(\n        workspace.packed_a_view,\n",
    "    _dynamic_launch_entry()(  # " + MARKER + "\n        workspace.packed_a_view,\n",
)
# define the entry selector right after the fake registration of the op
replace_once(
    "@_tp_moe_dynamic_launch_op.register_fake\n",
    f'''# {MARKER}: begin
_DIRECT_DYNAMIC_LAUNCH = os.environ.get("B12X_DIRECT_DYNAMIC_LAUNCH", "1") == "1"


def _dynamic_launch_entry():
    """Raw launch function outside compile tracing; the dispatcher op otherwise."""
    if _DIRECT_DYNAMIC_LAUNCH and not torch.compiler.is_compiling():
        return _tp_moe_dynamic_launch_op._init_fn
    return torch.ops.b12x.tp_moe_dynamic_launch
# {MARKER}: end


@_tp_moe_dynamic_launch_op.register_fake
''')
path.write_text(src); py_compile.compile(str(path), doraise=True); print("patched + compiled:", path)
