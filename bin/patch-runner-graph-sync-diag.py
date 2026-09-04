#!/usr/bin/env python3
"""Env-gated sync after full-graph replay to measure true GPU step time.

entry_drain=43ms at speculator entry is ambiguous: pending stream work could
be the runner's post-sampling bookkeeping OR the still-running graph replay
itself (the 5.5ms full_graph_replay annotation may only be CPU launch time
on the async-output path). With K3_DIAG_SYNC_AFTER_GRAPH=1 the replay block
synchronizes before closing, making the annotation equal true GPU forward
time and shrinking entry_drain to the residual. Diagnostic only — the sync
serializes CPU/GPU overlap; do not leave enabled in production.
"""

import os
from pathlib import Path

PATH = Path(
    os.environ.get(
        "K3PATCH_MODEL_RUNNER",
        "/opt/infernal-invocation/vllm/vllm/v1/worker/gpu/model_runner.py",
    )
)
MARKER = "kimi-k3-graph-sync-diag"

OLD = '''            with record_function_or_nullcontext(
                f"vllm:v2/target/{phase}/full_graph_replay"
            ):
                self.kv_connector.pre_forward(scheduler_output)
                model_output = self.cudagraph_manager.run_fullgraph(batch_desc)
'''

NEW = '''            with record_function_or_nullcontext(
                f"vllm:v2/target/{phase}/full_graph_replay"
            ):
                self.kv_connector.pre_forward(scheduler_output)
                model_output = self.cudagraph_manager.run_fullgraph(batch_desc)
                # kimi-k3-graph-sync-diag: make the annotation equal true GPU
                # time when K3_DIAG_SYNC_AFTER_GRAPH=1 (diagnostic only).
                import os as _os

                if _os.environ.get("K3_DIAG_SYNC_AFTER_GRAPH") == "1":
                    torch.cuda.synchronize()
'''


def main() -> None:
    text = PATH.read_text()
    if MARKER in text:
        print(f"model_runner already patched: {MARKER}")
        return
    count = text.count(OLD)
    if count != 1:
        raise SystemExit(f"ERROR: expected exactly one anchor, found {count}")
    text = text.replace(OLD, NEW, 1)
    compile(text, str(PATH), "exec")
    PATH.write_text(text)
    print(f"model_runner patched: {MARKER}")


if __name__ == "__main__":
    main()
