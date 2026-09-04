#!/usr/bin/env python3
"""Split the remote speculator's metadata_d2h span into attributable waits.

The decode-80 profile shows the verify step spends ~43 ms inside
``metadata_d2h`` — dominated by a ``current_stream().synchronize()`` whose
comment says it also drains "the preceding token-table update". This patch
adds two extra drain points so the timing log attributes the wait:

  entry_drain  — synchronize at propose() entry: GPU/UVA work queued by the
                 runner BEFORE the speculator ran (token-table update etc.)
  prep_drain   — synchronize after the anchor/where prep ops: work launched
                 by the prep itself
  copy_sync    — the original two tiny D2H copies + sync

Diagnostic only: adds two extra stream synchronizations per step (a few us
when the stream is empty). Remove after the campaign.
"""

import os
from pathlib import Path

PATH = Path(
    os.environ.get(
        "K3PATCH_REMOTE_SPECULATOR",
        "/opt/infernal-invocation/vllm/vllm/v1/worker/gpu/spec_decode/dspark/"
        "remote_speculator.py",
    )
)

MARKER = "kimi-k3-metadata-d2h-detail"

OLD_ENTRY = '''        started = time.perf_counter()
        if aux_hidden_states is None or len(aux_hidden_states) != self.num_aux_layers:
'''

NEW_ENTRY = '''        started = time.perf_counter()
        # kimi-k3-metadata-d2h-detail: drain the stream at entry to measure
        # how much pending GPU/UVA work the runner queued before propose().
        torch.cuda.current_stream(self.device).synchronize()
        self._detail_entry_drain = time.perf_counter() - started
        if aux_hidden_states is None or len(aux_hidden_states) != self.num_aux_layers:
'''

OLD_SYNC = '''        rejected_staging = self._rejected_staging[:num_reqs]
        anchor_staging = self._anchor_staging[:num_reqs]
        rejected_staging.copy_(num_rejected[:num_reqs], non_blocking=True)
        anchor_staging.copy_(anchor_tokens, non_blocking=True)
        torch.cuda.current_stream(self.device).synchronize()
'''

NEW_SYNC = '''        _detail_prep_t0 = time.perf_counter()
        torch.cuda.current_stream(self.device).synchronize()
        self._detail_prep_drain = time.perf_counter() - _detail_prep_t0
        rejected_staging = self._rejected_staging[:num_reqs]
        anchor_staging = self._anchor_staging[:num_reqs]
        rejected_staging.copy_(num_rejected[:num_reqs], non_blocking=True)
        anchor_staging.copy_(anchor_tokens, non_blocking=True)
        _detail_copy_t0 = time.perf_counter()
        torch.cuda.current_stream(self.device).synchronize()
        self._detail_copy_sync = time.perf_counter() - _detail_copy_t0
'''

OLD_TIMING = '''        timing_ms = {
            "metadata_d2h": (metadata_ready - started) * 1000,
'''

NEW_TIMING = '''        timing_ms = {
            "entry_drain": getattr(self, "_detail_entry_drain", 0.0) * 1000,
            "prep_drain": getattr(self, "_detail_prep_drain", 0.0) * 1000,
            "copy_sync": getattr(self, "_detail_copy_sync", 0.0) * 1000,
            "metadata_d2h": (metadata_ready - started) * 1000,
'''


def apply(text: str, old: str, new: str, what: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"ERROR: expected exactly one {what} anchor in {PATH}, found {count}"
        )
    return text.replace(old, new, 1)


def main() -> None:
    text = PATH.read_text()
    if MARKER in text:
        print(f"remote_speculator already patched: {MARKER}")
        return
    text = apply(text, OLD_ENTRY, NEW_ENTRY, "entry")
    text = apply(text, OLD_SYNC, NEW_SYNC, "sync")
    text = apply(text, OLD_TIMING, NEW_TIMING, "timing")
    compile(text, str(PATH), "exec")
    PATH.write_text(text)
    print(f"remote_speculator patched: {MARKER}")


if __name__ == "__main__":
    main()
