# Bulk copy for the W4A16 direct table

Served-lineage file from b12x PR #300 (`agent/kimi-k3-w4a16-smallm-crossover-20260903-served`, commit `1dfe1f27 (64305ff1 + mbarrier accounting fix)`). It equals the production W4A16 direct-table kernel plus one change: direct-table CTAs stage their 65,536-byte rate slice with one `cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes` transaction and wait on the CTA mbarrier before FC1; the modal-table path retains its parallel load/store loop. Table contents and arithmetic are unchanged.

Status: implemented, GPU-qualified, not deployed. Evidence: `research/k3-smallm-crossover-20260903/REPORT.md`.
