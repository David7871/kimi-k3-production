# Top-k draft-distribution transport (target side)

`remote_speculator.py` = the served remote speculator
(`overlays/draft-async-ingest-20260901/remote_speculator.py`, vLLM PR #564
lineage) plus vLLM PR #585: `VLLM_K3_DRAFT_LOGITS_TOPK=k` requests the k
largest draft logits per proposed position from the draft server, stages
the frame (BF16 values + int32 indices) through pinned memory, broadcasts
the two tensors to the TP ranks and scatters them over the fill logit
`-1e4` into the dense draft-logits buffer that the Gumbel draft sampler
and the block rejection sampler read.

Exact with respect to the target: the draft token is sampled from, and
verified against, the same truncated distribution (speculative sampling
reproduces the target for any draft distribution). Only the acceptance
rate depends on the draft mass outside the top-k, which the draft server's
`K3_DRAFT_LOGITS_MASS_DIAG` logs on served traffic.

Evidence: `research/decode-step-20260902/topk_rejection_exactness.py`
(vLLM `rejection_sample`, 200k trials, dense vs top-8 truncated draft:
first-token histogram matches the target, chi-square 52-87 at 63 dof) and
the frame checks in `research/decode-step-20260902/out/draft_profile_varC_*.log`.

Mounted by `candidates/k3-draft-topk-logits-20260902/compose.draft-topk-logits.override.yml`
(target restart required; `K3_DRAFT_LOGITS_TOPK` selects k, 0 = dense).

Status: deployed 2026-09-02 20:40 KST with k=16384 (`candidates/k3-decode-20260902`,
chosen from the draft's served-traffic diagnostic: mass outside top-16384
mean 2.4e-4..1.4e-3, max 2.6e-2); acceptance length 2.45-2.51 afterwards.
