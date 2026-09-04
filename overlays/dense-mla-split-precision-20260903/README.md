# Dense-MLA split precision (Kimi-K3 decode)

Served-lineage files of b12x branch `agent/kimi-k3-dense-mla-split-precision-20260903`
(`~/k3-longctx-b12x-mla-precision`, commit c3f0832, on the balanced-split
commit 00a14d3; master-lineage PR: b12x #292) and vLLM branch
`agent/kimi-k3-dense-mla-split-precision-20260903`
(`~/k3-longctx-vllm-mla-precision`, commit 7e7aed3ce, on 2ca9d200e;
master-lineage PR: vLLM #595).

| file | mount target | content |
|---|---|---|
| `_forward.py` | `b12x/attention/dense_mla/_forward.py` | split ranges: one split for requests within `single_split_chunks` scanned chunks, balanced ranges above |
| `_scratch.py` | `b12x/attention/dense_mla/_scratch.py` | `Caps.partial_dtype`, `Caps.single_split_chunks`, partial scratch sized by dtype |
| `_kernel.py` | `b12x/attention/dense_mla/_kernel.py` | compile keys and operands for both options (spec versions forward 6 / merge 5) |
| `_merge.py` | `b12x/attention/dense_mla/_merge.py` | float32 partial loads (`ld.global.v4.f32`) beside the bf16 path |
| `b12x_mla.py` | `vllm/v1/attention/backends/mla/b12x_mla.py` | `VLLM_K3_DENSE_MLA_PARTIAL_DTYPE` / `VLLM_K3_DENSE_MLA_SINGLE_SPLIT_CHUNKS` into every plan; eager launches use one split within the threshold |
| `envs.py` | `vllm/envs.py` | `overlays/dcp-gather-pipeline-20260902/envs.dma4.py` plus the two variables |

Mechanism: the dense MLA kernel stores each split's normalized partial
output in bf16 before the merge, so a result merged from several splits is
rounded twice while a one-split result is rounded once. With balanced ranges
(b12x #291) every request was merged, which made requests that the
fixed-range kernel scanned with one split (≤ 44 live chunks per rank,
≈ ≤ 22.5k tokens of total context at DCP 8) 9–19 % less accurate
(fp64-relative L2) while longer requests became 1–12 % more accurate.
Measurements: `research/longctx-20260902/ASSESSMENT-dense-mla-balanced-splits-20260903.md`,
`dense_mla_split_precision_harness.log`, `dense_mla_split_precision_bench.log`.

Options (engine environment):
- `VLLM_K3_DENSE_MLA_SINGLE_SPLIT_CHUNKS=-1` (plan run = 44): bit-identical to
  the fixed-range kernel for ≤ 44 live chunks per rank (`torch.equal` on
  output and LSE at 1…2,816 local tokens in both logit regimes); balanced
  ranges above. `0` balances every request (kernel 47–125 → 8–9 µs per
  layer at 1k–2.8k local tokens; not bit-identical there).
- `VLLM_K3_DENSE_MLA_PARTIAL_DTYPE=fp32`: merged results rounded once; 8–25 %
  lower fp64-relative error than the fixed-range kernel at 2,817–131,072
  local tokens (`bf16` keeps the served merge numerics).

Status: implemented and qualified offline (b12x tests 15/15 served lineage,
28/28 master lineage; vLLM b12x_mla tests 45/45); production deployment via
`candidates/k3-dense-mla-precision-20260903/deploy.sh` requires approval.
