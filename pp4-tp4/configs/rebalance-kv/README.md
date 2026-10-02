# rebalance-kv: PP layer partition 24,24,24,21 (+16.7% KV)

- Delta: `VLLM_PP_LAYER_PARTITION=24,24,24,21`. The default is 23,23,24,23, which puts 7 MLA layers on stage 3. The hybrid KV grouping then clamps every GPU to 6/7 of its blocks. 24,24,24,21 is the only split with 6 MLA layers on every stage.
- Launch: `bash ../launch.sh rebalance-kv`.
- Measured 2026-10-01 15:00–15:41Z (`results/rb-a|rb-b|rb-c`, scripts `k3-rebalance-ab.sh` / `k3-rebalance-c.sh`):

| arm | split | C16 | C256 | KV tokens |
|---|---|---|---|---|
| rb-a | 23,23,24,23 (default) | 326.3 | 1200.3 | 13,080,081 |
| rb-b | **24,24,24,21** | 322.3 (−1.2%) | 1197.8 (noise) | **15,258,588 (+16.67%)** |
| rb-c | 23,24,24,22 | 322.5 | 1204.3 | 13,080,081 |

- All arms passed the probe. Max concurrency per 262k request goes from 49.90x to 58.21x.
- Use it when KV capacity or long-context headroom matters. It is not a throughput lever. Full analysis: `../../levers/rebalance/RESULT.md`.
