# Saved K3 PP4xTP4 serving configs (2026-10-01)

Each folder is one configuration of the lane described in `../README.md`. All of them sit on top of the lane snapshot that `restore.sh` installs, and share the base env `baseline/int6.envstr` (MNS 256, prefix caching, b12x PCIe allreduce, int6-best kernel flags, DCP b12x options). KV cache is fp8 with `--kv-cache-memory-bytes` 16 GiB/GPU unless a config says otherwise.

Boot any one of them on rtx-01 with `bash launch.sh <config> [conc] [tag]`.

| config | delta vs baseline | image | C16 tok/s | C256 tok/s | KV tokens | quality |
|---|---|---|---|---|---|---|
| `baseline` | none | `kimi-k3-up-ray:local` | 326.3 (rb-a, 15:00Z session) | 1200.3 (rb-a) | 13,080,081 | probe PASS |
| `moet-off` | none (measured Moet control) | `kimi-k3-up-ray:moet-mvp`, flag unset | 334.65 | not measured | 13,080,081 | GSM8K-200 196/200 |
| `moet-off2` | none (fu-off, README headline baseline) | `kimi-k3-up-ray:moet-mvp2`, flag unset | 335.1 | 1268.3 | 13,080,081 | probe PASS, GSM8K-200 197/200 |
| `moet` | Moet 2-bit `moe_w2_mm` routed experts | `kimi-k3-up-ray:moet-mvp2` | 373.1 (+11.5% vs moet-off) | not measured | 13,080,081 | GSM8K-200 196/200, McNemar 0/0 |
| `nvfp4-kv` | `K3_KVDT=nvfp4_ds_mla` | `kimi-k3-up-ray:local` | 314.9 (older config, no same-session control) | not measured | 13,080,081 (no gain) | probe coherent |
| `rebalance-kv` | `VLLM_PP_LAYER_PARTITION=24,24,24,21` | `kimi-k3-up-ray:local` | 322.3 (−1.2% vs rb-a, same session) | 1197.8 (noise) | **15,258,588 (+16.7%)** | probe PASS |

The configs are independent deltas. `MOET=1 bash launch.sh rebalance-kv` composes Moet with the rebalance, but that combination has never been booted.

The C16 numbers come from different sessions. Compare them only within a row's stated control: Moet against moet-off, rebalance against rb-a. ON and OFF for Moet were two separate scripts; `moet/README.md` has the details.
