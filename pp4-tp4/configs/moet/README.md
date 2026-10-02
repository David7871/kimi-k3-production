# moet: Moet 2-bit routed experts (vLLM-Moet `moe_w2_mm`)

## What was measured: two separate runs, not one A/B script

The ON and OFF arms came from different scripts and slightly different images. Both used the same base env (`../baseline/int6.envstr`), the same launcher (`scripts/k3-arm-moet.sh`, EXEC=up2l3), the same hosts and the same C16 bench. They ran back to back while holding the lane lock.

| arm | script | image | Moet flags | time (UTC) | C16 tok/s | GSM8K-200 | probe |
|---|---|---|---|---|---|---|---|
| OFF (`results/moet-off`) | `scripts/moet-ab.sh`, step (d) | `kimi-k3-up-ray:moet-mvp` `sha256:d35a7e4a…` | none (`$INT6` only) | 19:23–19:35 | 334.9 / 334.4 | 196 | PASS |
| ON (`results/moet-on`) | `scripts/moet-on2.sh`, ON only | `kimi-k3-up-ray:moet-mvp2` `sha256:14757877…` | `$INT6` + `moet.envstr` | 19:35–19:54 | 374.4 / 371.8 | 196 | PASS |

- **Result:** C16 is +11.5% (373.1 vs 334.65). McNemar, computed by `moet-on2.sh` against the earlier OFF jsonl, is n=200 with on_only=0, off_only=0, both=196.
- **Image difference:** the two images differ in exactly one file, `utils/moe_w2_cubit.py`. `moet-mvp` has the md5 9d1f49d1 copy, saved as `build/moet-mvp-v1/moe_w2_cubit.py`. `moet-mvp2` has the resident-fit PP fix, md5 a214f87c, at `build/payload/.../moe_w2_cubit.py`. I checked `kquant_hybrid.py`, the other `moe_w2_*.py` files and all 6 cubins by md5 inside both images, and they are identical.
- **Not verified:** that the OFF arm on `moet-mvp` behaves byte-for-byte like `kimi-k3-up-ray:local`, i.e. that the flag-off path never touches the Moet modules. This is asserted in the script comments but was never measured separately.
- **Why `moet-ab.sh` didn't produce an ON result:** its ON arm (19:20) died at boot on `moet-mvp`. That failure is what `moet-mvp2` fixes. The 19:04 attempt aborted at the op-test gate, which was then fixed to compare kernel error against reference rather than quantization error. All of this is in `results/moet-ab.log`. `scripts/*.bak-*` are the earlier versions of `moet-ab.sh`.

## Reproduce

- **Build:** `docker build -f build/Dockerfile.moet-mvp -t kimi-k3-up-ray:moet-mvp2 build/` on top of `kimi-k3-up-ray:local`. For the exact OFF image (`moet-mvp`), first copy `build/moet-mvp-v1/moe_w2_cubit.py` over `build/payload/vllm/model_executor/layers/quantization/utils/moe_w2_cubit.py`.
- **Packs:** `/data/models/k3-moet-packs` are 714 GB and are not committed. Regenerate them with `build/convert_moet_packs.py` (driven by `build/conv-all.sh`). The checkpoint ID is `k3-mxfp4-moet-v1`.
- **ON:** `bash ../launch.sh moet`. This uses `moet-mvp2`, `$INT6 + moet.envstr`, and `scripts/k3-arm-moet.sh`, which forwards `UP_IMAGE` to the rtx-02 worker.
- **OFF, exactly as measured:** `bash ../launch.sh moet-off` (`moet-mvp`, `$INT6` only, same launcher).
- **GSM8K:** `python3 scripts/k3_gsm8k.py 8001 200 out.jsonl 8`.

## Notes

- `VLLM_MOE_W2_DELTA=0`, because the FP4 delta-tier cubins are not in the image.
- Each GPU uses about 5.5 GiB more memory (weights +4.6–5.0, graphs +0.75), from the two head.logs. KV stays pinned at 13,080,081.
- Open items: C256 has never been benched. `VLLM_MOE_W2_PREFILL_T=96` is below the 512 graph-capture size, and 512 is a candidate fix. Token inflation (+8–11% upstream) has not been measured, because the harness records no completion-token counts.
