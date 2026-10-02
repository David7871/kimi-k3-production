# baseline: current production lane (Moet off)

- Image: `kimi-k3-up-ray:local` (`sha256:c8c3c9f5…`, see `../../IMAGE.txt`) plus the l3 overlays from `restore.sh`.
- Env: `int6.envstr`. KV is fp8, 16 GiB/GPU, 13,080,081 tokens.
- Launch: `bash ../launch.sh baseline`.
- Measured on `:local` with this env: C16 326.3 and C256 1200.3, from the rebalance A arm `../rebalance-kv/results/rb-a` at 15:00Z.
- The Moet control at 19:23Z (`../moet/results/moet-off`) used this same env but on the `moet-mvp` image with the flag unset. It gave C16 334.9 / 334.4 and GSM8K-200 196/200. Use that row when comparing against Moet; see `../moet/README.md`.
