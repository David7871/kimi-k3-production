# W4A8 MXFP8 expert-entry deployment overlay

Offline-qualified for the active Kimi K3 TP8/DCP8 remote-DFlash profile on 2026-08-30.

Source commits:

- B12X: `fed066aace44a2f6cdf8e8b7651972ee9ce18ae1`
- vLLM: `9acf420e1fa28672406c90e0791adc947ad729d0`

Apply with:

```bash
docker compose \
  -f /home/g0san/kimi-k3-production/candidates/all-compatible-20260829/runtime/compose.tp8-dflash-flashkda-320k-w4a8.yml \
  -f /home/g0san/kimi-k3-production/research/mxfp8-entry-20260830/compose.mxfp8-entry.override.yml \
  up -d --force-recreate kimi-k3-production
```

Rollback uses the proven base Compose file without the override.

## Files

- `kquant_hybrid.py`: `625703bbf38bb8e508365081e41e498fb02b9583960b30ec4810494f6a8348f4`
- `b12x_fused_moe_impl.py`: `520854b2062775d3447d2f9305fe8ae4d5e6e5a8ca576ec12ebb36221454fcc6`
- `b12x_dynamic.py`: `227ed7be2459161840190c70ed1b138ffeb11f998415594b99bf68c239991d88`
- `b12x_silu.py`: `cda6cffff5285786e59a67f7c34cdb996a885e421a3ffbe5a659120662ec9920`
- `b12x_situ.py`: `4ade7b2a423010bf5b963727582957bae0232c7f8dec8b77c3b7c15861cf2391`
- `b12x_relu2.py`: `2dbc463cedca3729c75725838a07d26eab036635e6125ec59201da07b5a3d754`

See `/home/g0san/kimi-k3-production/research/mxfp8-entry-20260830/RESULTS-20260830.md` for the precision, performance, memory, graph-replay, and rollback evidence.
