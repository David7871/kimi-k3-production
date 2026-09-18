# Kimi-K3 production serving specification

Status: **implemented; qualified** on the inspected reference host. The canonical
portable deployment is [production/README.md](../production/README.md), with
[source-lock.json](../production/source-lock.json) defining the exact source,
model, and binary identities. The independently packaged image has build/CPU
validation; it has not replaced the running service.

The current profile uses nine RTX PRO 6000 Blackwell GPUs with TP9/DCP9 and a
colocated TP9 DFlash2 draft. Target weights are QSRT K2 with existing MXFP8 dense
projections, BF16 model dtype, `fp8_ds_mla` target KV, and FP8 draft KV. Context
is configured to 1,048,576 tokens, with four concurrent requests, a 4,624-token
scheduler budget, 4,608-token recurrent/cache chunks, and 1,536-token prefix
matching. The LMCache host pool is 48 GiB; the disk tier has a 2457.6 GiB ceiling.

The runtime source identities are vLLM `a3693719528811ac605dfa236321a53e5770b120`,
B12X `0d31d259c6f0e6733190a6b3af9881b874f10241`, and LMCache
`98c8be0a655bdf5411a8a2335f21cb44a76083d7`. The vLLM identity is the production
base plus the KDA integration; LMCache includes the resident-L1 prefetch fix.
These supersede the earlier shared three-branch snapshot.

The deployed fused KDA kernel changes intermediate precision and reduction
order. Prior kernel checks reported smaller error against float64; model output
is not bit-identical to the previous chain, and its cache namespace is isolated.
The full configured context ceiling has not been retested. See the recipe's
validation limits and [packaging evidence](../production/validation.json).

Earlier TP8 / remote-draft instructions and measurements are historical exports.
Use the current recipe's single Compose file; the historical override chain is
not part of its deployment contract. No model weights, credentials, cache data,
or production traffic logs are included in the current package.
