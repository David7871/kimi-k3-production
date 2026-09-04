# causal_conv1d_update token-load hoist (port of vLLM #579)

`causal_conv1d.py` — `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`
from branch `agent/kimi-k3-conv1d-update-hoist-20260903` (vLLM #579 applied
to `dev/infernal-invocation`; the op file is identical between the GLM-5.3
and Kimi-K3 lineages and the served tree). Width-4 decode/spec updates
(1-8 tokens) load every token row before the loop and use a 64-channel
tile; bitwise identical; `VLLM_CAUSAL_CONV1D_UPDATE_HOIST=0` restores the
generic loop.

Kimi-K3: 69 KDA layers call it on the speculative-decode path (K=3 draft,
4 tokens per request). Expected ~0.2 ms per decode step. Tests: 272 passed
in the production image. Not deployed.
