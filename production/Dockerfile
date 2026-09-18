# Exact inspected production image, published as runtime-base-20260918.
# The two overlays and original launchers are copied from the live container.
# The native extension is not recompiled.
FROM ghcr.io/myshytf/kimi-k3@sha256:9a0a5818b664e263ca68aba772f76ae2b2948e97e42d6bbfd0c3859507e76b30
COPY overlays/kda.py /opt/infernal-invocation/vllm/vllm/models/kimi_k3/nvidia/kda.py
COPY overlays/_C_k3kda.abi3.so /opt/k3kda/_C_k3kda.abi3.so
COPY runtime/ /runtime/
COPY source-lock.json runtime-manifest.sha256 verify-image.py /recipe/
RUN /opt/venv/bin/python /recipe/verify-image.py
LABEL org.opencontainers.image.source="https://github.com/myshytf/kimi-k3-production" \
      org.opencontainers.image.title="Kimi-K3 QSRT-K2 exact runtime snapshot" \
      local.kimi-k3.vllm.revision="a3693719528811ac605dfa236321a53e5770b120" \
      local.kimi-k3.b12x.revision="0d31d259c6f0e6733190a6b3af9881b874f10241" \
      local.kimi-k3.lmcache.revision="98c8be0a655bdf5411a8a2335f21cb44a76083d7"
ENTRYPOINT ["/bin/bash", "/runtime/serve_tp9.sh"]
