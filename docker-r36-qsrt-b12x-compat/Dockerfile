ARG B12X_SOURCE_IMAGE=kimi-k3-production-issue75-dspark:20260823-r1
ARG BASE_IMAGE=voipmonitor/vllm@sha256:c41bf15095b2316c7335d305115ad26bab14ec4234f3109b1d1ebb807895a3ea

FROM ${B12X_SOURCE_IMAGE} AS b12x_compat
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# r36's installed vLLM passes the QSRT atoms-v2 descriptor to B12X, but the
# packaged B12X plan_weights ABI omits those keyword arguments. Carry forward
# the source-matched QSRT/route-pack package qualified by the issue-75 runtime.
RUN rm -rf \
      /opt/venv/lib/python3.12/site-packages/b12x \
      /opt/infernal-invocation/b12x/b12x \
    && mkdir -p \
      /opt/venv/lib/python3.12/site-packages \
      /opt/infernal-invocation/b12x
COPY --from=b12x_compat /opt/infernal-invocation/b12x/b12x /opt/venv/lib/python3.12/site-packages/b12x
COPY --from=b12x_compat /opt/infernal-invocation/b12x/b12x /opt/infernal-invocation/b12x/b12x
COPY verify_b12x_compat.py /opt/local-inference/verify_b12x_compat.py

RUN /opt/venv/bin/python -c 'import pathlib,shutil; roots=[pathlib.Path("/opt/venv/lib/python3.12/site-packages/b12x"),pathlib.Path("/opt/infernal-invocation/b12x/b12x")]; [shutil.rmtree(p) for r in roots for p in list(r.rglob("__pycache__")) if p.exists()]' \
    && /opt/venv/bin/python /opt/local-inference/verify_b12x_compat.py

LABEL org.opencontainers.image.title="Kimi-K3 r36 QSRT B12X compatibility runtime" \
      org.opencontainers.image.description="Upstream-aligned r36 runtime with the issue-75 QSRT atoms-v2 and stable route-pack B12X package" \
      local-inference.runtime.base-digest="sha256:c41bf15095b2316c7335d305115ad26bab14ec4234f3109b1d1ebb807895a3ea" \
      local-inference.runtime.b12x-compat-source-image="sha256:f18184728c27959de26cf11952d7a8b96e57b513a6badf8bf3e09fd4cda16a76" \
      local-inference.runtime.b12x-compat-tree-sha256="6e7a050c63068d827ced00032215f6ea9e1d49467a75a341023cf11537d32988" \
      local-inference.runtime.qsrt-plan-abi="qsrt_atoms_v2+k2_coupled_h512_h128"
