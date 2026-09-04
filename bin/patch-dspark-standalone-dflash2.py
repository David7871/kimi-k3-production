#!/usr/bin/env python3
"""Teach the standalone draft server to serve a DFlash2 drafter.

lightseekorg/kimi-k3-dflash2 ships `DFlash2DraftModel`: five auxiliary taps
instead of six, block size 8, and per-layer grouped convolutions plus a
candidate selector on top of the DFlash decoder. Its implementation
(qwen3_dflash2.py) subclasses DFlashQwen3{DecoderLayer,Model,ForCausalLM}, and
those three base classes are byte-for-byte compatible between the GLM-5.3
image that carries DFlash2 and this one -- same __init__ and forward
signatures, same helper set -- so the drafter drops in as a subclass rather
than a fork.

Two things this does NOT need to change, both checked rather than assumed:

* `dflash_config.attention_mode: "mla"` is not read anywhere in either
  qwen3_dflash.py or qwen3_dflash2.py. It is metadata for a different serving
  stack; here the drafter uses the ordinary DFlash attention path, so the
  server keeps TRITON_ATTN and the `layer.self_attn.attn` KV layout.
* Model construction: load_dflash_model resolves the architecture through the
  model registry, so registering DFlash2DraftModel is enough.

* The method name: vLLM's SpeculativeConfig.method is a closed Literal with no
  "dflash2" in it -- in this image or in the GLM one that does serve DFlash2.
  That image keeps method="dflash" and routes on the architecture declared in
  the draft's config.json. So does this.

The standalone server additionally needs a deterministic version of the
DFlash2 selector walk.  The colocated upstream speculator has that walk, but
the remote K3 protocol returns tokens rather than the draft distribution.  We
therefore use the selector's greedy path; this preserves the existing target
protocol and block verifier while exercising the DFlash2 pairwise scorer.

The remaining model-loader changes are to register the architecture and
backport the two subclass hooks the DFlash2 classes rely on. DFlash2Qwen3Model sets
`decoder_layer_cls` and DFlash2Qwen3ForCausalLM sets `model_cls`; the GLM base
reads both (upstream PR#52816) while this one still names the classes
directly, so the subclass would silently build plain DFlash layers -- which is
exactly how it fails: "no module or parameter named 'candidate_selector'".
Defaulting each hook to the class it replaces leaves the existing DFlash path
byte-identical.

Idempotent, and inert if the files are missing.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("K3PATCH_VLLM_ROOT", "/opt/k3-dspark-3090/source/vllm"))
REGISTRY = ROOT / "model_executor/models/registry.py"
DFLASH = ROOT / "model_executor/models/qwen3_dflash.py"
K3_DSPARK = ROOT / "models/kimi_k3/nvidia/dspark_mla.py"
STANDALONE = ROOT / "entrypoints/k3_dspark_standalone.py"
RPC = ROOT / "entrypoints/k3_dspark_rpc.py"
MARKER = "k3-dflash2-standalone"
K3_DSPARK_MARKER = "k3-dflash2-k3-mla-hooks"
STANDALONE_MARKER = "k3-dflash2-mla-runtime"
RPC_MARKER = "k3-dflash2-selector-walk"

REGISTRY_ANCHOR = '    "DFlashDraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),\n'
REGISTRY_ADDITION = (
    '    "DFlash2DraftModel": ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM"),  '
    "# k3-dflash2-standalone\n"
)

# Every method test in the server. DFlash2 shares DFlash's runtime shape, so
# each of these becomes a membership test instead of an equality one.
def patch_registry() -> None:
    if not REGISTRY.is_file():
        print(f"{REGISTRY}: absent, skipping")
        return
    source = REGISTRY.read_text()
    if MARKER in source:
        print(f"{REGISTRY}: already patched")
        return
    if REGISTRY_ANCHOR not in source:
        raise SystemExit(f"{REGISTRY}: DFlashDraftModel entry not found")
    REGISTRY.write_text(source.replace(REGISTRY_ANCHOR, REGISTRY_ANCHOR + REGISTRY_ADDITION, 1))
    print(f"{REGISTRY}: DFlash2DraftModel registered")


HOOKS = [
    (
        "class DFlashQwen3Model(nn.Module):\n",
        "class DFlashQwen3Model(nn.Module):\n"
        "    # k3-dflash2-standalone: subclass hook (upstream PR#52816).\n"
        "    decoder_layer_cls = DFlashQwen3DecoderLayer\n",
    ),
    (
        "                DFlashQwen3DecoderLayer(\n",
        "                self.decoder_layer_cls(  # k3-dflash2-standalone\n",
    ),
    (
        "class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):\n",
        "class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):\n"
        "    # k3-dflash2-standalone: subclass hook (upstream PR#52816).\n"
        "    model_cls = DFlashQwen3Model\n",
    ),
    (
        "        self.model = DFlashQwen3Model(\n",
        "        self.model = self.model_cls(  # k3-dflash2-standalone\n",
    ),
]


def patch_dflash_hooks() -> None:
    if not DFLASH.is_file():
        print(f"{DFLASH}: absent, skipping")
        return
    source = DFLASH.read_text()
    if MARKER in source:
        print(f"{DFLASH}: already patched")
        return
    for old, new in HOOKS:
        if old not in source:
            raise SystemExit(f"{DFLASH}: anchor not found: {old.strip()[:60]!r}")
        source = source.replace(old, new, 1)
    DFLASH.write_text(source)
    print(f"{DFLASH}: subclass hooks backported")


K3_DSPARK_HOOKS = [
    (
        "class K3DSparkModel(nn.Module):\n",
        "class K3DSparkModel(nn.Module):\n"
        "    # k3-dflash2-k3-mla-hooks: subclass hook.\n"
        "    decoder_layer_cls = K3DSparkDecoderLayer\n",
    ),
    (
        "                K3DSparkDecoderLayer(\n",
        "                self.decoder_layer_cls(  # k3-dflash2-k3-mla-hooks\n",
    ),
    (
        "class K3DSparkForCausalLM(nn.Module):\n",
        "class K3DSparkForCausalLM(nn.Module):\n"
        "    # k3-dflash2-k3-mla-hooks: subclass hook.\n"
        "    model_cls = K3DSparkModel\n",
    ),
    (
        "        self.model = K3DSparkModel(\n",
        "        self.model = self.model_cls(  # k3-dflash2-k3-mla-hooks\n",
    ),
]


def patch_k3_dspark_hooks() -> None:
    if not K3_DSPARK.is_file():
        raise SystemExit(f"K3 MLA draft implementation is missing: {K3_DSPARK}")
    source = K3_DSPARK.read_text()
    if K3_DSPARK_MARKER in source:
        print(f"{K3_DSPARK}: already patched")
        return
    for old, new in K3_DSPARK_HOOKS:
        if old not in source:
            raise SystemExit(f"{K3_DSPARK}: hook anchor not found: {old.strip()!r}")
        source = source.replace(old, new, 1)
    compile(source, str(K3_DSPARK), "exec")
    K3_DSPARK.write_text(source)
    print(f"{K3_DSPARK}: DFlash2 MLA subclass hooks backported")


STANDALONE_EDITS = [
    (
        'def _build_vllm_config(args: argparse.Namespace):\n'
        '    attention_backend = "TRITON_ATTN" if args.method == "dflash" else "TRITON_MLA"\n',
        'def _build_vllm_config(args: argparse.Namespace):\n'
        '    draft_metadata = json.loads((args.draft_model / "config.json").read_text())\n'
        '    nested = draft_metadata.get("dflash_config") or {}\n'
        '    args.dflash2_mla = bool(  # k3-dflash2-mla-runtime\n'
        '        args.method == "dflash"\n'
        '        and (\n'
        '            "DFlash2DraftModel" in draft_metadata.get("architectures", [])\n'
        '            or "selector_rank" in nested\n'
        '        )\n'
        '        and nested.get("attention_mode") == "mla"\n'
        '    )\n'
        '    attention_backend = (\n'
        '        "TRITON_MLA"\n'
        '        if args.method != "dflash" or args.dflash2_mla\n'
        '        else "TRITON_ATTN"\n'
        '    )\n',
    ),
    (
        '    if method == "dflash":\n'
        '        attentions = [layer.self_attn.attn for layer in model.model.layers]\n'
        '    else:\n'
        '        attentions = [layer.self_attn for layer in model.model.layers]\n',
        '    uses_mla = method != "dflash" or not hasattr(\n'
        '        model.model.layers[0].self_attn, "attn"\n'
        '    )\n'
        '    if uses_mla:\n'
        '        attentions = [layer.self_attn for layer in model.model.layers]\n'
        '    else:\n'
        '        attentions = [layer.self_attn.attn for layer in model.model.layers]\n',
    ),
    (
        '    if method == "dflash":\n'
        '        cache_shapes = {\n',
        '    if not uses_mla:\n'
        '        cache_shapes = {\n',
    ),
    (
        '    for attn in attentions:\n'
        '        if method == "dflash":\n'
        '            # TritonAttention exposes logical B,H,N,2D while its preferred\n',
        '    for attn in attentions:\n'
        '        if not uses_mla:\n'
        '            # TritonAttention exposes logical B,H,N,2D while its preferred\n',
    ),
    (
        '        if args.method == "dflash":\n'
        '            from vllm.v1.worker.gpu.spec_decode.dflash.utils import (\n',
        '        if args.method == "dflash" and not args.dflash2_mla:\n'
        '            from vllm.v1.worker.gpu.spec_decode.dflash.utils import (\n',
    ),
    (
        '        _build_dflash_metadata_builder(model, draft_vllm_config, device)\n'
        '        if args.method == "dflash"\n'
        '        else None\n',
        '        _build_dflash_metadata_builder(model, draft_vllm_config, device)\n'
        '        if args.method == "dflash" and not args.dflash2_mla\n'
        '        else None\n',
    ),
    (
        'def _run_eager_smoke(runtime: StandaloneRuntime, device: torch.device) -> int:\n'
        '    if runtime.method == "dflash":\n'
        '        return _run_dflash_eager_smoke(runtime, device)\n',
        'def _run_eager_smoke(runtime: StandaloneRuntime, device: torch.device) -> int:\n'
        '    if runtime.method == "dflash" and runtime.attn_metadata_builder is not None:\n'
        '        return _run_dflash_eager_smoke(runtime, device)\n',
    ),
    (
        '    sample_from_anchor = bool(getattr(draft_config, "sample_from_anchor", True))\n',
        '    sample_from_anchor = bool(\n'
        '        getattr(draft_config, "sample_from_anchor", runtime.method != "dflash")\n'
        '    )\n',
    ),
    (
        '    base_logits = model.compute_draft_logits(hidden[-1:])\n'
        '    markov = model.markov_bias(model.markov_embed(input_ids[-1:]))\n'
        '    token = int((base_logits + markov).argmax(dim=-1).item())\n',
        '    if runtime.method == "dflash":\n'
        '        sample_hidden = hidden.view(1, query_len, -1)[:, 1:]\n'
        '        candidate_ids, unary_logits = model.compute_candidates(\n'
        '            sample_hidden.flatten(0, 1)\n'
        '        )\n'
        '        top_k = int(model.model.candidate_selector.top_k)\n'
        '        candidate_ids = candidate_ids.view(1, query_len - 1, top_k)\n'
        '        unary_logits = unary_logits.view_as(candidate_ids)\n'
        '        scores = model.model.candidate_selector(\n'
        '            candidate_ids, unary_logits, sample_hidden, input_ids[:1]\n'
        '        )\n'
        '        choice = scores[:, 0, 0].argmax(dim=-1, keepdim=True)\n'
        '        token = int(candidate_ids[:, 0].gather(1, choice).item())\n'
        '    else:\n'
        '        base_logits = model.compute_draft_logits(hidden[-1:])\n'
        '        markov = model.markov_bias(model.markov_embed(input_ids[-1:]))\n'
        '        token = int((base_logits + markov).argmax(dim=-1).item())\n',
    ),
]


def patch_standalone_mla_runtime() -> None:
    if not STANDALONE.is_file():
        raise SystemExit(f"standalone draft runtime is missing: {STANDALONE}")
    source = STANDALONE.read_text()
    if STANDALONE_MARKER in source:
        print(f"{STANDALONE}: already patched")
        return
    for old, new in STANDALONE_EDITS:
        if old not in source:
            raise SystemExit(
                f"{STANDALONE}: DFlash2 MLA anchor not found: {old[:100]!r}"
            )
        source = source.replace(old, new, 1)
    compile(source, str(STANDALONE), "exec")
    STANDALONE.write_text(source)
    print(f"{STANDALONE}: DFlash2 MLA runtime path backported")


RPC_CLASS_ANCHOR = (
    "        self._cuda_graphs: dict[tuple[int, int], _DraftCudaGraphState] = {}\n"
    "\n"
    "    def _make_cuda_graph_state(\n"
)
RPC_CLASS_REPLACEMENT = (
    "        self._cuda_graphs: dict[tuple[int, int], _DraftCudaGraphState] = {}\n"
    "\n"
    "    def _select_dflash_tokens(  # k3-dflash2-selector-walk\n"
    "        self,\n"
    "        *,\n"
    "        hidden: torch.Tensor,\n"
    "        batch_size: int,\n"
    "        num_speculative_tokens: int,\n"
    "        anchor_token_ids: torch.Tensor,\n"
    "    ) -> tuple[torch.Tensor, torch.Tensor]:\n"
    "        query_len = 1 + num_speculative_tokens\n"
    "        sample_hidden = hidden.view(batch_size, query_len, -1)[:, 1:]\n"
    "        flat_hidden = sample_hidden.reshape(\n"
    "            batch_size * num_speculative_tokens, -1\n"
    "        )\n"
    "        if not hasattr(self.model, \"compute_candidates\"):\n"
    "            logits = self.model.compute_logits(flat_hidden)\n"
    "            return (\n"
    "                logits.argmax(dim=-1).view(\n"
    "                    batch_size, num_speculative_tokens\n"
    "                ),\n"
    "                logits,\n"
    "            )\n"
    "\n"
    "        candidate_ids, unary_logits = self.model.compute_candidates(flat_hidden)\n"
    "        top_k = int(self.model.model.candidate_selector.top_k)\n"
    "        candidate_ids = candidate_ids.view(\n"
    "            batch_size, num_speculative_tokens, top_k\n"
    "        )\n"
    "        unary_logits = unary_logits.view_as(candidate_ids)\n"
    "        scores = self.model.model.candidate_selector(\n"
    "            candidate_ids,\n"
    "            unary_logits,\n"
    "            sample_hidden,\n"
    "            anchor_token_ids,\n"
    "        )\n"
    "        rows = torch.arange(batch_size, dtype=torch.long, device=self.device)\n"
    "        previous = torch.zeros(batch_size, dtype=torch.long, device=self.device)\n"
    "        selected: list[torch.Tensor] = []\n"
    "        for step in range(num_speculative_tokens):\n"
    "            choice = scores[:, step][rows, previous].argmax(dim=-1)\n"
    "            selected.append(\n"
    "                candidate_ids[:, step].gather(1, choice[:, None]).squeeze(1)\n"
    "            )\n"
    "            previous = choice\n"
    "        return torch.stack(selected, dim=1), scores\n"
    "\n"
    "    def _make_cuda_graph_state(\n"
)

RPC_GRAPH_OLD = """        if self.method == "dflash":
            sample_hidden = hidden.view(state.batch_size, state.query_len, -1)[:, 1:]
            logits = self.model.compute_logits(
                sample_hidden.reshape(
                    state.batch_size * state.num_speculative_tokens, -1
                )
            )
            state.output_tokens.copy_(
                logits.argmax(dim=-1).view(
                    state.batch_size, state.num_speculative_tokens
                )
            )
"""
RPC_GRAPH_NEW = """        if self.method == "dflash":
            tokens, logits = self._select_dflash_tokens(
                hidden=hidden,
                batch_size=state.batch_size,
                num_speculative_tokens=state.num_speculative_tokens,
                anchor_token_ids=state.input_ids.view(
                    state.batch_size, state.query_len
                )[:, 0],
            )
            state.output_tokens.copy_(tokens)
"""

RPC_EAGER_OLD = """        if self.method == "dflash":
            sample_hidden = hidden.view(batch_size, query_len, -1)[:, 1:]
            return (
                self.model.compute_logits(
                    sample_hidden.reshape(batch_size * num_speculative_tokens, -1)
                )
                .argmax(dim=-1)
                .view(batch_size, num_speculative_tokens)
            )
"""
RPC_EAGER_NEW = """        if self.method == "dflash":
            tokens, _ = self._select_dflash_tokens(
                hidden=hidden,
                batch_size=batch_size,
                num_speculative_tokens=num_speculative_tokens,
                anchor_token_ids=torch.tensor(
                    anchor_token_ids, dtype=torch.int64, device=self.device
                ),
            )
            return tokens
"""

RPC_ATTENTION_EDITS = [
    (
        "        self.method = runtime.method\n"
        "        self.device = device\n",
        "        self.method = runtime.method\n"
        "        self.uses_mla_attention = (\n"
        "            self.method != \"dflash\"\n"
        "            or runtime.attn_metadata_builder is None\n"
        "        )\n"
        "        self.device = device\n",
    ),
    (
        '        if self.method == "dflash" and self.runtime.attn_metadata_builder is None:\n',
        '        if (\n'
        '            self.method == "dflash"\n'
        '            and not self.uses_mla_attention\n'
        '            and self.runtime.attn_metadata_builder is None\n'
        '        ):\n',
    ),
    (
        '        if self.method == "dflash":\n'
        '            common = CommonAttentionMetadata(\n',
        '        if self.method == "dflash" and not self.uses_mla_attention:\n'
        '            common = CommonAttentionMetadata(\n',
    ),
    (
        '        if self.method == "dflash":\n'
        '            query_start_loc_cpu = torch.arange(\n',
        '        if self.method == "dflash" and not self.uses_mla_attention:\n'
        '            query_start_loc_cpu = torch.arange(\n',
    ),
]


def patch_rpc_selector() -> None:
    if not RPC.is_file():
        print(f"{RPC}: absent, skipping")
        return
    source = RPC.read_text()
    if RPC_MARKER in source:
        print(f"{RPC}: already patched")
        return
    edits = [
        *RPC_ATTENTION_EDITS,
        (RPC_CLASS_ANCHOR, RPC_CLASS_REPLACEMENT),
        (RPC_GRAPH_OLD, RPC_GRAPH_NEW),
        (RPC_EAGER_OLD, RPC_EAGER_NEW),
    ]
    for old, new in edits:
        if old not in source:
            raise SystemExit(f"{RPC}: selector anchor not found: {old[:80]!r}")
        source = source.replace(old, new, 1)
    compile(source, str(RPC), "exec")
    RPC.write_text(source)
    print(f"{RPC}: deterministic DFlash2 selector walk backported")


def main() -> None:
    patch_registry()
    patch_dflash_hooks()
    patch_k3_dspark_hooks()
    patch_standalone_mla_runtime()
    patch_rpc_selector()


if __name__ == "__main__":
    main()
