#!/usr/bin/env python3
"""Memoize b12x TP-MoE execution planning (marker: kimi-k3-moe-plan-cache).

`b12x_moe_fp4` re-lowers a `TPMoEPlan` on every call, and the lowering walks
every token count from 1 to `num_tokens` in Python (`_dynamic_capacity_geometry`),
costing about 5.5 us per token per layer (8.4 ms at m=1536). The plan is a
frozen dataclass that depends only on hashable call arguments, the weight
plan object, and B12X_* environment variables, so it is memoized here.
`B12X_MOE_PLAN_CACHE_SIZE=0` disables the cache.
"""
import re, sys, py_compile
from pathlib import Path

path = Path(sys.argv[1])
src = path.read_text()
MARKER = "kimi-k3-moe-plan-cache"
if MARKER in src:
    print("already patched"); sys.exit(0)

def replace_once(old, new):
    global src
    n = src.count(old)
    assert n == 1, f"anchor count {n} for {old[:60]!r}"
    src = src.replace(old, new)

replace_once(
    "from collections.abc import Mapping, Sequence\n",
    "from collections.abc import Mapping, Sequence\n"
    f"import functools  # {MARKER}\n"
    f"import threading  # {MARKER}\n"
    f"from collections import OrderedDict  # {MARKER}\n",
)
replace_once(
    "def _dynamic_capacity_geometry(\n",
    f"@functools.lru_cache(maxsize=8192)  # {MARKER}: pure function of its keyword arguments\n"
    "def _dynamic_capacity_geometry(\n",
)
replace_once("def plan_tp_moe_execution(\n", "def _plan_tp_moe_execution_uncached(\n")
wrapper = f'''
# {MARKER}: begin
_MOE_PLAN_CACHE_MAX = int(os.environ.get("B12X_MOE_PLAN_CACHE_SIZE", "8192") or 0)
_MOE_PLAN_CACHE: "OrderedDict[tuple, tuple[object, TPMoEPlan]]" = OrderedDict()
_MOE_PLAN_CACHE_LOCK = threading.Lock()
_MOE_PLAN_CACHE_STATS = {{"hits": 0, "misses": 0}}


def _b12x_env_signature() -> tuple:
    """B12X_* environment values that steer plan lowering (part of the cache key)."""
    return tuple(sorted(item for item in os.environ.items() if item[0].startswith("B12X_")))


def clear_moe_plan_cache() -> None:
    with _MOE_PLAN_CACHE_LOCK:
        _MOE_PLAN_CACHE.clear()
    _dynamic_capacity_geometry.cache_clear()


def plan_tp_moe_execution(
    *,
    num_tokens: int,
    num_topk: int,
    device: torch.device | str,
    weight_plan: MoEWeightPreparationPlan,
    quant_mode: str,
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    apply_router_weight_on_input: bool = False,
    deterministic_output: bool | None = None,
) -> TPMoEPlan:
    """Memoized `_plan_tp_moe_execution_uncached`.

    The lowered plan is a frozen dataclass determined by the call arguments,
    the identity of the immutable weight-preparation plan, and B12X_*
    environment variables. Entries keep a reference to the weight plan so an
    `id()` can never be reused by a different live object.
    """
    if _MOE_PLAN_CACHE_MAX <= 0:
        return _plan_tp_moe_execution_uncached(
            num_tokens=num_tokens, num_topk=num_topk, device=device,
            weight_plan=weight_plan, quant_mode=quant_mode,
            swiglu_limit=swiglu_limit, swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            apply_router_weight_on_input=apply_router_weight_on_input,
            deterministic_output=deterministic_output,
        )
    key = (
        int(num_tokens), int(num_topk), str(torch.device(device)), id(weight_plan),
        quant_mode, swiglu_limit, swiglu_alpha, swiglu_beta,
        bool(apply_router_weight_on_input), deterministic_output,
        _b12x_env_signature(),
    )
    with _MOE_PLAN_CACHE_LOCK:
        entry = _MOE_PLAN_CACHE.get(key)
        if entry is not None and entry[0] is weight_plan:
            _MOE_PLAN_CACHE.move_to_end(key)
            _MOE_PLAN_CACHE_STATS["hits"] += 1
            return entry[1]
    plan = _plan_tp_moe_execution_uncached(
        num_tokens=num_tokens, num_topk=num_topk, device=device,
        weight_plan=weight_plan, quant_mode=quant_mode,
        swiglu_limit=swiglu_limit, swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        apply_router_weight_on_input=apply_router_weight_on_input,
        deterministic_output=deterministic_output,
    )
    with _MOE_PLAN_CACHE_LOCK:
        _MOE_PLAN_CACHE_STATS["misses"] += 1
        _MOE_PLAN_CACHE[key] = (weight_plan, plan)
        while len(_MOE_PLAN_CACHE) > _MOE_PLAN_CACHE_MAX:
            _MOE_PLAN_CACHE.popitem(last=False)
    return plan
# {MARKER}: end


'''
replace_once("\n\ndef _validate_workspace(\n", wrapper + "def _validate_workspace(\n")
path.write_text(src)
py_compile.compile(str(path), doraise=True)
print("patched + compiled:", path)
