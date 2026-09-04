#!/usr/bin/env python3
"""Validate the SM120 FlashKDA runtime patch against the Triton backend."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from vllm.models.kimi_k3.nvidia.kda import _flashkda_prefill
from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
    chunk_kda_with_fused_gate,
)
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd


@dataclass(frozen=True)
class Case:
    name: str
    lengths: tuple[int, ...]
    checkpoint: int


def relative_rmse(reference: torch.Tensor, actual: torch.Tensor) -> tuple[float, float]:
    reference_f32 = reference.float()
    actual_f32 = actual.float()
    difference = reference_f32 - actual_f32
    maximum = difference.abs().max().item()
    rmse = difference.square().mean().sqrt().item()
    baseline = reference_f32.square().mean().sqrt().item()
    return maximum, rmse / (baseline + 1e-8)


def assert_close(
    label: str,
    reference: torch.Tensor,
    actual: torch.Tensor,
    tolerance: float = 0.015,
) -> None:
    maximum, ratio = relative_rmse(reference, actual)
    print(f"{label}: max_abs={maximum:.6f} rel_rmse={ratio:.6f}", flush=True)
    if not torch.isfinite(actual.float()).all():
        raise RuntimeError(f"{label}: non-finite output")
    if ratio >= tolerance:
        raise AssertionError(
            f"{label}: relative RMSE {ratio:.6f} >= {tolerance:.6f}"
        )


def cumulative_lengths(lengths: tuple[int, ...]) -> list[int]:
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return result


def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = v.dtype
    _, tokens, _, dim = q.shape
    q_f32 = q.float() * dim**-0.5
    k_f32 = k.float()
    v_f32 = v.float()
    gate_f32 = gate.float()
    beta_f32 = beta.float()
    state = initial_state.float().clone()
    output = torch.zeros_like(v_f32)
    for token in range(tokens):
        q_token = q_f32[:, token]
        k_token = k_f32[:, token]
        v_token = v_f32[:, token]
        gate_token = gate_f32[:, token]
        beta_token = beta_f32[:, token]
        state = state * gate_token[..., None].exp()
        state = state + torch.einsum(
            "bhk,bhv->bhkv",
            beta_token[..., None] * k_token,
            v_token - (k_token[..., None] * state).sum(-2),
        )
        output[:, token] = torch.einsum(
            "bhk,bhkv->bhv", q_token, state
        )
    return output.to(dtype), state


def naive_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    cumulative: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    q_normalized = l2norm_fwd(q.contiguous())
    k_normalized = l2norm_fwd(k.contiguous())
    gate = -3.0 * torch.sigmoid(
        a_log.exp()[None, None, :, None]
        * (raw_g.float() + dt_bias[None, None, :, :])
    )
    beta = raw_beta.float().sigmoid()
    outputs = []
    states = []
    for sequence, (start, end) in enumerate(zip(cumulative[:-1], cumulative[1:])):
        output, state = naive_recurrent_kda(
            q_normalized[:, start:end],
            k_normalized[:, start:end],
            v[:, start:end],
            gate[:, start:end],
            beta[:, start:end],
            initial_state[sequence : sequence + 1].transpose(-1, -2),
        )
        outputs.append(output)
        states.append(state)
    return (
        torch.cat(outputs, dim=1),
        torch.cat(states, dim=0).transpose(-1, -2).contiguous(),
    )


@torch.inference_mode()
def run_case(case: Case, repeats: int) -> None:
    batch, heads, dim = 1, 12, 128
    cumulative = cumulative_lengths(case.lengths)
    tokens = cumulative[-1]
    sequences = len(case.lengths)
    if any(length < case.checkpoint for length in case.lengths):
        raise ValueError("checkpoint must fit every sequence")

    torch.manual_seed(1000 + tokens + sequences)
    device = torch.device("cuda")
    q, k, v, raw_g = [
        torch.randn(
            batch,
            tokens,
            heads,
            dim,
            dtype=torch.bfloat16,
            device=device,
        )
        for _ in range(4)
    ]
    beta_storage = torch.randn(
        batch,
        tokens,
        2 * heads + 3,
        dtype=torch.bfloat16,
        device=device,
    )
    raw_beta = beta_storage[..., 1 : heads + 1]
    a_log = torch.randn(heads, dtype=torch.float32, device=device) * 0.5
    dt_bias = torch.randn(heads, dim, dtype=torch.float32, device=device) * 0.1
    initial_state = torch.randn(
        sequences,
        heads,
        dim,
        dim,
        dtype=torch.float32,
        device=device,
    )
    cu_seqlens = torch.tensor(cumulative, dtype=torch.int32, device=device)
    checkpoint_offsets = torch.full(
        (sequences,),
        case.checkpoint,
        dtype=torch.int32,
        device=device,
    )

    if tokens <= 4096:
        reference_out, reference_state = naive_reference(
            q,
            k,
            v,
            raw_g,
            raw_beta,
            a_log,
            dt_bias,
            initial_state,
            cumulative,
        )
    else:
        reference_out, reference_state = chunk_kda_with_fused_gate(
            q=q,
            k=k,
            v=v,
            raw_g=raw_g,
            raw_beta=raw_beta.contiguous(),
            A_log=a_log,
            g_bias=dt_bias.flatten(),
            initial_state=initial_state,
            output_final_state=True,
            lower_bound=-3.0,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
        torch.cuda.synchronize()
        if reference_state is None:
            raise RuntimeError("Triton did not return a final state")

    flash_out, flash_state, _ = _flashkda_prefill(
        q=q,
        k=k,
        v=v,
        g=raw_g,
        beta=raw_beta.contiguous(),
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=-3.0,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
    )
    torch.cuda.synchronize()
    assert_close(f"{case.name}/no_checkpoint/out", reference_out, flash_out)
    assert_close(f"{case.name}/no_checkpoint/state", reference_state, flash_state)

    checkpoint_out, checkpoint_final, checkpoint_state = _flashkda_prefill(
        q=q,
        k=k,
        v=v,
        g=raw_g,
        beta=raw_beta.contiguous(),
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=-3.0,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        checkpoint_offsets=checkpoint_offsets,
    )
    torch.cuda.synchronize()
    if checkpoint_state is None:
        raise RuntimeError("FlashKDA did not return checkpoint state")
    assert_close(f"{case.name}/checkpoint/out", reference_out, checkpoint_out)
    assert_close(f"{case.name}/checkpoint/final", reference_state, checkpoint_final)

    prefix_indices = torch.cat(
        [
            torch.arange(start, start + case.checkpoint, device=device)
            for start in cumulative[:-1]
        ]
    )
    prefix_cu = torch.arange(
        0,
        (sequences + 1) * case.checkpoint,
        case.checkpoint,
        dtype=torch.int32,
        device=device,
    )
    prefix_cumulative = cumulative_lengths(
        tuple(case.checkpoint for _ in range(sequences))
    )
    if tokens <= 4096:
        _, reference_checkpoint = naive_reference(
            q[:, prefix_indices],
            k[:, prefix_indices],
            v[:, prefix_indices],
            raw_g[:, prefix_indices],
            raw_beta[:, prefix_indices],
            a_log,
            dt_bias,
            initial_state,
            prefix_cumulative,
        )
    else:
        _, reference_checkpoint = chunk_kda_with_fused_gate(
            q=q[:, prefix_indices],
            k=k[:, prefix_indices],
            v=v[:, prefix_indices],
            raw_g=raw_g[:, prefix_indices],
            raw_beta=raw_beta[:, prefix_indices].contiguous(),
            A_log=a_log,
            g_bias=dt_bias.flatten(),
            initial_state=initial_state,
            output_final_state=True,
            lower_bound=-3.0,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=prefix_cu,
        )
        torch.cuda.synchronize()
        if reference_checkpoint is None:
            raise RuntimeError("Triton did not return checkpoint reference state")
    assert_close(
        f"{case.name}/checkpoint/state",
        reference_checkpoint,
        checkpoint_state,
    )

    workspace_bytes = torch.ops._flashkda_C.get_workspace_size(
        tokens, heads, sequences
    )
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
    reusable_out = torch.empty_like(checkpoint_out)
    reusable_final = torch.empty_like(checkpoint_final)
    reusable_checkpoint = torch.empty_like(checkpoint_state)
    for iteration in range(repeats):
        repeated_out, repeated_final, repeated_checkpoint = _flashkda_prefill(
            q=q,
            k=k,
            v=v,
            g=raw_g,
            beta=raw_beta.contiguous(),
            A_log=a_log,
            dt_bias=dt_bias,
            lower_bound=-3.0,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            checkpoint_offsets=checkpoint_offsets,
            out=reusable_out,
            final_state=reusable_final,
            checkpoint_state=reusable_checkpoint,
            workspace=workspace,
        )
        if repeated_checkpoint is None:
            raise RuntimeError("repeated call omitted checkpoint state")
        if not torch.equal(checkpoint_out, repeated_out):
            raise AssertionError(f"{case.name}: output changed on repeat {iteration}")
        if not torch.equal(checkpoint_final, repeated_final):
            raise AssertionError(f"{case.name}: final state changed on repeat {iteration}")
        if not torch.equal(checkpoint_state, repeated_checkpoint):
            raise AssertionError(
                f"{case.name}: checkpoint state changed on repeat {iteration}"
            )

    print(
        f"{case.name}: PASS tokens={tokens} sequences={sequences} repeats={repeats}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--include-long", action="store_true")
    args = parser.parse_args()

    cases = [
        Case("vsplit", (97,), 16),
        Case("default", (17, 18, 19, 20, 21, 22, 23, 24), 16),
    ]
    if args.include_long:
        cases.append(Case("production_chunk", (3080,), 3072))
    for case in cases:
        run_case(case, args.repeats)
    print("validation=PASS", flush=True)


if __name__ == "__main__":
    main()
