#!/usr/bin/env python3
"""Measure FlashKDA's first-call CUDA driver residency in a fresh process."""

from __future__ import annotations

import argparse
import hashlib
import time

import torch


MIB = 1024 * 1024


def snapshot(label: str) -> tuple[int, int, int]:
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(
        f"{label}: free={free / MIB:.1f} MiB total={total / MIB:.1f} MiB "
        f"allocated={allocated / MIB:.1f} MiB reserved={reserved / MIB:.1f} MiB",
        flush=True,
    )
    return free, allocated, reserved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=int, required=True)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--hold-seconds", type=int, default=0)
    parser.add_argument("--skip-finite-check", action="store_true")
    parser.add_argument("--benchmark-iterations", type=int, default=0)
    args = parser.parse_args()

    if args.tokens % args.sequences:
        raise ValueError("--tokens must be divisible by --sequences")

    import vllm._flashkda_C as extension

    extension_path = extension.__file__
    with open(extension_path, "rb") as extension_file:
        extension_sha = hashlib.sha256(extension_file.read()).hexdigest()
    props = torch.cuda.get_device_properties(0)
    use_vsplit = 2 * args.heads * args.sequences <= props.multi_processor_count
    print(f"extension={extension_path}")
    print(f"sha256={extension_sha}")
    print(
        f"gpu={props.name} sms={props.multi_processor_count} "
        f"sequences={args.sequences} use_vsplit={use_vsplit} "
        f"checkpoint={args.checkpoint}",
        flush=True,
    )
    snapshot("cuda_initialized")

    device = torch.device("cuda")
    batch, tokens, heads, dim = 1, args.tokens, args.heads, 128
    generator = torch.Generator(device=device).manual_seed(1234)
    q, k, v, g = [
        torch.randn(
            batch,
            tokens,
            heads,
            dim,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        for _ in range(4)
    ]
    beta = torch.randn(
        batch,
        tokens,
        heads,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    a_log = torch.randn(heads, dtype=torch.float32, device=device)
    dt_bias = torch.randn(heads, dim, dtype=torch.float32, device=device)
    initial_state = torch.randn(
        args.sequences,
        heads,
        dim,
        dim,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    final_state = torch.empty_like(initial_state)
    out = torch.empty_like(v)
    sequence_length = tokens // args.sequences
    cu_seqlens = torch.arange(
        0,
        tokens + 1,
        sequence_length,
        dtype=torch.int32,
        device=device,
    )
    checkpoint_state = torch.empty_like(initial_state) if args.checkpoint else None
    checkpoint_offsets = (
        ((cu_seqlens[1:] - cu_seqlens[:-1]) // 16) * 16
        if args.checkpoint
        else None
    )
    workspace_bytes = torch.ops._flashkda_C.get_workspace_size(
        tokens, heads, args.sequences
    )
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)

    before = snapshot("inputs_ready")
    for iteration in range(1, args.repeat + 1):
        torch.ops._flashkda_C.fwd(
            q,
            k,
            v,
            g,
            beta,
            dim**-0.5,
            out,
            workspace,
            a_log,
            dt_bias,
            -3.0,
            initial_state,
            final_state,
            cu_seqlens,
            checkpoint_state,
            checkpoint_offsets,
        )
        current = snapshot(f"after_call_{iteration}")
        print(
            f"call_{iteration}_delta: free={current[0] - before[0]:+.0f} B "
            f"allocated={current[1] - before[1]:+.0f} B "
            f"reserved={current[2] - before[2]:+.0f} B",
            flush=True,
        )

    if args.benchmark_iterations:
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start_event.record()
        for _ in range(args.benchmark_iterations):
            torch.ops._flashkda_C.fwd(
                q,
                k,
                v,
                g,
                beta,
                dim**-0.5,
                out,
                workspace,
                a_log,
                dt_bias,
                -3.0,
                initial_state,
                final_state,
                cu_seqlens,
                checkpoint_state,
                checkpoint_offsets,
            )
        end_event.record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - wall_start) * 1000
        gpu_ms = start_event.elapsed_time(end_event)
        print(
            f"benchmark: iterations={args.benchmark_iterations} "
            f"gpu_ms_per_call={gpu_ms / args.benchmark_iterations:.4f} "
            f"wall_ms_per_call={wall_ms / args.benchmark_iterations:.4f}",
            flush=True,
        )

    if not args.skip_finite_check:
        if not torch.isfinite(out.float()).all():
            raise RuntimeError("FlashKDA output contains non-finite values")
        if not torch.isfinite(final_state).all():
            raise RuntimeError("FlashKDA final state contains non-finite values")
        if checkpoint_state is not None and not torch.isfinite(checkpoint_state).all():
            raise RuntimeError("FlashKDA checkpoint state contains non-finite values")
    print("result=PASS", flush=True)
    if args.hold_seconds:
        print(f"holding_process={args.hold_seconds}s", flush=True)
        time.sleep(args.hold_seconds)


if __name__ == "__main__":
    main()
