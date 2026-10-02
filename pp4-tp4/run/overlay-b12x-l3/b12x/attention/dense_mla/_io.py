"""Dense page-table traversal and asynchronous K3 record staging."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics import (
    cp_async_bulk_g2s_mbar,
    get_ptr_as_int64,
    ld_global_nc_u32,
    ld_global_nc_v2_u32,
    ld_global_nc_v4_u32,
    shared_ptr_to_u32,
    st_shared_u32,
    st_shared_v4_u32,
)

from ._layout import CANDIDATES_PER_CHUNK


@cute.jit
def issue_dense_page_gather(
    cache_bytes: cute.Tensor,
    page_table: cute.Tensor,
    kv_dst_addr: Int32,
    full_mbar_ptr,
    request: Int32,
    token_begin: Int32,
    token_end: Int32,
    io_lane: Int32,
    page_stride_bytes: Int64,
    page_table_stride: Int64,
    *,
    page_size: cutlass.Constexpr,
    record_bytes: cutlass.Constexpr,
    record_stride_bytes: cutlass.Constexpr,
):
    """Stage one 64-record dense window with an mbarrier transaction.

    Every pool-scaled product is promoted before multiplication. Invalid tail
    rows copy physical record zero and are masked by the consumer; issuing a
    fixed byte count keeps the full-barrier protocol independent of live
    sequence length and therefore graph-replay stable.
    """
    full_mbar_addr = shared_ptr_to_u32(full_mbar_ptr)
    if io_lane == Int32(0):
        cute.arch.mbarrier_arrive_and_expect_tx(
            full_mbar_ptr,
            Int32(CANDIDATES_PER_CHUNK * record_bytes),
        )

    entry = io_lane
    for _ in cutlass.range_constexpr(CANDIDATES_PER_CHUNK // 32):
        logical_token = token_begin + entry
        safe_token = logical_token
        if logical_token >= token_end:
            safe_token = Int32(0)
        logical_page = safe_token // Int32(page_size)
        in_page = safe_token - logical_page * Int32(page_size)
        physical_page = Int32(
            page_table[request.to(Int64) * page_table_stride + logical_page.to(Int64)]
        )

        # Load-bearing Int64 conversions. Neither multiplication is allowed to
        # occur in Int32, even when benchmark page ids happen to be small.
        source_offset = physical_page.to(Int64) * page_stride_bytes + in_page.to(
            Int64
        ) * Int64(record_bytes)
        cp_async_bulk_g2s_mbar(
            kv_dst_addr + entry * Int32(record_stride_bytes),
            get_ptr_as_int64(cache_bytes, source_offset),
            Int32(record_bytes),
            full_mbar_addr,
        )
        entry += Int32(32)


__all__ = ["issue_dense_page_gather", "issue_dense_page_gather_nvfp4"]

# nvfp4_ds_mla 432-byte record ABI (stock writer, bf16 RoPE):
#   [0,   256)  512 x E2M1 NoPE latent, packed 2/byte (low nibble = even dim)
#   [256, 288)  32  x E4M3 group scales, group g covers dims [16g, 16g+16)
#   [288, 304)  16  B pad
#   [304, 432)  64  x BF16 RoPE, unquantized
_NVFP4_RECORD_BYTES = 432
_NVFP4_SCALE_OFFSET = 256
_NVFP4_ROPE_OFFSET = 304
_BF16_ROPE_BYTE_OFFSET = 1024  # 512 BF16 latent values precede the rope tail


@dsl_user_op
def nvfp4_pair_to_bf16x2_sm120(
    packed: Uint32, scale: Uint32, *, loc=None, ip=None
) -> Uint32:
    """Decode one packed E2M1 byte and one E4M3 scale byte to BF16x2.

    Native SM120a conversions; every finite E2M1 * E4M3 product is exactly
    representable in BF16. Low nibble -> low BF16 half (even dim), matching
    the writer's cvt.rn.satfinite.e2m1x2.f32 packing order. Ported verbatim
    from the b12x sparse-MLA NVFP4 reader (jovian-judgement r38 lineage).
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(packed).ir_value(loc=loc, ip=ip),
             Uint32(scale).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b8 q;
                .reg .b16 sf;
                .reg .b32 values, factors, sf_pair;
                cvt.u8.u32 q, $1;
                shl.b32 sf_pair, $2, 8;
                or.b32 sf_pair, $2, sf_pair;
                cvt.u16.u32 sf, sf_pair;
                cvt.rn.bf16x2.e2m1x2 values, q;
                cvt.rn.bf16x2.e4m3x2 factors, sf;
                mul.bf16x2 $0, values, factors;
            }
            """,
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def issue_dense_page_gather_nvfp4(
    cache_bytes: cute.Tensor,
    page_table: cute.Tensor,
    kv_dst_addr: Int32,
    full_mbar_ptr,
    request: Int32,
    token_begin: Int32,
    token_end: Int32,
    io_lane: Int32,
    page_stride_bytes: Int64,
    page_table_stride: Int64,
    *,
    page_size: cutlass.Constexpr,
    record_stride_bytes: cutlass.Constexpr,
):
    """Stage one 64-record NVFP4 window, dequantizing to BF16 in SMEM.

    Drop-in producer for ``issue_dense_page_gather`` when the cache holds the
    432-byte nvfp4_ds_mla record. Each lane owns two records: packed E2M1
    latent + E4M3 group scales are read straight from global memory, converted
    with the native SM120a pair instruction (E2M1 * E4M3 is exact in BF16),
    and stored into the same BF16 staging layout the BF16 consumer reads. The
    BF16 RoPE tail is copied verbatim. The writer quantizes with outer scale
    1.0, so no latent scale is applied here.

    Barrier protocol: the TMA path expects a transaction byte count; this path
    performs ordinary SMEM stores, so the full barrier is initialized with an
    arrival count of 32 and every lane arrives after its own stores (each
    arrive releases that lane's prior stores).

    Invalid tail rows dequant physical record zero (a real, masked record),
    mirroring the TMA producer's safe-token policy.
    """
    entry = io_lane
    for _ in cutlass.range_constexpr(CANDIDATES_PER_CHUNK // 32):
        logical_token = token_begin + entry
        safe_token = logical_token
        if logical_token >= token_end:
            safe_token = Int32(0)
        logical_page = safe_token // Int32(page_size)
        in_page = safe_token - logical_page * Int32(page_size)
        physical_page = Int32(
            page_table[request.to(Int64) * page_table_stride + logical_page.to(Int64)]
        )

        # Load-bearing Int64 conversions; see issue_dense_page_gather.
        source_offset = physical_page.to(Int64) * page_stride_bytes + in_page.to(
            Int64
        ) * Int64(_NVFP4_RECORD_BYTES)
        src = get_ptr_as_int64(cache_bytes, source_offset)
        dst = kv_dst_addr + entry * Int32(record_stride_bytes)

        # All 32 group scales: eight u32 registers, byte g in register g//4.
        scale_words = []
        for sw in cutlass.range_constexpr(8):
            scale_words.append(
                ld_global_nc_u32(src + Int64(_NVFP4_SCALE_OFFSET + sw * 4))
            )

        for group in cutlass.range_constexpr(32):
            lo, hi = ld_global_nc_v2_u32(src + Int64(group * 8))
            scale_word = scale_words[group // 4]
            scale_byte = (scale_word >> Uint32(8 * (group % 4))) & Uint32(0xFF)
            for byte_index in cutlass.range_constexpr(4):
                packed_lo = (lo >> Uint32(8 * byte_index)) & Uint32(0xFF)
                pair_lo = nvfp4_pair_to_bf16x2_sm120(packed_lo, scale_byte)
                st_shared_u32(
                    dst + Int32(group * 32 + byte_index * 4),
                    pair_lo,
                )
                packed_hi = (hi >> Uint32(8 * byte_index)) & Uint32(0xFF)
                pair_hi = nvfp4_pair_to_bf16x2_sm120(packed_hi, scale_byte)
                st_shared_u32(
                    dst + Int32(group * 32 + 16 + byte_index * 4),
                    pair_hi,
                )

        # BF16 RoPE tail: 128 B, verbatim.
        for rope_block in cutlass.range_constexpr(8):
            v0, v1, v2, v3 = ld_global_nc_v4_u32(
                src + Int64(_NVFP4_ROPE_OFFSET + rope_block * 16)
            )
            st_shared_v4_u32(
                dst + Int32(_BF16_ROPE_BYTE_OFFSET + rope_block * 16),
                v0,
                v1,
                v2,
                v3,
            )

        entry += Int32(32)

    # Release every lane's stores into the phase the consumer waits on.
    cute.arch.mbarrier_arrive(full_mbar_ptr)
