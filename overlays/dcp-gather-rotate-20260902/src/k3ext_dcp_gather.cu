// SPDX-License-Identifier: Apache-2.0
// Rotated-destination variant of the direct DCP peer KV gather.
//
// The served kernel (vllm csrc/libtorch_stable/attention/dcp_utils/
// dcp_direct_kv_gather.cu, direct_dcp_kv_gather_peer_kernel) walks its
// work destination-major: every rank pushes its whole slice to destination 0,
// then to destination 1, and so on. Because all ranks run the same order at
// the same time, the eight sources funnel into one destination's inbound PCIe
// link at a time and the other seven links idle, which is why the gather
// moves ~3.7 GB/s per rank on a fabric whose links carry ~25 GB/s each.
//
// This variant assigns destinations per thread block, rotated by the source
// rank (block b serves destination (b + rank) mod world_size), so at any
// moment every inbound link receives from a different source. Payload,
// layout, completion counting and the epoch signalling are unchanged; only
// the order in which the same stores are issued differs, so the gathered
// bytes are identical.
//
// Registered as _C_k3ext::direct_dcp_kv_gather_rotated with the same
// signature as _C::direct_dcp_kv_gather; the multicast (NVLS) path is not
// provided here, callers keep using the base op when kv_mc_ptr != 0.

#include <torch/csrc/stable/library.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cstdio>

#include "dcp_direct_common.cuh"

namespace {

using vllm::direct_dcp::check_cuda_launch;
using vllm::direct_dcp::get_peer_ptr;
using vllm::direct_dcp::increment_epoch_kernel;
using vllm::direct_dcp::store_release_system;
using vllm::direct_dcp::wait_for_epoch;

constexpr int kThreads = 256;
constexpr int64_t kMaxPeerBlocks = 1024;

__global__ void direct_dcp_kv_gather_peer_rotated_kernel(
    const uint4* local_kv, const int32_t* dst_rows,
    const int64_t* peer_kv_ptrs, const int64_t* peer_signal_ptrs,
    const uint32_t* received_signal, const int64_t* epoch_ptr,
    uint32_t* completion, int64_t world_size, int64_t rank,
    int64_t num_tokens, int64_t items_per_row, int64_t kv_c_items_per_row,
    int64_t output_tokens, int64_t max_gathered_tokens,
    int64_t buffer_slot, int64_t slot_stride_items) {
  uint32_t epoch = static_cast<uint32_t>(epoch_ptr[0]);
  int64_t source_items = num_tokens * items_per_row;
  // gridDim.x is a multiple of world_size (host side): blocks with the same
  // residue share a destination and stride over that destination's items.
  int64_t destination_rank =
      (static_cast<int64_t>(blockIdx.x) + rank) % world_size;
  int64_t blocks_per_destination = static_cast<int64_t>(gridDim.x) / world_size;
  int64_t block_in_destination = static_cast<int64_t>(blockIdx.x) / world_size;
  int64_t item_stride = blocks_per_destination * blockDim.x;
  uint4* peer_kv = get_peer_ptr<uint4>(peer_kv_ptrs, destination_rank) +
                   buffer_slot * slot_stride_items;
  int64_t k_pe_items_per_row = items_per_row - kv_c_items_per_row;
  int64_t k_pe_plane_base = max_gathered_tokens * kv_c_items_per_row;
  for (int64_t item = block_in_destination * blockDim.x + threadIdx.x;
       item < source_items; item += item_stride) {
    int64_t src_row = item / items_per_row;
    int32_t dst_row = dst_rows[src_row];
    if (dst_row < 0) {
      continue;
    }
    if (dst_row >= output_tokens) {
      printf(
          "direct DCP peer final-layout destination out of bounds "
          "source=%lld dst=%d output_tokens=%lld\n",
          static_cast<long long>(rank), dst_row,
          static_cast<long long>(output_tokens));
      asm volatile("trap;");
    }
    int64_t row_item = item - src_row * items_per_row;
    int64_t dst_item;
    if (row_item < kv_c_items_per_row) {
      dst_item = static_cast<int64_t>(dst_row) * kv_c_items_per_row + row_item;
    } else {
      dst_item = k_pe_plane_base +
                 static_cast<int64_t>(dst_row) * k_pe_items_per_row +
                 row_item - kv_c_items_per_row;
    }
    peer_kv[dst_item] = local_kv[item];
  }

  // Completion protocol identical to the base kernel: every block publishes
  // its system-scope payload before contributing to the completion count;
  // the final block releases one signal per peer, then waits until every
  // source has published into this rank's local replica.
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x != 0) {
    return;
  }
  uint32_t completed = atomicAdd(completion + buffer_slot, 1u);
  if (completed + 1u != gridDim.x) {
    return;
  }
  atomicExch(completion + buffer_slot, 0u);

  int64_t signal_item = buffer_slot * world_size + rank;
  for (int64_t destination = 0; destination < world_size; ++destination) {
    uint32_t* peer_signal = get_peer_ptr<uint32_t>(peer_signal_ptrs, destination);
    store_release_system(peer_signal + signal_item, epoch);
  }
  for (int64_t source_rank = 0; source_rank < world_size; ++source_rank) {
    int64_t source_signal_item = buffer_slot * world_size + source_rank;
    if (!wait_for_epoch(received_signal + source_signal_item, epoch)) {
      printf("direct DCP peer final-layout timeout source=%lld epoch=%u\n",
             static_cast<long long>(source_rank), epoch);
      asm volatile("trap;");
    }
  }
}

void direct_dcp_kv_gather_rotated(
    const torch::stable::Tensor& local_kv, const torch::stable::Tensor& dst_rows,
    const torch::stable::Tensor& peer_kv_ptrs,
    const torch::stable::Tensor& peer_signal_ptrs,
    torch::stable::Tensor& received_kv, torch::stable::Tensor& received_signal,
    torch::stable::Tensor& completion, torch::stable::Tensor& epoch,
    int64_t output_tokens, int64_t plane_split_dim, int64_t buffer_slot,
    int64_t world_size, int64_t rank, int64_t max_gathered_tokens,
    int64_t kv_mc_ptr, int64_t signal_mc_ptr) {
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(kv_mc_ptr == 0 && signal_mc_ptr == 0,
                  "the rotated kv-gather serves the PCIe peer path only");
  STD_TORCH_CHECK(local_kv.is_cuda(), "local kv must be a CUDA tensor");
  ScalarType dtype = local_kv.scalar_type();
  STD_TORCH_CHECK(dtype == ScalarType::Half || dtype == ScalarType::BFloat16 ||
                      dtype == ScalarType::Float8_e4m3fn,
                  "direct DCP final-layout kv-gather only supports FP16, "
                  "BF16, and FP8");
  STD_TORCH_CHECK(local_kv.dim() == 2 && local_kv.is_contiguous(),
                  "local kv must be a contiguous [T,D] tensor");
  STD_TORCH_CHECK(dst_rows.is_cuda() && dst_rows.is_contiguous() &&
                      dst_rows.scalar_type() == ScalarType::Int &&
                      dst_rows.dim() == 1 &&
                      dst_rows.numel() == local_kv.size(0),
                  "final-layout destination rows must be CUDA int32 [T]");
  STD_TORCH_CHECK(
      peer_kv_ptrs.is_cuda() && peer_kv_ptrs.is_contiguous() &&
          peer_kv_ptrs.scalar_type() == ScalarType::Long &&
          peer_kv_ptrs.dim() == 1 && peer_kv_ptrs.numel() == world_size,
      "KV peer pointer table must be CUDA int64 [world_size]");
  STD_TORCH_CHECK(
      peer_signal_ptrs.is_cuda() && peer_signal_ptrs.is_contiguous() &&
          peer_signal_ptrs.scalar_type() == ScalarType::Long &&
          peer_signal_ptrs.dim() == 1 &&
          peer_signal_ptrs.numel() == world_size,
      "signal peer pointer table must be CUDA int64 [world_size]");
  STD_TORCH_CHECK(world_size > 1, "world_size must be greater than 1");
  STD_TORCH_CHECK(rank >= 0 && rank < world_size, "invalid rank");
  STD_TORCH_CHECK(buffer_slot == 0 || buffer_slot == 1,
                  "final-layout buffer slot must be 0 or 1");
  STD_TORCH_CHECK(output_tokens > 0 && output_tokens <= max_gathered_tokens,
                  "final-layout output exceeds symmetric buffer capacity");

  int64_t num_tokens = local_kv.size(0);
  int64_t token_dim = local_kv.size(1);
  int64_t element_size = local_kv.element_size();
  STD_TORCH_CHECK(num_tokens > 0 && token_dim > 0,
                  "local kv dimensions must be positive");
  STD_TORCH_CHECK(plane_split_dim > 0 && plane_split_dim < token_dim,
                  "final-layout plane split must be within the token row");
  STD_TORCH_CHECK(num_tokens * world_size <= max_gathered_tokens,
                  "padded gathered kv exceeds symmetric buffer capacity");
  STD_TORCH_CHECK(received_kv.is_cuda() && received_kv.scalar_type() == dtype &&
                      received_kv.is_contiguous() && received_kv.dim() == 3 &&
                      received_kv.size(0) == 2 &&
                      received_kv.size(1) == max_gathered_tokens &&
                      received_kv.size(2) == token_dim,
                  "received kv has the wrong symmetric buffer layout");
  STD_TORCH_CHECK(
      received_signal.is_cuda() && received_signal.is_contiguous() &&
          received_signal.scalar_type() == ScalarType::Int &&
          received_signal.dim() == 2 && received_signal.size(0) == 2 &&
          received_signal.size(1) == world_size,
      "received signal has the wrong symmetric buffer layout");
  STD_TORCH_CHECK(completion.is_cuda() && completion.is_contiguous() &&
                      completion.scalar_type() == ScalarType::Int &&
                      completion.numel() == 2,
                  "completion counter must be a two-element CUDA int32 tensor");
  STD_TORCH_CHECK(epoch.is_cuda() && epoch.is_contiguous() &&
                      epoch.scalar_type() == ScalarType::Long &&
                      epoch.numel() == 1,
                  "epoch must be a one-element CUDA int64 tensor");

  int64_t device_index = local_kv.get_device_index();
  STD_TORCH_CHECK(dst_rows.get_device_index() == device_index &&
                      peer_kv_ptrs.get_device_index() == device_index &&
                      peer_signal_ptrs.get_device_index() == device_index &&
                      received_kv.get_device_index() == device_index &&
                      received_signal.get_device_index() == device_index &&
                      completion.get_device_index() == device_index &&
                      epoch.get_device_index() == device_index,
                  "direct DCP final-layout tensors must share a CUDA device");

  int64_t row_bytes = token_dim * element_size;
  int64_t kv_c_row_bytes = plane_split_dim * element_size;
  int64_t k_pe_row_bytes = row_bytes - kv_c_row_bytes;
  int64_t slot_stride_bytes = max_gathered_tokens * row_bytes;
  bool vectorized =
      reinterpret_cast<uintptr_t>(local_kv.data_ptr()) % alignof(uint4) == 0 &&
      reinterpret_cast<uintptr_t>(received_kv.data_ptr()) % alignof(uint4) ==
          0 &&
      row_bytes % sizeof(uint4) == 0 && kv_c_row_bytes % sizeof(uint4) == 0 &&
      k_pe_row_bytes % sizeof(uint4) == 0 &&
      slot_stride_bytes % sizeof(uint4) == 0;
  STD_TORCH_CHECK(vectorized,
                  "direct DCP final-layout kv-gather requires 16-byte-aligned "
                  "planes, rows, and pointers");
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  cudaStream_t stream = get_current_cuda_stream();
  increment_epoch_kernel<<<1, 1, 0, stream>>>(
      epoch.mutable_data_ptr<int64_t>());
  check_cuda_launch("direct DCP final-layout kv-gather (rotated)");

  int64_t items_per_row = row_bytes / sizeof(uint4);
  int64_t kv_c_items_per_row = kv_c_row_bytes / sizeof(uint4);
  int64_t item_count = num_tokens * items_per_row;
  // Enough blocks for the per-destination stride loops, rounded up to a
  // multiple of world_size so every destination gets the same block count.
  int64_t blocks = (world_size * item_count + kThreads - 1) / kThreads;
  blocks = blocks < kMaxPeerBlocks ? blocks : kMaxPeerBlocks;
  blocks = ((blocks + world_size - 1) / world_size) * world_size;
  direct_dcp_kv_gather_peer_rotated_kernel<<<blocks, kThreads, 0, stream>>>(
      reinterpret_cast<const uint4*>(local_kv.data_ptr()),
      dst_rows.const_data_ptr<int32_t>(),
      peer_kv_ptrs.const_data_ptr<int64_t>(),
      peer_signal_ptrs.const_data_ptr<int64_t>(),
      reinterpret_cast<const uint32_t*>(
          received_signal.const_data_ptr<int32_t>()),
      epoch.const_data_ptr<int64_t>(),
      reinterpret_cast<uint32_t*>(completion.mutable_data_ptr<int32_t>()),
      world_size, rank, num_tokens, items_per_row, kv_c_items_per_row,
      output_tokens, max_gathered_tokens, buffer_slot,
      slot_stride_bytes / sizeof(uint4));
  check_cuda_launch("direct DCP final-layout kv-gather (rotated)");
}

}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_C_k3ext, direct_dcp_kv_gather_rotated_ops) {
  direct_dcp_kv_gather_rotated_ops.def(
      "direct_dcp_kv_gather_rotated("
      "Tensor local_kv, Tensor dst_rows, Tensor peer_kv_ptrs, "
      "Tensor peer_signal_ptrs, Tensor! received_kv, "
      "Tensor! received_signal, Tensor! completion, Tensor! epoch, "
      "int output_tokens, int plane_split_dim, int buffer_slot, "
      "int world_size, int rank, int max_gathered_tokens, "
      "int kv_mc_ptr, int signal_mc_ptr) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C_k3ext, CUDA, direct_dcp_kv_gather_rotated_ops) {
  direct_dcp_kv_gather_rotated_ops.impl("direct_dcp_kv_gather_rotated",
                                        TORCH_BOX(&direct_dcp_kv_gather_rotated));
}
