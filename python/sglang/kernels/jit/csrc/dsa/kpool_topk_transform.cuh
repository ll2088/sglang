// Radix top-k core adapted from https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_v32/topk_selector.py.
// This JIT variant adds pool expansion and page-table/ragged-offset transforms.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#ifndef USE_ROCM
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#else
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#endif

namespace sglang {
namespace {

#ifndef C10_LIKELY
#define C10_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#endif

#ifndef SGL_GROUP_TOPK
#define SGL_GROUP_TOPK 256
#endif

inline constexpr int kGroupTopK = SGL_GROUP_TOPK;
inline constexpr int kThreadsPerBlock = 1024;

// Keep 4K threshold-bin candidates per round; rescan larger bins without clipping.
inline constexpr std::size_t kSmem = 8 * 1024 * sizeof(uint32_t);  // 32KB (bytes)

template <typename InT>
struct FastTopKParams {
  const InT* __restrict__ input;           // [B, input_stride]
  const int32_t* __restrict__ row_starts;  // [B] or nullptr
  int32_t* __restrict__ indices;           // unused here (kept for layout parity)
  const int32_t* __restrict__ lengths;     // [B]
  int64_t input_stride;
};

// Elements per 16-byte vector: rows and slices are aligned to this.
template <typename InT>
inline constexpr int kVecWidth = 16 / sizeof(InT);

// Ordered 16-bit key of a BF16 score, matching convert_to_uint8's fp16 key.
__device__ __forceinline__ auto convert_bf16_to_uint8(__nv_bfloat16 x) -> uint8_t {
  const uint16_t bits = __bfloat16_as_ushort(x);
  const uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

// Coarse bins of the two BF16 scores packed in `w` (low half first), from one
// packed conversion and key flip.  The flip is the per-half form of
// convert_bf16_to_uint8: ~bits if negative, else bits | 0x8000.
__device__ __forceinline__ auto convert_bf16x2_to_uint8x2(uint32_t w) -> uint2 {
  const __nv_bfloat162 h = __floats2bfloat162_rn(__uint_as_float(w << 16), __uint_as_float(w & 0xFFFF0000u));
  const uint32_t bits = *reinterpret_cast<const uint32_t*>(&h);
  const uint32_t sign = bits & 0x80008000u;
  const uint32_t key = bits ^ ((sign - (sign >> 15)) | 0x80008000u);
  return make_uint2((key & 0xFFFFu) >> 8, (key >> 16) >> 8);
}

// Coarse bin of score e of a staged vector: FP32 bins each score, BF16 bins
// both halves of the packed word with one conversion.
template <typename InT>
__device__ __forceinline__ auto vec_coarse_bin(const InT* buf, uint32_t e) -> uint8_t {
  if constexpr (std::is_same_v<InT, __nv_bfloat16>) {
    const auto* words = reinterpret_cast<const uint32_t*>(buf);
    const uint2 bins = convert_bf16x2_to_uint8x2(words[e / 2]);
    return static_cast<uint8_t>(e % 2 ? bins.y : bins.x);
  } else {
    return convert_to_uint8(buf[e]);
  }
}

__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ auto make_topk_key(float score, int index) -> uint64_t {
  // Unique keys resolve an overfull bin even when all scores are equal.
  return (static_cast<uint64_t>(convert_to_uint32(score)) << 32) | ~static_cast<uint32_t>(index);
}

template <int K, typename InT>
__device__ void
fast_topk_cuda_tl_impl(const InT* __restrict__ input, int* __restrict__ index, int row_start, int length) {
  // We assume length > K here, or it will crash
  int topk = K;
  constexpr auto BLOCK_SIZE = 1024;
  constexpr auto RADIX = 256;
  constexpr auto SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));

  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];
  alignas(128) __shared__ uint64_t s_key_prefix;

  auto& s_histogram = s_histogram_buf[0];
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  const int tx = threadIdx.x;

  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();

  for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = vec_coarse_bin(input + row_start, static_cast<uint32_t>(idx));
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
      if (C10_LIKELY(tx < RADIX)) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += s_histogram_buf[k][tx + j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  topk -= s_histogram[threshold_bin + 1];
  const auto num_threshold_candidates = s_histogram[threshold_bin] - s_histogram[threshold_bin + 1];

  if (topk == 0) {
    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto bin = static_cast<int>(vec_coarse_bin(input + row_start, static_cast<uint32_t>(idx)));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      }
    }
    __syncthreads();
    return;
  } else if (num_threshold_candidates > int(SMEM_INPUT_SIZE)) {
    // Rescan the full row only on overflow. Preserve the unsorted output contract.
    if (tx == 0) s_key_prefix = 0;
    __syncthreads();

#pragma unroll 8
    for (int round = 0; round < 8; ++round) {
      if (tx < RADIX + 1) s_histogram[tx] = 0;
      __syncthreads();
      const auto prefix = s_key_prefix;
      const auto offset = 56 - round * 8;
      for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
        const auto raw_input = input[idx + row_start];
        if (vec_coarse_bin(input + row_start, static_cast<uint32_t>(idx)) != threshold_bin) continue;
        const auto key = make_topk_key(raw_input, idx);
        const bool prefix_matches = round == 0 || (key >> (64 - round * 8)) == prefix;
        if (prefix_matches) ::atomicAdd(&s_histogram[(key >> offset) & 0xFF], 1);
      }
      __syncthreads();
      run_cumsum();
      if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) s_threshold_bin_id = tx;
      __syncthreads();
      const auto key_bin = s_threshold_bin_id;
      topk -= s_histogram[key_bin + 1];
      if (tx == 0) s_key_prefix = (prefix << 8) | static_cast<uint64_t>(key_bin);
      __syncthreads();

      if (topk == 0) {
        const auto selected_prefix = s_key_prefix;
        const auto prefix_bits = (round + 1) * 8;
        if (tx == 0) s_counter = 0;
        __syncthreads();
        for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
          const auto raw_input = input[idx + row_start];
          const auto coarse_bin = vec_coarse_bin(input + row_start, static_cast<uint32_t>(idx));
          const auto key_prefix = make_topk_key(raw_input, idx) >> (64 - prefix_bits);
          if (coarse_bin > threshold_bin || (coarse_bin == threshold_bin && key_prefix > selected_prefix)) {
            const auto pos = ::atomicAdd(&s_counter, 1);
            index[pos] = idx;
          }
        }
        __syncthreads();
        return;
      }
    }

    const auto threshold_key = s_key_prefix;
    if (tx == 0) s_counter = 0;
    __syncthreads();
    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      if (make_topk_key(input[idx + row_start], idx) >= threshold_key) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      }
    }
    __syncthreads();
    return;
  } else {
    __syncthreads();
    if (tx < RADIX + 1) {
      s_histogram[tx] = 0;
    }
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto raw_input = input[idx + row_start];
      const auto bin = static_cast<int>(vec_coarse_bin(input + row_start, static_cast<uint32_t>(idx)));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      } else if (bin == threshold_bin) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
          s_input_idx[0][pos] = idx;
          const auto bin = convert_to_uint32(raw_input);
          const auto sub_bin = (bin >> 24) & 0xFF;
          ::atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    __shared__ int s_last_remain;
    const auto r_idx = round % 2;

    const auto _raw_num_input = s_num_input[r_idx];
    const auto num_input = (_raw_num_input < int(SMEM_INPUT_SIZE)) ? _raw_num_input : int(SMEM_INPUT_SIZE);

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
      s_threshold_bin_id = tx;
      s_num_input[r_idx ^ 1] = 0;
      s_last_remain = topk - s_histogram[tx + 1];
    }
    __syncthreads();

    const auto threshold_bin = s_threshold_bin_id;
    topk -= s_histogram[threshold_bin + 1];

    if (topk == 0) {
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(input[idx + row_start]) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        }
      }
      __syncthreads();
      break;
    } else {
      __syncthreads();
      if (tx < RADIX + 1) {
        s_histogram[tx] = 0;
      }
      __syncthreads();
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto raw_input = input[idx + row_start];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == 3) {
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0) {
              index[K - pos] = idx;
            }
          } else {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
              s_input_idx[r_idx ^ 1][pos] = idx;
              const auto bin = convert_to_uint32(raw_input);
              const auto sub_bin = (bin >> (offset - 8)) & 0xFF;
              ::atomicAdd(&s_histogram[sub_bin], 1);
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

__device__ __forceinline__ int32_t transform_kpool_token(
    int32_t raw_token,
    const int32_t* __restrict__ page_table_entry,
    const int32_t* __restrict__ topk_indices_offset,
    int32_t offset) {
  if (page_table_entry != nullptr) {
    return page_table_entry[raw_token];
  }
  if (topk_indices_offset != nullptr) {
    return raw_token + offset;
  }
  return raw_token;
}

template <int K, typename InT>
__global__ __launch_bounds__(kThreadsPerBlock) void kpool_topk_transform_kernel(
    const __grid_constant__ FastTopKParams<InT> params,
    int32_t* __restrict__ dst_token_indices,
    const int64_t dst_stride,
    const int32_t pool_size,
    const int32_t token_topk,
    const int32_t out_cols,
    const int32_t* __restrict__ page_table,
    const int64_t page_table_stride,
    const int32_t* __restrict__ page_table_row_index,
    const int32_t* __restrict__ topk_indices_offset,
    const int32_t* __restrict__ seq_lens) {
  const auto& [input, row_starts, _, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto length = lengths[bid];
  const auto score = input + bid * input_stride;
  const auto dst = dst_token_indices + bid * dst_stride;
  const auto page_table_row = page_table_row_index == nullptr ? bid : static_cast<uint64_t>(page_table_row_index[bid]);
  const auto page_table_entry = page_table == nullptr ? nullptr : page_table + page_table_row * page_table_stride;
  const auto offset = topk_indices_offset == nullptr ? 0 : topk_indices_offset[bid];
  const bool append_tail = seq_lens != nullptr;
  const auto full_pool_token_len = length * pool_size;
  const auto history_len = full_pool_token_len < token_topk ? full_pool_token_len : token_topk;
  const auto tail_count = append_tail ? seq_lens[bid] % pool_size : 0;

  if (length <= K) {
    for (int col = tid; col < out_cols; col += kThreadsPerBlock) {
      if (col < history_len) {
        const auto group_rank = col / pool_size;
        const auto slot = col % pool_size;
        const auto raw_token = group_rank * pool_size + slot;
        dst[col] = transform_kpool_token(raw_token, page_table_entry, topk_indices_offset, offset);
      } else if (append_tail && col < history_len + tail_count) {
        const auto raw_token = length * pool_size + (col - history_len);
        dst[col] = transform_kpool_token(raw_token, page_table_entry, topk_indices_offset, offset);
      } else {
        dst[col] = -1;
      }
    }
    return;
  }

  __shared__ int s_indices[K];
  fast_topk_cuda_tl_impl<K, InT>(score, s_indices, row_start, length);
  for (int col = tid; col < out_cols; col += kThreadsPerBlock) {
    if (col < history_len) {
      const auto group_rank = col / pool_size;
      const auto group_id = s_indices[group_rank];
      const auto slot = col % pool_size;
      const auto raw_token = group_id * pool_size + slot;
      dst[col] = transform_kpool_token(raw_token, page_table_entry, topk_indices_offset, offset);
    } else if (append_tail && col < history_len + tail_count) {
      const auto raw_token = length * pool_size + (col - history_len);
      dst[col] = transform_kpool_token(raw_token, page_table_entry, topk_indices_offset, offset);
    } else {
      dst[col] = -1;
    }
  }
}

template <typename T>
const T* optional_data_ptr(const tvm::ffi::Optional<tvm::ffi::TensorView>& opt) {
  if (!opt.has_value()) {
    return nullptr;
  }
  return static_cast<const T*>(opt.value().data_ptr());
}

struct KpoolTopKTransformKernel {
  static constexpr auto kernel_fp32 = kpool_topk_transform_kernel<kGroupTopK, float>;
  static constexpr auto kernel_bf16 = kpool_topk_transform_kernel<kGroupTopK, __nv_bfloat16>;

  // BF16 logits are read natively when the row stride keeps every row 16-byte
  // aligned; otherwise the caller upcasts and takes the FP32 kernel.
  static auto takes_bf16_logits(const tvm::ffi::TensorView& score) -> bool {
    return score.dtype().code == kDLBfloat && score.dtype().bits == 16 && score.dtype().lanes == 1 &&
           score.stride(0) % kVecWidth<__nv_bfloat16> == 0;
  }

  static void transform(
      const tvm::ffi::TensorView score,
      const tvm::ffi::TensorView lengths,
      const tvm::ffi::TensorView dst_token_indices,
      const int64_t pool_size,
      const tvm::ffi::Optional<tvm::ffi::TensorView> page_table_opt,
      const tvm::ffi::Optional<tvm::ffi::TensorView> topk_indices_offset_opt,
      const tvm::ffi::Optional<tvm::ffi::TensorView> row_starts_opt,
      const tvm::ffi::Optional<tvm::ffi::TensorView> seq_lens_opt,
      const tvm::ffi::Optional<tvm::ffi::TensorView> page_table_row_index_opt) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto S = SymbolicSize{"score_stride"};
    auto out_cols_sym = SymbolicSize{"out_cols"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();

    auto score_dtype = SymbolicDType{};
    score_dtype.set_options<kDLFloat, kDLBfloat>();
    TensorMatcher({B, -1}).with_strides({S, 1}).with_dtype(score_dtype).with_device(device).verify(score);
    RuntimeCheck(
        (score.dtype().code == kDLFloat && score.dtype().bits == 32) ||
            (score.dtype().code == kDLBfloat && score.dtype().bits == 16),
        "score must be float32 or bfloat16");
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(lengths);
    TensorMatcher({B, out_cols_sym}).with_dtype<int32_t>().with_device(device).verify(dst_token_indices);

    RuntimeCheck(pool_size > 1, "pool_size must be > 1, got ", pool_size);
    RuntimeCheck(
        !(page_table_opt.has_value() && topk_indices_offset_opt.has_value()),
        "page_table and topk_indices_offset are mutually exclusive");
    RuntimeCheck(
        !page_table_row_index_opt.has_value() || page_table_opt.has_value(),
        "page_table_row_index requires page_table");

    const auto out_cols = static_cast<int32_t>(out_cols_sym.unwrap());
    const auto tail_cols = seq_lens_opt.has_value() ? static_cast<int32_t>(pool_size) - 1 : 0;
    RuntimeCheck(out_cols > tail_cols, "dst_token_indices columns ", out_cols, " must exceed tail ", tail_cols);
    const auto token_topk = out_cols - tail_cols;
    RuntimeCheck(token_topk % static_cast<int32_t>(pool_size) == 0, "token_topk must be a multiple of pool_size");
    RuntimeCheck(
        token_topk / static_cast<int32_t>(pool_size) == kGroupTopK,
        "this module is built for group_topk=",
        kGroupTopK,
        " but got ",
        token_topk / static_cast<int32_t>(pool_size));

    const auto batch_size = static_cast<uint32_t>(B.unwrap());

    int64_t page_table_stride = 0;
    const int32_t* page_table_ptr = nullptr;
    if (page_table_opt.has_value()) {
      auto P = SymbolicSize{"page_table_stride"};
      if (page_table_row_index_opt.has_value()) {
        auto page_table_rows = SymbolicSize{"page_table_rows"};
        TensorMatcher({page_table_rows, -1})
            .with_strides({P, 1})
            .with_dtype<int32_t>()
            .with_device(device)
            .verify(page_table_opt.value());
      } else {
        TensorMatcher({B, -1}).with_strides({P, 1}).with_dtype<int32_t>().with_device(device).verify(
            page_table_opt.value());
      }
      page_table_ptr = static_cast<const int32_t*>(page_table_opt.value().data_ptr());
      page_table_stride = static_cast<int64_t>(P.unwrap());
    }

    if (topk_indices_offset_opt.has_value()) {
      TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(topk_indices_offset_opt.value());
    }
    if (row_starts_opt.has_value()) {
      TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(row_starts_opt.value());
    }
    if (seq_lens_opt.has_value()) {
      TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(seq_lens_opt.value());
    }
    if (page_table_row_index_opt.has_value()) {
      TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(page_table_row_index_opt.value());
    }

    const auto input_stride = static_cast<int64_t>(S.unwrap());
    const auto dst_stride = static_cast<int64_t>(dst_token_indices.strides()[0]);
    const auto* page_table_row_index_ptr = optional_data_ptr<int32_t>(page_table_row_index_opt);
    const auto* topk_indices_offset_ptr = optional_data_ptr<int32_t>(topk_indices_offset_opt);
    const auto* seq_lens_ptr = optional_data_ptr<int32_t>(seq_lens_opt);
    const auto* row_starts_ptr = optional_data_ptr<int32_t>(row_starts_opt);

    if (takes_bf16_logits(score)) {
      const auto params = FastTopKParams<__nv_bfloat16>{
          .input = static_cast<const __nv_bfloat16*>(score.data_ptr()),
          .row_starts = row_starts_ptr,
          .indices = nullptr,
          .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
          .input_stride = input_stride,
      };
      runtime::set_max_dynamic_smem_per_device<kernel_bf16>(device.unwrap().device_id, kSmem);
      LaunchKernel(batch_size, kThreadsPerBlock, device.unwrap(), kSmem)(
          kernel_bf16,
          params,
          static_cast<int32_t*>(dst_token_indices.data_ptr()),
          dst_stride,
          static_cast<int32_t>(pool_size),
          token_topk,
          out_cols,
          page_table_ptr,
          page_table_stride,
          page_table_row_index_ptr,
          topk_indices_offset_ptr,
          seq_lens_ptr);
      return;
    }

    const auto params = FastTopKParams<float>{
        .input = static_cast<const float*>(score.data_ptr()),
        .row_starts = row_starts_ptr,
        .indices = nullptr,
        .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
        .input_stride = input_stride,
    };
    runtime::set_max_dynamic_smem_per_device<kernel_fp32>(device.unwrap().device_id, kSmem);
    LaunchKernel(batch_size, kThreadsPerBlock, device.unwrap(), kSmem)(
        kernel_fp32,
        params,
        static_cast<int32_t*>(dst_token_indices.data_ptr()),
        dst_stride,
        static_cast<int32_t>(pool_size),
        token_topk,
        out_cols,
        page_table_ptr,
        page_table_stride,
        page_table_row_index_ptr,
        topk_indices_offset_ptr,
        seq_lens_ptr);
  }
};

}  // namespace

}  // namespace sglang
