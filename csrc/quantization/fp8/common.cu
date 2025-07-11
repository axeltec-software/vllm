#include "common.cuh"
#include "dispatch_utils.h"

#include <c10/cuda/CUDAGuard.h>

#ifndef USE_ROCM
  #include <cub/cub.cuh>
#else
  #include <hipcub/hipcub.hpp>
#endif

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Float8_e4m3fn.h>

#include <cmath>
#include <flashinfer/vec_dtypes.cuh>

namespace vllm {

template <typename scalar_t, typename fp8_type>
__global__ void scaled_fp8_quant_kernel(fp8_type* __restrict__ out,
                                        const scalar_t* __restrict__ input,
                                        const float* __restrict__ scale,
                                        int64_t num_elems) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  // Invert the scale so that we can use multiplications to avoid expensive
  // division.
  const float inverted_scale = 1.0f / (*scale);
  scaled_fp8_conversion_vec<scalar_t, true>(
      out, input, inverted_scale, num_elems, tid, blockDim.x * gridDim.x);
}

template <typename scalar_t, typename fp8_type>
__global__ void dynamic_per_token_scaled_fp8_quant_kernel(
    fp8_type* __restrict__ out, float* __restrict__ scale,
    scalar_t const* __restrict__ input, float const* __restrict__ scale_ub,
    const int hidden_size) {
  int const tid = threadIdx.x;
  int const token_idx = blockIdx.x;

  // Use int64 to avoid overflowing an int32 when calculating this offset
  int64_t offset = static_cast<int64_t>(token_idx) * hidden_size;
  scalar_t const* __restrict__ token_input = &input[offset];
  fp8_type* __restrict__ token_output = &out[offset];

  // For vectorization, token_input and token_output pointers need to be
  // aligned at 32-byte and 16-byte addresses respectively.
  bool const can_vectorize = hidden_size % 16 == 0;

  float absmax_val = 0.0f;
  if (can_vectorize) {
    absmax_val = thread_max_vec(token_input, hidden_size, tid, blockDim.x);
  } else {
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      float const x = static_cast<float>(token_input[i]);
      absmax_val = fmaxf(absmax_val, fabsf(x));
    }
  }

  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ typename BlockReduce::TempStorage reduceStorage;
  float const block_absmax_val_maybe =
      BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim.x);
  __shared__ float token_scale;
  if (tid == 0) {
    if (scale_ub) {
      token_scale = fminf(block_absmax_val_maybe, *scale_ub);
    } else {
      token_scale = block_absmax_val_maybe;
    }
    // token scale computation
    token_scale = fmaxf(token_scale / quant_type_max_v<fp8_type>,
                        min_scaling_factor<fp8_type>::val());
    scale[token_idx] = token_scale;
  }
  __syncthreads();

  // Note that we don't use inverted scales so we can match FBGemm impl.
  if (can_vectorize) {
    scaled_fp8_conversion_vec<scalar_t, false>(
        token_output, token_input, token_scale, hidden_size, tid, blockDim.x);
  } else {
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      token_output[i] = scaled_fp8_conversion<false, fp8_type>(
          static_cast<float>(token_input[i]), token_scale);
    }
  }
}

}  // namespace vllm

void static_scaled_fp8_quant(torch::Tensor& out,          // [..., d]
                             torch::Tensor const& input,  // [..., d]
                             torch::Tensor const& scale)  // [1]
{
  int const block_size = 256;
  int const num_tokens = input.numel() / input.size(-1);
  int const num_elems = input.numel();
  dim3 const grid(num_tokens);
  dim3 const block(block_size);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "scaled_fp8_quant_kernel_scalar_type", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "scaled_fp8_quant_kernel_fp8_type", [&] {
              vllm::scaled_fp8_quant_kernel<scalar_t, fp8_t>
                  <<<grid, block, 0, stream>>>(
                      out.data_ptr<fp8_t>(), input.data_ptr<scalar_t>(),
                      scale.data_ptr<float>(), num_elems);
            });
      });
}

void dynamic_scaled_fp8_quant(torch::Tensor& out,          // [..., d]
                              torch::Tensor const& input,  // [..., d]
                              torch::Tensor& scale)        // [1]
{
  int const block_size = 256;
  int const num_tokens = input.numel() / input.size(-1);
  int const num_elems = input.numel();
  dim3 const grid(num_tokens);
  dim3 const block(block_size);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "scaled_fp8_quant_kernel_scalar_type", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "scaled_fp8_quant_kernel_fp8_type", [&] {
              vllm::segmented_max_reduction<scalar_t, fp8_t>
                  <<<grid, block, 0, stream>>>(scale.data_ptr<float>(),
                                               input.data_ptr<scalar_t>(),
                                               num_elems);
              vllm::scaled_fp8_quant_kernel<scalar_t, fp8_t>
                  <<<grid, block, 0, stream>>>(
                      out.data_ptr<fp8_t>(), input.data_ptr<scalar_t>(),
                      scale.data_ptr<float>(), num_elems);
            });
      });
}

void dynamic_per_token_scaled_fp8_quant(
    torch::Tensor& out,          // [..., d]
    torch::Tensor const& input,  // [..., d]
    torch::Tensor& scales, std::optional<at::Tensor> const& scale_ub) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());

  int const hidden_size = input.size(-1);
  int const num_tokens = input.numel() / hidden_size;
  int const block_size = 256;
  dim3 const grid(num_tokens);
  dim3 const block(std::min(hidden_size, block_size));

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(),
      "dynamic_per_token_scaled_fp8_quant_kernel_scalar_type", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(),
            "dynamic_per_token_scaled_fp8_quant_kernel_fp8_type", [&] {
              vllm::dynamic_per_token_scaled_fp8_quant_kernel<scalar_t, fp8_t>
                  <<<grid, block, 0, stream>>>(
                      out.data_ptr<fp8_t>(), scales.data_ptr<float>(),
                      input.data_ptr<scalar_t>(),
                      scale_ub.has_value() ? scale_ub->data_ptr<float>()
                                           : nullptr,
                      hidden_size);
            });
      });
}

__device__ __forceinline__ float GroupReduceMax(float val, const int tid) {
  unsigned mask = 0xffff;

  val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
  return val;
}

template <
    typename T, typename DST_DTYPE, bool IS_COLUMN_MAJOR = false,
    bool SCALE_UE8M0 = false,
    typename scale_packed_t = std::conditional_t<SCALE_UE8M0, uint32_t, float>>
__global__ void per_token_group_quant_8bit_kernel(
    const T* __restrict__ input, void* __restrict__ output_q,
    scale_packed_t* __restrict__ output_s, const int group_size,
    const int num_groups, const int groups_per_block, const float eps,
    const float min_8bit, const float max_8bit, const int scale_num_rows = 0,
    const int scale_stride = 0) {
  const int threads_per_group = 16;
  const int local_group_id = threadIdx.x / threads_per_group;
  const int lane_id = threadIdx.x % threads_per_group;

  const int block_group_id = blockIdx.x * groups_per_block;
  const int global_group_id = block_group_id + local_group_id;
  const int block_group_offset = global_group_id * group_size;

  float local_absmax = eps;

  using scale_element_t = std::conditional_t<SCALE_UE8M0, uint8_t, float>;
  static_assert(sizeof(scale_packed_t) % sizeof(scale_element_t) == 0);

  const T* group_input = input + block_group_offset;
  DST_DTYPE* group_output =
      static_cast<DST_DTYPE*>(output_q) + block_group_offset;
  scale_element_t* scale_output;

  if constexpr (IS_COLUMN_MAJOR) {
    const int num_elems_per_pack =
        static_cast<int>(sizeof(scale_packed_t) / sizeof(scale_element_t));
    const int scale_num_rows_element = scale_num_rows * num_elems_per_pack;
    const int row_idx = global_group_id / scale_num_rows_element;
    const int col_idx_raw = global_group_id % scale_num_rows_element;
    const int col_idx = col_idx_raw / num_elems_per_pack;
    const int pack_idx = col_idx_raw % num_elems_per_pack;
    scale_output = reinterpret_cast<scale_element_t*>(output_s) +
                   (col_idx * scale_stride * num_elems_per_pack +
                    row_idx * num_elems_per_pack + pack_idx);
  } else {
    static_assert(!SCALE_UE8M0);
    scale_output = output_s + global_group_id;
  }

  constexpr uint32_t vec_size = 16 / sizeof(T);
  using vec_t = flashinfer::vec_t<T, vec_size>;

  const int32_t num_vec_elems = group_size / vec_size;

  for (int32_t i = lane_id; i < num_vec_elems; i += 16) {
    vec_t input_vec;
    input_vec.cast_load(group_input + i * vec_size);

#pragma unroll
    for (uint32_t j = 0; j < vec_size; ++j) {
      float val = static_cast<float>(input_vec[j]);
      float abs_val = fabsf(val);
      local_absmax = fmaxf(local_absmax, abs_val);
    }
  }

  local_absmax = GroupReduceMax(local_absmax, lane_id);

  float y_s = local_absmax / max_8bit;
  if constexpr (SCALE_UE8M0) {
    y_s = exp2f(ceilf(log2f(fmaxf(fabsf(y_s), 1e-10f))));
  }

  // TODO can optimize
  scale_element_t y_s_quant;
  if constexpr (SCALE_UE8M0) {
    y_s_quant = (uint8_t)(((int)log2f(y_s)) + 127);
  } else {
    y_s_quant = y_s;
  }

  if (lane_id == 0) {
    *scale_output = y_s_quant;
  }

  for (int32_t i = lane_id; i < num_vec_elems; i += 16) {
    vec_t input_vec;
    input_vec.cast_load(group_input + i * vec_size);

#pragma unroll
    for (uint32_t j = 0; j < vec_size; ++j) {
      float val = static_cast<float>(input_vec[j]);
      float q_val = fminf(fmaxf(val / y_s, min_8bit), max_8bit);
      group_output[i * vec_size + j] = DST_DTYPE(q_val);
    }
  }
}

void sgl_per_token_group_quant_8bit(torch::Tensor input, torch::Tensor output_q,
                                    torch::Tensor output_s, int64_t group_size,
                                    double eps, double min_8bit,
                                    double max_8bit, bool scale_ue8m0 = false) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);

  const int num_groups = input.numel() / group_size;

  CHECK_EQ(input.numel() % group_size, 0);
  CHECK_EQ(output_s.dim(), 2);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  constexpr int THREADS_PER_GROUP = 16;

  int groups_per_block = 1;

  if (num_groups % 16 == 0) {
    groups_per_block = 16;
  } else if (num_groups % 8 == 0) {
    groups_per_block = 8;
  } else if (num_groups % 4 == 0) {
    groups_per_block = 4;
  } else if (num_groups % 2 == 0) {
    groups_per_block = 2;
  }

  auto dst_type = output_q.scalar_type();
  const int num_blocks = num_groups / groups_per_block;
  const int num_threads = groups_per_block * THREADS_PER_GROUP;

  const bool is_column_major = output_s.stride(0) < output_s.stride(1);
  const int scale_num_rows = output_s.size(1);
  const int scale_stride = output_s.stride(1);

#define LAUNCH_KERNEL(T, DST_DTYPE)                                        \
  do {                                                                     \
    dim3 grid(num_blocks);                                                 \
    dim3 block(num_threads);                                               \
    if (is_column_major) {                                                 \
      if (scale_ue8m0) {                                                   \
        per_token_group_quant_8bit_kernel<T, DST_DTYPE, true, true>        \
            <<<grid, block, 0, stream>>>(                                  \
                static_cast<T*>(input.data_ptr()), output_q.data_ptr(),    \
                static_cast<uint32_t*>(output_s.data_ptr()), group_size,   \
                num_groups, groups_per_block, (float)eps, (float)min_8bit, \
                (float)max_8bit, scale_num_rows, scale_stride);            \
      } else {                                                             \
        per_token_group_quant_8bit_kernel<T, DST_DTYPE, true, false>       \
            <<<grid, block, 0, stream>>>(                                  \
                static_cast<T*>(input.data_ptr()), output_q.data_ptr(),    \
                static_cast<float*>(output_s.data_ptr()), group_size,      \
                num_groups, groups_per_block, (float)eps, (float)min_8bit, \
                (float)max_8bit, scale_num_rows, scale_stride);            \
      }                                                                    \
    } else {                                                               \
      assert(!scale_ue8m0);                                                \
      per_token_group_quant_8bit_kernel<T, DST_DTYPE, false>               \
          <<<grid, block, 0, stream>>>(                                    \
              static_cast<T*>(input.data_ptr()), output_q.data_ptr(),      \
              static_cast<float*>(output_s.data_ptr()), group_size,        \
              num_groups, groups_per_block, (float)eps, (float)min_8bit,   \
              (float)max_8bit);                                            \
    }                                                                      \
  } while (0)

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(
      input.scalar_type(), scalar_t, [&] {
        if (dst_type == at::ScalarType::Char) {
          LAUNCH_KERNEL(scalar_t, int8_t);
          return true;
        } else if (dst_type == at::ScalarType::Float8_e4m3fn) {
          LAUNCH_KERNEL(scalar_t, c10::Float8_e4m3fn);
          return true;
        }
        return false;
      });

#undef LAUNCH_KERNEL
}

void sgl_per_token_group_quant_int8(torch::Tensor input, torch::Tensor output_q,
                                    torch::Tensor output_s, int64_t group_size,
                                    double eps, double int8_min,
                                    double int8_max) {
  sgl_per_token_group_quant_8bit(input, output_q, output_s, group_size, eps,
                                 int8_min, int8_max);
}

void sgl_per_token_group_quant_fp8(torch::Tensor input, torch::Tensor output_q,
                                   torch::Tensor output_s, int64_t group_size,
                                   double eps, double fp8_min, double fp8_max,
                                   bool scale_ue8m0) {
  sgl_per_token_group_quant_8bit(input, output_q, output_s, group_size, eps,
                                 fp8_min, fp8_max, scale_ue8m0);
}