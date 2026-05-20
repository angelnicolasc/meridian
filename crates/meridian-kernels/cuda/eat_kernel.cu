// EAT (Entropy After </think>) signal kernel.
//
// Sums the softmax probability mass placed on the configured set of
// `</think>` token ids:
//
//     EAT = sum_{j in end_ids} softmax(x)_j
//         = sum_{j in end_ids} exp(x_j - m) / Z
//
// where m = max_i x_i and Z = sum_i exp(x_i - m). Uses the standard
// two-pass log-sum-exp followed by a sparse gather over `end_ids`.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "entropy_probe.cuh"

namespace {

constexpr int kBlockThreads = 256;
constexpr int kWarpSize     = 32;
constexpr int kMaxWarps     = kBlockThreads / kWarpSize;

template <typename T>
__device__ __forceinline__ float to_float(T v);

template <> __device__ __forceinline__ float to_float<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}
template <> __device__ __forceinline__ float to_float<__half>(__half v) {
    return __half2float(v);
}

__device__ __forceinline__ float warp_max(float v) {
    for (int d = kWarpSize / 2; d > 0; d >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, v, d);
        v = fmaxf(v, other);
    }
    return v;
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int d = kWarpSize / 2; d > 0; d >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, d);
    }
    return v;
}

template <typename T>
__global__ void eat_kernel(
    const T* __restrict__   logits,
    int                     vocab_size,
    const int32_t* __restrict__ end_ids,
    int                     n_end_ids,
    float* __restrict__     out_eat
) {
    const int tid     = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane    = tid % kWarpSize;

    __shared__ float s_max[kMaxWarps];
    __shared__ float s_lse[kMaxWarps];
    __shared__ float s_block_max;
    __shared__ float s_block_lse;

    // ---- block-wide max -------------------------------------------------------
    float local_max = -INFINITY;
    for (int i = tid; i < vocab_size; i += kBlockThreads) {
        local_max = fmaxf(local_max, to_float<T>(logits[i]));
    }
    local_max = warp_max(local_max);
    if (lane == 0) s_max[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < kMaxWarps) ? s_max[lane] : -INFINITY;
        v = warp_max(v);
        if (lane == 0) s_block_max = v;
    }
    __syncthreads();

    // ---- block-wide log-sum-exp ----------------------------------------------
    float local_lse = 0.0f;
    for (int i = tid; i < vocab_size; i += kBlockThreads) {
        local_lse += __expf(to_float<T>(logits[i]) - s_block_max);
    }
    local_lse = warp_sum(local_lse);
    if (lane == 0) s_lse[warp_id] = local_lse;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < kMaxWarps) ? s_lse[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) s_block_lse = v;
    }
    __syncthreads();

    // ---- sparse gather: sum over end_ids -------------------------------------
    if (tid == 0) {
        *out_eat = 0.0f;
    }
    __syncthreads();

    float local_eat = 0.0f;
    for (int j = tid; j < n_end_ids; j += kBlockThreads) {
        const int32_t idx = end_ids[j];
        if (idx >= 0 && idx < vocab_size) {
            local_eat += __expf(to_float<T>(logits[idx]) - s_block_max);
        }
    }
    local_eat = warp_sum(local_eat);
    if (lane == 0 && local_eat > 0.0f) {
        atomicAdd(out_eat, local_eat / s_block_lse);
    }
}

cudaError_t dispatch_eat(
    const void*    logits,
    int            vocab_size,
    int            dtype,
    const int32_t* end_ids,
    int            n_end_ids,
    float*         out_eat,
    cudaStream_t   stream
) {
    dim3 grid(1);
    dim3 block(kBlockThreads);
    switch (dtype) {
        case kMeridianDtypeF32:
            eat_kernel<float><<<grid, block, 0, stream>>>(
                static_cast<const float*>(logits), vocab_size,
                end_ids, n_end_ids, out_eat);
            break;
        case kMeridianDtypeBf16:
            eat_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(logits), vocab_size,
                end_ids, n_end_ids, out_eat);
            break;
        case kMeridianDtypeF16:
            eat_kernel<__half><<<grid, block, 0, stream>>>(
                static_cast<const __half*>(logits), vocab_size,
                end_ids, n_end_ids, out_eat);
            break;
        default:
            return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

} // namespace

extern "C" int meridian_eat_launch(
    const void*    logits,
    size_t         vocab_size,
    int            dtype,
    const int32_t* think_end_ids,
    size_t         n_end_ids,
    float*         out_eat
) {
    if (logits == nullptr || out_eat == nullptr)               return -3;
    if (vocab_size == 0 || vocab_size > INT32_MAX)             return -4;
    if (n_end_ids > 0 && think_end_ids == nullptr)             return -5;
    if (n_end_ids > INT32_MAX)                                 return -6;

    cudaError_t e = dispatch_eat(
        logits,
        static_cast<int>(vocab_size),
        dtype,
        think_end_ids,
        static_cast<int>(n_end_ids),
        out_eat,
        /*stream=*/0
    );
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}
