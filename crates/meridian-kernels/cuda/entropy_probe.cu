// Shannon-entropy CUDA kernel.
//
// Computes H(p) over a vocab-size logit array using the log-sum-exp identity:
//
//     H(p) = log Z - (1/Z) * sum_i exp(x_i - m) * (x_i - m)
//
// where m = max_i x_i and Z = sum_i exp(x_i - m). Numerically stable for the
// full bf16/fp16/fp32 range a serving model produces.
//
// Launch: 1 block, 256 threads. Each thread strides over vocab_size.
// Template-specialised on T ∈ {float, __nv_bfloat16, __half} so the only
// branch at runtime is the dtype dispatch in the entry point.

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
__global__ void entropy_kernel(
    const T* __restrict__ logits,
    int vocab_size,
    float* __restrict__ out_entropy
) {
    const int tid     = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane    = tid % kWarpSize;

    __shared__ float s_partial[kMaxWarps];

    // ---- Pass 1: block-wide max ------------------------------------------------
    float local_max = -INFINITY;
    for (int i = tid; i < vocab_size; i += kBlockThreads) {
        local_max = fmaxf(local_max, to_float<T>(logits[i]));
    }
    local_max = warp_max(local_max);
    if (lane == 0) s_partial[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < kMaxWarps) ? s_partial[lane] : -INFINITY;
        v = warp_max(v);
        if (lane == 0) s_partial[0] = v;
    }
    __syncthreads();
    const float block_max = s_partial[0];

    // ---- Pass 2: log-sum-exp + weighted shifted sum ----------------------------
    float local_lse = 0.0f;
    float local_ent = 0.0f;
    for (int i = tid; i < vocab_size; i += kBlockThreads) {
        const float x  = to_float<T>(logits[i]) - block_max;
        const float ex = __expf(x);
        local_lse += ex;
        local_ent += ex * x;
    }
    local_lse = warp_sum(local_lse);
    local_ent = warp_sum(local_ent);

    __shared__ float s_lse[kMaxWarps];
    __shared__ float s_ent[kMaxWarps];
    if (lane == 0) {
        s_lse[warp_id] = local_lse;
        s_ent[warp_id] = local_ent;
    }
    __syncthreads();

    if (warp_id == 0) {
        float lse = (lane < kMaxWarps) ? s_lse[lane] : 0.0f;
        float ent = (lane < kMaxWarps) ? s_ent[lane] : 0.0f;
        lse = warp_sum(lse);
        ent = warp_sum(ent);
        if (lane == 0) {
            // H = log(Z) - sum(ex * x) / Z, with x = (x_i - m). The "+ m" cancels
            // because both terms include it.
            const float h = __logf(lse) - ent / lse;
            *out_entropy = h;
        }
    }
}

cudaError_t dispatch_entropy(
    const void* logits,
    int vocab_size,
    int dtype,
    float* out_entropy,
    cudaStream_t stream
) {
    dim3 grid(1);
    dim3 block(kBlockThreads);
    switch (dtype) {
        case kMeridianDtypeF32:
            entropy_kernel<float><<<grid, block, 0, stream>>>(
                static_cast<const float*>(logits), vocab_size, out_entropy);
            break;
        case kMeridianDtypeBf16:
            entropy_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(logits), vocab_size, out_entropy);
            break;
        case kMeridianDtypeF16:
            entropy_kernel<__half><<<grid, block, 0, stream>>>(
                static_cast<const __half*>(logits), vocab_size, out_entropy);
            break;
        default:
            return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

} // namespace

extern "C" int meridian_entropy_launch(
    const void* logits,
    size_t      vocab_size,
    int         dtype,
    float*      out_entropy
) {
    if (logits == nullptr || out_entropy == nullptr) return -3;
    if (vocab_size == 0 || vocab_size > INT32_MAX)  return -4;

    cudaError_t e = dispatch_entropy(
        logits,
        static_cast<int>(vocab_size),
        dtype,
        out_entropy,
        /*stream=*/0
    );
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}
