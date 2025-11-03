#include "solve.h"
#include <cuda_runtime.h>
#include <math.h>

constexpr int kThreadsPerBlock = 1024;

__global__ void gpu_bn_inference_nchw(
    const float* __restrict__ x,      // [N,C,H,W]
    const float* __restrict__ gamma,  // [C]
    const float* __restrict__ beta,   // [C]
    float* __restrict__ y,            // [N,C,H,W]
    int N, int C, int H, int W,
    float eps)
{
    const int c = blockIdx.x;
    const int tid = threadIdx.x;

    const int spatialSize = H * W;
    const int imageSize = C * spatialSize;

    float sum = 0.0f;
    float sumSq = 0.0f;

    for (int n = 0; n < N; ++n) {

        const int base = (n * imageSize) + (c * spatialSize);

        for (int idx = tid; idx < spatialSize; idx += blockDim.x) {

            const float v = x[base + idx];
            sum += v;
            sumSq += v * v;
        }
    }

    __shared__ float s_sum[kThreadsPerBlock];
    __shared__ float s_sumSq[kThreadsPerBlock];

    s_sum[tid] = sum;
    s_sumSq[tid] = sumSq;
    __syncthreads();

    for (int offset = (blockDim.x / 2); offset > 0; offset >>= 1) {

        if (tid < offset) {

            s_sum[tid] += s_sum[tid + offset];
            s_sumSq[tid] += s_sumSq[tid + offset];
        }

        __syncthreads();
    }

    float mean = 0.0f;
    float invStd = 0.0f;

    if (tid == 0) {

        const float count = static_cast<float>(N * spatialSize);

        mean = s_sum[0] / count;

        const float ex2 = s_sumSq[0] / count;
        const float var = fmaxf(ex2 - mean * mean, 0.0f);

        invStd = 1.0f / (sqrtf(var + eps));

        s_sum[0] = mean;
        s_sumSq[0] = invStd;
    }

    __syncthreads();

    mean   = s_sum[0];
    invStd = s_sumSq[0];

    float g = gamma[c];
    float b = beta[c];

    for (int n = 0; n < N; ++n) {

        int base = (n * imageSize) + (c * spatialSize);

        for (int idx = tid; idx < spatialSize; idx += blockDim.x) {

            const float v = x[base + idx];
            const float xHat = (v - mean) * invStd;

            y[base + idx] = (g * xHat) + b;
        }
    }
}

void solve(const float* input, // [N,C,H,W] (input shape: batch, channel, height, weight)  
           const float* gamma, // [C] 
           const float* beta, // [C]
           float* output, // [N,C,H,W]
           int N, int C, int H, int W, float eps)
{
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return;

    dim3 block(kThreadsPerBlock);
    dim3 grid(C);
    gpu_bn_inference_nchw<<<grid, block>>>(input, gamma, beta, output, N, C, H, W, eps);
}
