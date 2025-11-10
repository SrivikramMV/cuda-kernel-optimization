#include "solve.h"
#include <cuda_runtime.h>

constexpr int kThreadsPerBlock = 256;

__global__ void gpu_dwconv2d(
    const float* __restrict__ x,      // [N,C,H,W]
    const float* __restrict__ k,      // [C,1,KH,KW] flattened as [C*KH*KW]
    float* __restrict__ y,            // [N,C,OH,OW]
    int N, int C, int H, int W,
    int KH, int KW,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int OH, int OW)
{
    const int tid = threadIdx.x;
    const int n = blockIdx.y;
    const int c = blockIdx.z;

    const int out_hw = OH * OW;

    const int idx = (blockIdx.x * blockDim.x) + tid;

    if (idx >= out_hw) return;

    const int oh = idx / OW;
    const int ow = idx % OW;

    const int w0 = (ow * stride_w) - pad_w;
    const int h0 = (oh * stride_h) - pad_h;

    float sum = 0.0f;

    for (int kh = 0; kh < KH; ++kh) {

        const int ih = h0 + kh;

        if (ih < 0 || ih >= H) continue;

        for (int kw = 0; kw < KW; ++kw) {

            const int iw = w0 + kw;

            if (iw < 0 || iw >= W) continue;

            const int x_idx = (((((n * C) + c) * H) + ih) * W) + iw;
            const int k_idx = (((c * KH) + kh) * KW) + kw;

            const float xv = x[x_idx];
            const float kv = k[k_idx];

            sum += xv * kv;
        }
    }

    const int y_idx = (((((n * C) + c) * OH) + oh) * OW) + ow;
    y[y_idx] = sum;
}

void solve(const float* input,      // [N,C,H,W] 
           const float* kernel,     // [C,1,KH,KW] flattened as [C*KH*KW]
           float* output,           // [N,C,OH,OW]
           int N, int C, int H, int W, // input shape: batch, channel, height, weight
           int KH, int KW, // kernel height and weight
           int pad_h, int pad_w,    // kernel padding
           int stride_h, int stride_w)  // kernel stride
{
    if (N<=0 || C<=0 || H<=0 || W<=0 || KH<=0 || KW<=0) return;

    int OH = (H + 2*pad_h - KH) / stride_h + 1;
    int OW = (W + 2*pad_w - KW) / stride_w + 1;
    if (OH <= 0 || OW <= 0) return;

   
    const int out_hw = OH * OW;

    dim3 block(kThreadsPerBlock);
    dim3 grid((out_hw + block.x - 1) / block.x, N, C);
    gpu_dwconv2d<<<grid, block>>>(
        input, kernel, output,
        N, C, H, W, KH, KW,
        pad_h, pad_w, stride_h, stride_w,
        OH, OW);
}
