#include "solve.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t error = (call);                                        \
        if (error != cudaSuccess) {                                              \
            throw std::runtime_error(cudaGetErrorString(error));                 \
        }                                                                        \
    } while (false)

struct Case {
    int n, c, h, w, kh, kw, pad_h, pad_w, stride_h, stride_w;
};

std::vector<float> reference_depthwise(
    const std::vector<float>& input,
    const std::vector<float>& kernel,
    const Case& config) {
    const int oh = (config.h + 2 * config.pad_h - config.kh) / config.stride_h + 1;
    const int ow = (config.w + 2 * config.pad_w - config.kw) / config.stride_w + 1;
    std::vector<float> output(static_cast<std::size_t>(config.n) * config.c * oh * ow, 0.0f);

    for (int n = 0; n < config.n; ++n) {
        for (int c = 0; c < config.c; ++c) {
            for (int out_h = 0; out_h < oh; ++out_h) {
                for (int out_w = 0; out_w < ow; ++out_w) {
                    float sum = 0.0f;
                    for (int kh = 0; kh < config.kh; ++kh) {
                        const int in_h = out_h * config.stride_h - config.pad_h + kh;
                        if (in_h < 0 || in_h >= config.h) continue;
                        for (int kw = 0; kw < config.kw; ++kw) {
                            const int in_w = out_w * config.stride_w - config.pad_w + kw;
                            if (in_w < 0 || in_w >= config.w) continue;
                            const int input_index = ((n * config.c + c) * config.h + in_h) * config.w + in_w;
                            const int kernel_index = (c * config.kh + kh) * config.kw + kw;
                            sum += input[input_index] * kernel[kernel_index];
                        }
                    }
                    output[((n * config.c + c) * oh + out_h) * ow + out_w] = sum;
                }
            }
        }
    }
    return output;
}

void run_case(const Case& config, unsigned seed) {
    const int oh = (config.h + 2 * config.pad_h - config.kh) / config.stride_h + 1;
    const int ow = (config.w + 2 * config.pad_w - config.kw) / config.stride_w + 1;
    const std::size_t input_count = static_cast<std::size_t>(config.n) * config.c * config.h * config.w;
    const std::size_t kernel_count = static_cast<std::size_t>(config.c) * config.kh * config.kw;
    const std::size_t output_count = static_cast<std::size_t>(config.n) * config.c * oh * ow;

    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> values(-1.0f, 1.0f);
    std::vector<float> input(input_count);
    std::vector<float> kernel(kernel_count);
    std::generate(input.begin(), input.end(), [&] { return values(generator); });
    std::generate(kernel.begin(), kernel.end(), [&] { return values(generator); });

    const auto expected = reference_depthwise(input, kernel, config);
    std::vector<float> actual(output_count);
    float *d_input = nullptr, *d_kernel = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, input_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, output_count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), input_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, kernel.data(), kernel_count * sizeof(float), cudaMemcpyHostToDevice));

    solve(d_input, d_kernel, d_output, config.n, config.c, config.h, config.w,
          config.kh, config.kw, config.pad_h, config.pad_w, config.stride_h, config.stride_w);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(actual.data(), d_output, output_count * sizeof(float), cudaMemcpyDeviceToHost));

    float max_error = 0.0f;
    for (std::size_t i = 0; i < output_count; ++i) {
        max_error = std::max(max_error, std::abs(actual[i] - expected[i]));
    }

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    if (max_error > 2e-4f) {
        throw std::runtime_error("Depthwise-convolution mismatch: max error=" + std::to_string(max_error));
    }
    std::cout << "PASS N=" << config.n << " C=" << config.c << " H=" << config.h
              << " W=" << config.w << " K=" << config.kh << 'x' << config.kw
              << " max_error=" << max_error << '\n';
}

int main() {
    try {
        run_case({2, 3, 9, 11, 3, 3, 1, 1, 1, 1}, 7);
        run_case({1, 8, 14, 14, 5, 5, 2, 2, 2, 2}, 11);
        run_case({4, 4, 8, 10, 3, 3, 0, 0, 1, 2}, 21);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
