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

struct Shape {
    int n;
    int c;
    int h;
    int w;
};

std::vector<float> reference_batch_norm(
    const std::vector<float>& input,
    const std::vector<float>& gamma,
    const std::vector<float>& beta,
    Shape shape,
    float eps) {
    const int spatial = shape.h * shape.w;
    const int image = shape.c * spatial;
    std::vector<float> output(input.size());

    for (int c = 0; c < shape.c; ++c) {
        double sum = 0.0;
        double sum_sq = 0.0;
        for (int n = 0; n < shape.n; ++n) {
            const int base = n * image + c * spatial;
            for (int i = 0; i < spatial; ++i) {
                const double value = input[base + i];
                sum += value;
                sum_sq += value * value;
            }
        }

        const double count = static_cast<double>(shape.n * spatial);
        const double mean = sum / count;
        const double variance = std::max(sum_sq / count - mean * mean, 0.0);
        const double inv_std = 1.0 / std::sqrt(variance + eps);

        for (int n = 0; n < shape.n; ++n) {
            const int base = n * image + c * spatial;
            for (int i = 0; i < spatial; ++i) {
                output[base + i] = static_cast<float>(
                    gamma[c] * (input[base + i] - mean) * inv_std + beta[c]);
            }
        }
    }
    return output;
}

void run_case(Shape shape, unsigned seed) {
    const std::size_t count = static_cast<std::size_t>(shape.n) * shape.c * shape.h * shape.w;
    std::mt19937 generator(seed);
    std::normal_distribution<float> values(0.0f, 1.0f);

    std::vector<float> input(count);
    std::vector<float> gamma(shape.c);
    std::vector<float> beta(shape.c);
    std::generate(input.begin(), input.end(), [&] { return values(generator); });
    std::generate(gamma.begin(), gamma.end(), [&] { return values(generator); });
    std::generate(beta.begin(), beta.end(), [&] { return values(generator); });

    const float eps = 1e-5f;
    const auto expected = reference_batch_norm(input, gamma, beta, shape, eps);
    std::vector<float> actual(count);

    float *d_input = nullptr, *d_gamma = nullptr, *d_beta = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, shape.c * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, shape.c * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, gamma.data(), shape.c * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, beta.data(), shape.c * sizeof(float), cudaMemcpyHostToDevice));

    solve(d_input, d_gamma, d_beta, d_output, shape.n, shape.c, shape.h, shape.w, eps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(actual.data(), d_output, count * sizeof(float), cudaMemcpyDeviceToHost));

    float max_error = 0.0f;
    for (std::size_t i = 0; i < count; ++i) {
        max_error = std::max(max_error, std::abs(actual[i] - expected[i]));
    }

    cudaFree(d_input);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_output);

    if (max_error > 2e-4f) {
        throw std::runtime_error("Batch-normalisation mismatch: max error=" + std::to_string(max_error));
    }
    std::cout << "PASS N=" << shape.n << " C=" << shape.c << " H=" << shape.h
              << " W=" << shape.w << " max_error=" << max_error << '\n';
}

int main() {
    try {
        run_case({2, 3, 4, 5}, 7);
        run_case({4, 8, 16, 16}, 11);
        run_case({1, 16, 7, 9}, 21);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
