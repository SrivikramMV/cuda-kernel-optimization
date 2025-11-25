# Recorded benchmark results

The kernels were evaluated on an NVIDIA RTX 5090 using CUDA 12.9. The tables below report the recorded average kernel time, effective throughput and approximate memory bandwidth after warm-up iterations.

## Batch normalisation

| Workload | Average time | Throughput | Approx. bandwidth |
|---|---:|---:|---:|
| ResNet-50 stem | 0.082 ms | 155.94 G elements/s | 1,247.54 GB/s |
| MobileNetV2 early layer | 0.146 ms | 87.90 G elements/s | 703.19 GB/s |
| LLaMA/Mistral 7B-shaped tensor | 0.198 ms | 169.23 G elements/s | 1,354.00 GB/s |
| LLaMA 70B-width tensor | 0.738 ms | 181.86 G elements/s | 1,455.00 GB/s |

All four deterministic correctness and performance cases passed.

## Depthwise convolution

| Workload | Average time | Throughput | Approx. bandwidth |
|---|---:|---:|---:|
| 3x3, stride 2, 112x112, 32 channels | 0.080 ms | 0.72 TFLOP/s | 800.27 GB/s |
| 3x3, stride 1, 56x56, 64 channels | 0.052 ms | 2.23 TFLOP/s | 992.40 GB/s |
| 3x3, stride 1, 14x14, 1,024 channels | 0.018 ms | 1.62 TFLOP/s | 722.73 GB/s |
| 5x5, stride 1, 28x28, 128 channels | 0.022 ms | 3.63 TFLOP/s | 580.65 GB/s |
| 5x5, stride 2, 28x28, 128 channels | 0.011 ms | 1.75 TFLOP/s | 699.94 GB/s |

All five deterministic correctness and performance cases passed.

Performance will vary with the GPU architecture, CUDA version and benchmark configuration.
