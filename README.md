# CUDA Kernel Optimization for ML Systems

This project contains my CUDA implementations of two common machine-learning operations:

- NCHW batch-normalisation inference
- NCHW depthwise 2D convolution

I built both kernels from low-level CUDA operations rather than relying on libraries such as cuDNN. My main objective was to understand how tensor operations that are normally represented by a single framework call can be mapped onto GPU threads, blocks and memory.

Both implementations were tested against CPU reference outputs and passed the complete correctness and performance test suite.

## Batch normalisation

The first kernel implements batch-normalisation inference for an input tensor with shape `N x C x H x W`.

For each channel, the kernel calculates the mean and variance across the batch and spatial dimensions before applying:

$$
y = \gamma \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

I used one CUDA block per channel. Threads within the block divide the flattened `N x H x W` region between themselves and independently accumulate partial sums and squared sums.

I then store these partial results in shared memory and combine them using a block-level parallel reduction. Once the final mean and inverse standard deviation are available, the threads reuse those values to normalise their assigned input elements and apply the learned `gamma` and `beta` parameters.

The implementation uses:

- One CUDA block per channel
- Strided accumulation across the batch and spatial dimensions
- Shared-memory reduction for channel statistics
- Coalesced reads and writes under NCHW layout
- A fused normalisation and affine-output pass
- A 1,024-thread block configuration

I experimented with the block configuration and found that increasing the block size to 1,024 threads improved performance on the target GPU. This allowed more of the channel reduction to happen in parallel without introducing a more complicated multi-stage reduction.

I considered splitting the reduction across multiple blocks, but the simpler one-block-per-channel design already passed every performance test. I therefore kept the implementation compact rather than introducing additional kernels, temporary buffers and global synchronisation.

## Depthwise convolution

The second kernel implements 2D depthwise convolution. Unlike standard convolution, each input channel is processed independently using its own spatial filter.

I arranged the CUDA grid across three dimensions:

- The `x` dimension covers tiles of output positions.
- The `y` dimension identifies the batch element.
- The `z` dimension identifies the channel.

Each thread is responsible for one output element. It calculates its output row and column, maps them to the corresponding input window using the supplied stride and padding, and then accumulates the `KH x KW` convolution result.

The implementation supports arbitrary batch and channel dimensions, configurable kernel sizes, independent horizontal and vertical strides, explicit padding and one filter per input channel.

I used 256-thread blocks and flattened the output spatial dimensions so that neighbouring threads generally process neighbouring output columns. With NCHW tensors, this produces naturally coalesced memory access along the width dimension.

I kept the convolution window as a pair of direct nested loops. Since the filters used in the benchmark were small and every channel was independent, shared-memory tiling would have introduced additional complexity and synchronisation without guaranteeing a meaningful improvement.

## Performance

I evaluated the kernels across CNN and transformer-shaped workloads. The complete recorded results are available in [`results/benchmark_summary.md`](results/benchmark_summary.md).

### Batch normalisation

| Workload | Average time | Throughput | Approx. bandwidth |
|---|---:|---:|---:|
| ResNet-50 stem | 0.082 ms | 155.94 G elements/s | 1,247.54 GB/s |
| MobileNetV2 early layer | 0.146 ms | 87.90 G elements/s | 703.19 GB/s |
| LLaMA/Mistral 7B-shaped tensor | 0.198 ms | 169.23 G elements/s | 1,354.00 GB/s |
| LLaMA 70B-width tensor | 0.738 ms | 181.86 G elements/s | 1,455.00 GB/s |

### Depthwise convolution

| Workload | Average time | Throughput | Approx. bandwidth |
|---|---:|---:|---:|
| 3x3, stride 2, 112x112, 32 channels | 0.080 ms | 0.72 TFLOP/s | 800.27 GB/s |
| 3x3, stride 1, 56x56, 64 channels | 0.052 ms | 2.23 TFLOP/s | 992.40 GB/s |
| 3x3, stride 1, 14x14, 1,024 channels | 0.018 ms | 1.62 TFLOP/s | 722.73 GB/s |
| 5x5, stride 1, 28x28, 128 channels | 0.022 ms | 3.63 TFLOP/s | 580.65 GB/s |

Performance will vary depending on the GPU architecture, CUDA version and benchmark configuration.

## Repository structure

```text
.
|-- kernels/
|   |-- batch_norm/
|   |   |-- gpu_bn_inference.cu
|   |   `-- solve.h
|   `-- depthwise_conv/
|       |-- gpu_dwconv2d_inference.cu
|       `-- solve.h
|-- tests/
|   |-- test_batch_norm.cu
|   `-- test_depthwise_conv.cu
|-- results/
|   `-- benchmark_summary.md
|-- CMakeLists.txt
`-- README.md
```

I have kept the repository deliberately small. The main implementations remain in the two CUDA source files, while the additional code provides a reproducible way to build and test them.

## Building the project

Requirements:

- CUDA Toolkit 12 or later
- CMake 3.24 or later
- An NVIDIA GPU with CUDA support

```bash
git clone <repository-url>
cd cuda-kernel-optimization

cmake -S . -B build
cmake --build build --config Release
```

## Running the tests

```bash
ctest --test-dir build --output-on-failure
```

The tests compare the CUDA results against direct CPU implementations across several tensor shapes, convolution sizes, strides and padding configurations.

## What I explored

This project gave me practical experience with:

- Mapping tensor operations onto CUDA's grid and block hierarchy
- Designing block-level parallel reductions
- Managing shared memory and thread synchronisation
- Working directly with NCHW tensor layouts
- Reasoning about coalesced global-memory access
- Choosing launch configurations for different workloads
- Using NVIDIA profiling tools to inspect kernel behaviour
- Deciding when a simpler kernel is preferable to a more elaborate optimisation
