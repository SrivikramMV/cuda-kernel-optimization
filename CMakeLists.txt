cmake_minimum_required(VERSION 3.24)
project(cuda_kernel_optimization LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

enable_testing()

add_executable(test_batch_norm
    tests/test_batch_norm.cu
    kernels/batch_norm/gpu_bn_inference.cu)
target_include_directories(test_batch_norm PRIVATE kernels/batch_norm)
set_target_properties(test_batch_norm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_depthwise_conv
    tests/test_depthwise_conv.cu
    kernels/depthwise_conv/gpu_dwconv2d_inference.cu)
target_include_directories(test_depthwise_conv PRIVATE kernels/depthwise_conv)
set_target_properties(test_depthwise_conv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_test(NAME batch_norm_correctness COMMAND test_batch_norm)
add_test(NAME depthwise_conv_correctness COMMAND test_depthwise_conv)
