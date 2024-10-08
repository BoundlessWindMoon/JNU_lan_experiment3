#include <hip/hip_fp16.h>  // 包含FP16支持
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

extern "C" __global__ void transpose_kernel(_Float16* A, _Float16* At, int m, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // if (row < m && col < k) {
    At[col * m + row] = A[row * k + col];
    // }
}


extern "C" __global__ void reshape_kernel(const _Float16* output_gemm_device, _Float16* output_gemm_device_rearrange,
                               int n, int k, int output_h, int output_w) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * k * output_h * output_w;

    if (index < total_elements) {
        int j = index % output_w;
        int i = (index / output_w) % output_h;
        int ki = (index / (output_w * output_h)) % k;
        int b = index / (output_w * output_h * k);

        output_gemm_device_rearrange[b * k * output_h * output_w + ki * output_h * output_w + i * output_w + j] =
            output_gemm_device[ki * n * output_h * output_w + b * output_h * output_w + i * output_w + j];
    }
}

void launch_reshape_kernel(const _Float16* output_gemm_device, _Float16* output_gemm_device_rearrange,
                 int n, int k, int output_h, int output_w) {
    int total_elements = n * k * output_h * output_w;

    // 分配块大小和线程大小
    int blockSize = 1024; // 每个块中的线程数
    int gridSize = (total_elements + blockSize - 1) / blockSize; // 计算网格大小

    // 启动内核
    hipLaunchKernelGGL(reshape_kernel, gridSize, blockSize, 0, 0,
                       output_gemm_device, output_gemm_device_rearrange,
                       n, k, output_h, output_w);

}