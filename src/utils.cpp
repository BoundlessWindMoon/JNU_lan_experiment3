#include <hip/hip_fp16.h>  // 包含FP16支持
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

extern "C" __global__ void transposeKernel(_Float16* A, _Float16* At, int m, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // if (row < m && col < k) {
    At[col * m + row] = A[row * k + col];
    // }
}