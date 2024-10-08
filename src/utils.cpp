#include <hip/hip_fp16.h>  // 包含FP16支持
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include "conv2d.h"

extern "C" __global__ void transpose_kernel(_Float16* A, _Float16* At, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // if (row < M && col < K) {
    At[col * M + row] = A[row * K + col];
    // }
}

void launch_transpose_kernel(_Float16* A, _Float16* At, int M, int K) {
    dim3 grid((K + 15) / 16, (M + 15) / 16);
    dim3 block(16, 16);
    transpose_kernel <<<grid, block>>> (A, At, M, K);
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

extern "C" __global__ void im2col_r_1_c_n_kernel(const _Float16* data_im, int n, int channels, int height, int width,
                                    int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w,
                                    int output_h, int output_w, _Float16* data_col) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int crs = blockIdx.y;  // 展开的卷积核索引，表示 (c * r * s)
    int c = crs / (kernel_h * kernel_w);  // 计算通道索引
    int kh = (crs % (kernel_h * kernel_w)) / kernel_w;  // 计算卷积核的行索引
    int kw = (crs % (kernel_h * kernel_w)) % kernel_w;  // 计算卷积核的列索引

    if (index < n * output_h * output_w) {
        int b = index / (output_h * output_w);  // 计算批次索引
        int oh = (index % (output_h * output_w)) / output_w;  // 计算输出行索引
        int ow = (index % (output_h * output_w)) % output_w;  // 计算输出列索引

        // 计算输入矩阵中的行和列位置，考虑padding
        int im_row = kh - pad_h + oh * stride_h;
        int im_col = kw - pad_w + ow * stride_w;

        // 计算最终在 data_col 中的偏移
        int offset_col = crs * n * output_h * output_w + b * output_h * output_w + oh * output_w + ow;

        if (im_row >= 0 && im_row < height && im_col >= 0 && im_col < width) {
            // 从输入矩阵中读取值并赋值到输出矩阵中
            data_col[offset_col] = data_im[(b * channels + c) * height * width + im_row * width + im_col];
        } else {
            // 填充0值
            data_col[offset_col] = 0.0;
        }
    }
}

void launch_im2col_r_1_c_n_kernel(const _Float16* data_im_device, int n, int channels, int height, int width,
                      int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w,
                      _Float16* data_col_device) {
    int output_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    // 每个block处理1024个index
    dim3 blockSize(1024);  // 每个block中的线程数
    // grid的x维度为处理n * output_h * output_w的大小，y维度为crs
    dim3 gridSize((n * output_h * output_w + blockSize.x - 1) / blockSize.x, channels * kernel_h * kernel_w);

    // 启动CUDA核函数
    hipLaunchKernelGGL(im2col_r_1_c_n_kernel, gridSize, blockSize, 0, 0,
    data_im_device, n, channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, output_h, output_w, data_col_device);

}