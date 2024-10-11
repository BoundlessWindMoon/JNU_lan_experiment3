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

extern "C" __global__ __launch_bounds__(1024) void reshape_2D_kernel(const _Float16* output_gemm_device, _Float16* output_gemm_device_rearrange,
                               int n, int k, int output_h, int output_w,
                               int expand_row, int expand_col) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * k * output_h * output_w;

    if (index < total_elements) {
        int j = index % output_w;
        int i = (index / output_w) % output_h;
        int ki = (index / (output_w * output_h)) % k;
        int b = index / (output_w * output_h * k);

        int input_row = (b / expand_col) * k + ki;
        int input_col = b % expand_col;

        output_gemm_device_rearrange[b * k * output_h * output_w + ki * output_h * output_w + i * output_w + j] =
            output_gemm_device[input_row * expand_col * output_h * output_w + input_col * output_h * output_w + i * output_w + j];
    }
}

extern "C" __global__ __launch_bounds__(1024) void im2col_2D_kernel(const _Float16* data_im, int n, int channels, int height, int width,
                                    int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w, 
                                    int span_expand_row, int span_unfold_crs, int span_expand_col, int expand_row, int expand_col,
                                    int output_h, int output_w, _Float16* data_col) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int crs = blockIdx.y;  // 展开的卷积核索引，表示 (c * r * s)
    int c = crs / (kernel_h * kernel_w);  // 计算通道索引
    int kh = (crs % (kernel_h * kernel_w)) / kernel_w;  // 计算卷积核的行索引
    int kw = (crs % (kernel_h * kernel_w)) % kernel_w;  // 计算卷积核的列索引

    if (index <  span_unfold_crs) {
        int c_b = index / (output_h * output_w);  // 计算批次索引
        int r_b = blockIdx.z;
        int b = r_b * expand_col + c_b;
        int oh = (index % (output_h * output_w)) / output_w;  // 计算输出行索引
        int ow = (index % (output_h * output_w)) % output_w;  // 计算输出列索引

        // 计算输入矩阵中的行和列位置，考虑padding
        int im_row = kh - pad_h + oh * stride_h;
        int im_col = kw - pad_w + ow * stride_w;

        // 计算最终在 data_col 中的偏移
        //int offset_col = crs * n * output_h * output_w + b * output_h * output_w + oh * output_w + ow;
        int offset_col = r_b * span_expand_row + crs * span_unfold_crs + c_b * span_expand_col  + oh * output_w + ow; 

        if (im_row >= 0 && im_row < height && im_col >= 0 && im_col < width) {
            // 从输入矩阵中读取值并赋值到输出矩阵中
            data_col[offset_col] = data_im[(b * channels + c) * height * width + im_row * width + im_col];
        } else {
            // 填充0值
            data_col[offset_col] = 0.0;
        }
    }
}

void launch_im2col_2D_kernel(const _Float16* data_im_device, 
                        int n, int channels, int height, int width, int kernel_h, int kernel_w, int output_h, int output_w,
                        int expand_row, int expand_col,
                        _Float16* data_col_device) {
    int span_expand_col = output_h * output_w;
    int span_unfold_crs = expand_col * span_expand_col;
    int span_expand_row = channels * kernel_h * kernel_w * span_unfold_crs;
    
    // 每个block处理1024个index
    dim3 blockSize(1024);  // 每个block中的线程数
    // grid的x维度为处理n * output_h * output_w的大小，y维度为crs
    dim3 gridSize((expand_col * output_h * output_w + blockSize.x - 1) / blockSize.x, channels * kernel_h * kernel_w, n / expand_col);

    // 启动CUDA核函数
    im2col_2D_kernel<<<gridSize, blockSize>>>(data_im_device,
                                                n, channels, height, width, kernel_h, kernel_w, 1,1,1,1,
                                                span_expand_row, span_unfold_crs, span_expand_col, expand_row, expand_col,
                                                output_h, output_w,
                                                data_col_device);

    hipError_t error = hipGetLastError();
    if (error != hipSuccess) {
        printf("HIP kernel launch failed with error: %s\n", hipGetErrorString(error));
        // 这里可以添加更多的错误处理代码
    }
}

void launch_reshape_2D_kernel(const _Float16* output_gemm_device, _Float16* output_gemm_device_rearrange,
                 int n, int k, int output_h, int output_w,
                 int expand_row, int expand_col) {
    int total_elements = n * k * output_h * output_w;

    // 分配块大小和线程大小
    int blockSize = 1024; // 每个块中的线程数
    int gridSize = (total_elements + blockSize - 1) / blockSize; // 计算网格大小

    // 启动内核
    hipLaunchKernelGGL(reshape_2D_kernel, gridSize, blockSize, 0, 0,
                       output_gemm_device, output_gemm_device_rearrange,
                       n, k, output_h, output_w,
                       expand_row, expand_col);
}

void launch_transpose_kernel(_Float16* A, _Float16* At, int M, int K) {
    dim3 grid((K + 15) / 16, (M + 15) / 16);
    dim3 block(16, 16);
    transpose_kernel <<<grid, block>>> (A, At, M, K);
}