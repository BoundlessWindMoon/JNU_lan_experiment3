#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include "conv2d.h"

/*选手需要返回自定义kernel入参结构体的size*/
int getParamsize(__in__ problem_t *problem, __out__ int *paramSize)
{
    *paramSize = sizeof(mykernelParamType);
    return 0;
}

/*选手需要返回自己优化的kernel的grid信息与kernel函数的指针*/
int getkernelInfo(__in__ problem_t *problem, __out__ kernelInfo_t *kernelInfo, __in_out__ void *param)
{

    mykernelParamType *pArgs = (mykernelParamType *)param;
    unsigned int algo = problem->algo;
    unsigned int n = problem->n;
    unsigned int c = problem->c;
    unsigned int h = problem->h;
    unsigned int w = problem->w;
    unsigned int k = problem->k;
    unsigned int r = problem->r;
    unsigned int s = problem->s;
    unsigned int u = problem->u;
    unsigned int v = problem->v;
    unsigned int p = problem->p;
    unsigned int q = problem->q;
    unsigned int outh = (h - r + 2 * p) / u + 1;
    unsigned int outw = (w - s + 2 * q) / v + 1;

    pArgs->pin = problem->in;
    pArgs->pweight = problem->weight;
    pArgs->pweight_trans = problem->weight_trans;
    pArgs->pout = problem->out;
    pArgs->data_col_device = problem->data_col_device;
    pArgs->output_gemm_device = problem->output_gemm_device;
    pArgs->algo = algo;
    pArgs->n = n;
    pArgs->c = c;
    pArgs->h = h;
    pArgs->w = w;
    pArgs->k = k;
    pArgs->r = r;
    pArgs->s = s;
    pArgs->u = u;
    pArgs->v = v;
    pArgs->p = p;
    pArgs->q = q;
    pArgs->Oh = outh;
    pArgs->Ow = outw;
    return 0;
}

void executeConvAlgos(mykernelParamType *param)
{

    unsigned int n = param->n;
    unsigned int c = param->c;
    unsigned int h = param->h;
    unsigned int w = param->w;
    unsigned int k = param->k;
    unsigned int r = param->r;
    unsigned int s = param->s;
    unsigned int p = param->p;
    unsigned int q = param->q;
    unsigned int u = param->u;
    unsigned int v = param->v;
    unsigned int outh = param->Oh;
    unsigned int outw = param->Ow;
    unsigned int algo = param->algo;
    unsigned int M = k;
    unsigned int N = n * outh * outw;
    unsigned int K = c * r * s;
    if (algo == DIRECT_CONV)
    {
        dim3 grid((outh * outw + 15) / 16, (k + 15) / 16, n);
        dim3 block(16, 16, 1);
        directConvolution<<<grid, block>>>(*param);
    }
    else if (algo == IM2COL_GEMM_1BATCH)
    {
        launch_im2col_r_1_c_n_kernel(param->pin, n, c, h, w, r, s, p, q, u, v, param->data_col_device);
        launch_gemm_128x128x8_fp32((__half *)param->pweight, (__half *)param->data_col_device, (__half *)param->output_gemm_device, M, N, K);
        launch_reshape_kernel(param->output_gemm_device, param->pout, n, k, outh, outw);
    }
    else if (algo == IM2COL_GEMM_NBATCH)
    {
        // TODO
    }
    else if (algo == IMPL_GEMM)
    {
        launch_implicit_gemm(outh, outw, k, n, param);
    }
    else if (algo == WINOGRAD)
    {
        // TODO
    }
    else if (algo == IM2COL_GEMM_1BATCH_64)
    {
        launch_im2col_r_1_c_n_kernel(param->pin, n, c, h, w, r, s, p, q, u, v, param->data_col_device);
        launch_gemm_64x64x8_fp32((__half *)param->pweight, (__half *)param->data_col_device, (__half *)param->output_gemm_device, M, N, K);
        launch_reshape_kernel(param->output_gemm_device, param->pout, n, k, outh, outw);
    }
    else if (algo == MMA_NAIVE)
    {
        launch_im2col_r_1_c_n_kernel(param->pin, n, c, h, w, r, s, p, q, u, v, param->data_col_device);
        launch_transpose_kernel(param->pweight, param->pweight_trans, M, K);
        launch_gemm_32x32x16_fp16(param->pweight_trans, param->data_col_device, param->output_gemm_device, M, N, K);
        launch_reshape_kernel(param->output_gemm_device, param->pout, n, k, outh, outw);
    }
}

void convolutionForward(void *p)
{
    mykernelParamType *param = (mykernelParamType *)p;
    executeConvAlgos(param);
}

unsigned int getAlgos(int n, int c, int h, int w, int k, int r, int s)
{
    if (n == 64 && c == 256 && h == 14 && w == 14 && k == 256 && r == 3 && s == 3)
    {
        // return IM2COL_GEMM_1BATCH;
        return MMA_NAIVE;
    }
    else if (n == 256 && c == 192 && h == 14 && w == 14 && k == 192 && r == 3 && s == 3)
    {
        return IM2COL_GEMM_1BATCH_64;
        // return IMPL_GEMM;
        // return IM2COL_GEMM_COMMON;
        // return MMA_NAIVE;
    }
    else if (n == 16 && c == 256 && h == 26 && w == 26 && k == 512 && r == 3 && s == 3)
    {
        // return IM2COL_GEMM_1BATCH_64;
        // return IMPL_GEMM;
        // return IM2COL_GEMM_COMMON;
        return MMA_NAIVE;
    }
    else if (n == 32 && c == 256 && h == 14 && w == 14 && k == 256 && r == 3 && s == 3)
    {
        // return IM2COL_GEMM_1BATCH;
        // return IMPL_GEMM;
        return MMA_NAIVE;
    }
    else if (n == 2 && c == 1280 && h == 16 && w == 16 && k == 1280 && r == 3 && s == 3)
    {
        // return IM2COL_GEMM_1BATCH;
        // return IMPL_GEMM;
        return MMA_NAIVE;
    }
    else if (n == 2 && c == 960 && h == 64 && w == 64 && k == 32 && r == 3 && s == 3)
    {
        return MMA_NAIVE;
    }
    else
    {
        return IMPL_GEMM;
    }
}
