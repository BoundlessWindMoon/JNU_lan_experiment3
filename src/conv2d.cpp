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
    // unsigned int algo = problem->algo;
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
    // pArgs->algo = algo;
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

