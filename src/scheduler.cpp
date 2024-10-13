#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include "conv2d.h"
#include <assert.h>
#include <stdio.h>

convParamType preliminary_1 = {64, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1};
convParamType preliminary_2 = {256, 192, 14, 14, 192, 3, 3, 1, 1, 1, 1};
convParamType preliminary_3 = {16, 256, 26, 26, 512, 3, 3, 1, 1, 1, 1};
convParamType preliminary_4 = {32, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1};
convParamType preliminary_5 = {2, 1280, 16, 16, 1280, 3, 3, 1, 1, 1, 1};
convParamType preliminary_6 = {2, 960, 64, 64, 32, 3, 3, 1, 1, 1, 1};

convParamType final_1= {16, 128, 64, 64, 27, 3, 3, 1, 1, 1, 1};
convParamType final_2= {16, 256, 32, 32, 256, 3, 3, 1, 1, 1, 1};
convParamType final_3= {16, 64, 128, 128, 64, 3, 3, 1, 1, 1, 1};
convParamType final_4= {2, 1920, 32, 32, 640, 3, 3, 1, 1, 1, 1};
convParamType final_5= {2, 640, 64, 64, 640, 3, 3, 1, 1, 1, 1};
convParamType final_6= {2, 320, 64, 64, 4, 3, 3, 1, 1, 1, 1};

#define UNROLL_PARAM(param) unsigned int n = param->n; \
                            unsigned int c = param->c; \
                            unsigned int h = param->h; \
                            unsigned int w = param->w; \
                            unsigned int k = param->k; \
                            unsigned int r = param->r; \
                            unsigned int s = param->s; \
                            unsigned int p = param->p; \
                            unsigned int q = param->q; \
                            unsigned int u = param->u; \
                            unsigned int v = param->v; \
                            unsigned int outh = param->Oh; \
                            unsigned int outw = param->Ow;

#define UNROLL_PROBLEM(problem) unsigned int n = problem->n; \
                            unsigned int c = problem->c; \
                            unsigned int h = problem->h; \
                            unsigned int w = problem->w; \
                            unsigned int k = problem->k; \
                            unsigned int r = problem->r; \
                            unsigned int s = problem->s; \
                            unsigned int u = problem->u; \
                            unsigned int v = problem->v; \
                            unsigned int p = problem->p; \
                            unsigned int q = problem->q; 

void im2col_log(problem_t *problem) {
    UNROLL_PROBLEM(problem);
    unsigned int outh = (h - r + 2 * p) / u + 1;
    unsigned int outw = (w - s + 2 * q) / v + 1;
    printf("weight: %d x %d, feature: %d x %d x %d\n", k, c*r*s, n, c*r*s, outh*outw);
}



void umimplement_init(mykernelParamType* param) {
    printf("unimplement init\n");
    assert(0);
}

void umimplement_run(mykernelParamType* param) {
    printf("unimplement run\n");
    assert(0);
}

void umimplement_exit(mykernelParamType* param) {
    printf("unimplement exit\n");
    assert(0);
}

void im2col_gemm_1batch_init(mykernelParamType * param) {
    UNROLL_PARAM(param);
    _Float16 *data_col_device, *output_gemm_device, *pWeight_trans;
    hipMalloc((void **)&data_col_device, n * c * r * s * outh * outw * sizeof(_Float16));
    hipMalloc((void **)&output_gemm_device, n * k * outh * outw * sizeof(_Float16));
    param->data_col_device = data_col_device;
    param->output_gemm_device = output_gemm_device;
}


void mma_naive_init(mykernelParamType * param) {
    UNROLL_PARAM(param);
    _Float16 *data_col_device, *output_gemm_device, *pWeight_trans;
    hipMalloc((void **)&data_col_device, n * c * r * s * outh * outw * sizeof(_Float16));
    hipMalloc((void **)&output_gemm_device, n * k * outh * outw * sizeof(_Float16));
    hipMalloc((void **)&pWeight_trans, k * c * r * s * sizeof(_Float16));
    param->data_col_device = data_col_device;
    param->output_gemm_device = output_gemm_device;
    param->pweight_trans = pWeight_trans;
}

void directConvolution_run(mykernelParamType * param) {
    // TODO
    assert(0);
}

void im2col_gemm_1batch_run(mykernelParamType * param) {
    // UNROLL_PARAM(param);
    // unsigned int M = k;
    // unsigned int N = n * outh * outw;
    // unsigned int K = c * r * s;
    // launch_im2col_r_1_c_n_kernel(param->pin, n, c, h, w, r, s, p, q, u, v, param->data_col_device);
    // launch_gemm_128x128x8_fp32((__half *)param->pweight, (__half *)param->data_col_device, (__half *)param->output_gemm_device, M, N, K);
    // launch_reshape_kernel(param->output_gemm_device, param->pout, n, k, outh, outw);
}

void im2col_gemm_nbatch_run(mykernelParamType * param) {
    // TODO
    assert(0);
}

void impl_gemm_1batch_run(mykernelParamType * param) {
    UNROLL_PARAM(param);
    launch_implicit_gemm(outh, outw, k, n, param);
}

void winograd_run(mykernelParamType * param) {
    // TODO
    assert(0);
}

void im2col_gemm_1batch_64_run(mykernelParamType * param) {
    // UNROLL_PARAM(param);
    // unsigned int M = k;
    // unsigned int N = n * outh * outw;
    // unsigned int K = c * r * s;
    // launch_im2col_r_1_c_n_kernel(param->pin, n, c, h, w, r, s, p, q, u, v, param->data_col_device);
    // launch_gemm_64x64x8_fp32((__half *)param->pweight, (__half *)param->data_col_device, (__half *)param->output_gemm_device, M, N, K);
    // launch_reshape_kernel(param->output_gemm_device, param->pout, n, k, outh, outw);
}

void mma_naive_run(mykernelParamType * param) {
    UNROLL_PARAM(param);
    unsigned int M = k;
    unsigned int N = n * outh * outw;
    unsigned int K = c * r * s;
    // launch_im2col_r_1_c_n_kernel(param->pin, n, c, h, w, r, s, p, q, u, v, param->data_col_device);
    // launch_transpose_kernel(param->pweight, param->pweight_trans, M, K);
    // launch_gemm_32x32x16_fp16(param->pweight_trans, param->data_col_device, param->output_gemm_device, M, N, K);
    // launch_reshape_kernel(param->output_gemm_device, param->pout, n, k, outh, outw);

    launch_im2col_2D_kernel(param->pin, n, c, h, w, r, s, h, w, param->expand_row, param->expand_col, param->data_col_device);
       // launch_im2col_r_1_c_n_kernel(param->pin, n, c, h, w, r, s, p, q, u, v, param->data_col_device);
    launch_transpose_kernel(param->pweight, param->pweight_trans, M, K);
    launch_gemm_32x32x16_fp16(param, param->expand_row, param->expand_col);
        //launch_reshape_kernel(param->output_gemm_device, param->pout, n, k, outh, outw);
    launch_reshape_2D_kernel(param->output_gemm_device, param->pout, n, k, outh, outw, param->expand_row, param->expand_col);
}


void im2col_gemm_1batch_exit(mykernelParamType * param) {
    hipFree(param->data_col_device);
    hipFree(param->output_gemm_device);
}


void mma_naive_exit(mykernelParamType * param) {
    hipFree(param->data_col_device);
    hipFree(param->output_gemm_device);
    hipFree(param->pweight_trans);
}

convPlanType conv_plans[13] = {
    {"preliminary_1", mma_naive_init, mma_naive_run, mma_naive_exit},
    {"preliminary_2", mma_naive_init, mma_naive_run, mma_naive_exit},
    //{"preliminary_2", im2col_gemm_1batch_init, im2col_gemm_1batch_64_run, im2col_gemm_1batch_exit},
    {"preliminary_3", mma_naive_init, mma_naive_run, mma_naive_exit},
    {"preliminary_4", mma_naive_init, mma_naive_run, mma_naive_exit},
    {"preliminary_5", mma_naive_init, mma_naive_run, mma_naive_exit},
    {"preliminary_6", mma_naive_init, mma_naive_run, mma_naive_exit},
    {"final_1", mma_naive_init, mma_naive_run, mma_naive_exit},
    {"final_2", mma_naive_init, mma_naive_run, mma_naive_exit},
    {"final_3", mma_naive_init, mma_naive_run, mma_naive_exit},
    {"final_4", mma_naive_init, mma_naive_run, mma_naive_exit},
    {"final_5", mma_naive_init, mma_naive_run, mma_naive_exit},
    {"final_6", mma_naive_init, mma_naive_run, mma_naive_exit},
    {"umimplement", umimplement_init, umimplement_run, umimplement_exit},
};

convPlanType scheduler(problem_t *problem, mykernelParamType * param) {
    im2col_log(problem);
    convParamType in_param = {problem->n, problem->c, problem->h, problem->w, problem->k, problem->r, problem->s, problem->u, problem->v, problem->p, problem->q};
    if (PARAM_EQUAL(preliminary_1, in_param)) {
        param->expand_row = 2;
        param->expand_col = param->n / param->expand_row;
        return conv_plans[0];
    }
    if (PARAM_EQUAL(preliminary_2, in_param)) {
        param->expand_row = 4;
        param->expand_col = param->n / param->expand_row;
        return conv_plans[1];
    }
    if (PARAM_EQUAL(preliminary_3, in_param)) {
        param->expand_row = 1;
        param->expand_col = param->n / param->expand_row;
        return conv_plans[2];
    }
    if (PARAM_EQUAL(preliminary_4, in_param)) {
        param->expand_row = 1;
        param->expand_col = param->n / param->expand_row;
        return conv_plans[3];
    }
    if (PARAM_EQUAL(preliminary_5, in_param)) {
        param->expand_row = 1;
        param->expand_col = param->n / param->expand_row;
        return conv_plans[4];
    }
    if (PARAM_EQUAL(preliminary_6, in_param)) {
        param->expand_row = 1;
        param->expand_col = param->n / param->expand_row;
        return conv_plans[5];
    }
    if (PARAM_EQUAL(final_1, in_param)) {
        param->expand_row = 1;
        param->expand_col = param->n / param->expand_row;
        return conv_plans[6];
    }
    if (PARAM_EQUAL(final_2, in_param)) {
        param->expand_row = 2;
        param->expand_col = param->n / param->expand_row;
        return conv_plans[7];
    }
    if (PARAM_EQUAL(final_3, in_param)) {
        param->expand_row = 16;
        param->expand_col = param->n / param->expand_row;
        return conv_plans[8];
    }
    if (PARAM_EQUAL(final_4, in_param)) {
        param->expand_row = 1;
        param->expand_col = param->n / param->expand_row;
        return conv_plans[9];
    }
    if (PARAM_EQUAL(final_5, in_param)) {
        param->expand_row = 1;
        param->expand_col = param->n / param->expand_row;
        return conv_plans[10];
    }
    if (PARAM_EQUAL(final_6, in_param)) {
        param->expand_row = 1;
        param->expand_col = param->n / param->expand_row;
        return conv_plans[11];
    }
    return conv_plans[12];
}
