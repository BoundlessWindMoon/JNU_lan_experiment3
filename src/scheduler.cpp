#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include "conv2d.h"
#include <assert.h>


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

void umimplement_init(problem_t* problem) {
    printf("unimplement init\n");
    assert(0);
}

void umimplement_run(mykernelParamType* param) {
    printf("unimplement run\n");
    assert(0);
}

void umimplement_exit(problem_t* problem) {
    printf("unimplement exit\n");
    assert(0);
}

convPlanType conv_plans[12] = {
    {"preliminary_1", umimplement_init, umimplement_run, umimplement_exit},
    {"preliminary_2", umimplement_init, umimplement_run, umimplement_exit},
    {"preliminary_3", umimplement_init, umimplement_run, umimplement_exit},
    {"preliminary_4", umimplement_init, umimplement_run, umimplement_exit},
    {"preliminary_5", umimplement_init, umimplement_run, umimplement_exit},
    {"preliminary_6", umimplement_init, umimplement_run, umimplement_exit},
    {"final_1", umimplement_init, umimplement_run, umimplement_exit},
    {"final_2", umimplement_init, umimplement_run, umimplement_exit},
    {"final_3", umimplement_init, umimplement_run, umimplement_exit},
    {"final_4", umimplement_init, umimplement_run, umimplement_exit},
    {"final_5", umimplement_init, umimplement_run, umimplement_exit},
    {"final_6", umimplement_init, umimplement_run, umimplement_exit},
};

convPlanType scheduler(problem_t *problem) {
    convParamType in_param = {problem->n, problem->c, problem->h, problem->w, problem->k, problem->r, problem->s, problem->u, problem->v, problem->p, problem->q};
    if (PARAM_EQUAL(preliminary_1, in_param)) {return conv_plans[0];}
    if (PARAM_EQUAL(preliminary_2, in_param)) {return conv_plans[1];}
    if (PARAM_EQUAL(preliminary_3, in_param)) {return conv_plans[2];}
    if (PARAM_EQUAL(preliminary_4, in_param)) {return conv_plans[3];}
    if (PARAM_EQUAL(preliminary_5, in_param)) {return conv_plans[4];}
    if (PARAM_EQUAL(preliminary_6, in_param)) {return conv_plans[5];}
    if (PARAM_EQUAL(final_1, in_param)) {return conv_plans[6];}
    if (PARAM_EQUAL(final_2, in_param)) {return conv_plans[7];}
    if (PARAM_EQUAL(final_3, in_param)) {return conv_plans[8];}
    if (PARAM_EQUAL(final_4, in_param)) {return conv_plans[9];}
    if (PARAM_EQUAL(final_5, in_param)) {return conv_plans[10];}
    if (PARAM_EQUAL(final_6, in_param)) {return conv_plans[11];}
}

// void executeConvAlgos(mykernelParamType *param)
// {

//     unsigned int n = param->n;
//     unsigned int c = param->c;
//     unsigned int h = param->h;
//     unsigned int w = param->w;
//     unsigned int k = param->k;
//     unsigned int r = param->r;
//     unsigned int s = param->s;
//     unsigned int p = param->p;
//     unsigned int q = param->q;
//     unsigned int u = param->u;
//     unsigned int v = param->v;
//     unsigned int outh = param->Oh;
//     unsigned int outw = param->Ow;
//     unsigned int algo = param->algo;
//     unsigned int M = k;
//     unsigned int N = n * outh * outw;
//     unsigned int K = c * r * s;
//     if (algo == DIRECT_CONV)
//     {
//         dim3 grid((outh * outw + 15) / 16, (k + 15) / 16, n);
//         dim3 block(16, 16, 1);
//         directConvolution<<<grid, block>>>(*param);
//     }
//     else if (algo == IM2COL_GEMM_1BATCH)
//     {
//         launch_im2col_r_1_c_n_kernel(param->pin, n, c, h, w, r, s, p, q, u, v, param->data_col_device);
//         launch_gemm_128x128x8_fp32((__half *)param->pweight, (__half *)param->data_col_device, (__half *)param->output_gemm_device, M, N, K);
//         launch_reshape_kernel(param->output_gemm_device, param->pout, n, k, outh, outw);
//     }
//     else if (algo == IM2COL_GEMM_NBATCH)
//     {
//         // TODO
//     }
//     else if (algo == IMPL_GEMM)
//     {
//         launch_implicit_gemm(outh, outw, k, n, param);
//     }
//     else if (algo == WINOGRAD)
//     {
//         // TODO
//     }
//     else if (algo == IM2COL_GEMM_1BATCH_64)
//     {
//         launch_im2col_r_1_c_n_kernel(param->pin, n, c, h, w, r, s, p, q, u, v, param->data_col_device);
//         launch_gemm_64x64x8_fp32((__half *)param->pweight, (__half *)param->data_col_device, (__half *)param->output_gemm_device, M, N, K);
//         launch_reshape_kernel(param->output_gemm_device, param->pout, n, k, outh, outw);
//     }
//     else if (algo == MMA_NAIVE)
//     {
//         launch_im2col_r_1_c_n_kernel(param->pin, n, c, h, w, r, s, p, q, u, v, param->data_col_device);
//         launch_transpose_kernel(param->pweight, param->pweight_trans, M, K);
//         launch_gemm_32x32x16_fp16(param->pweight_trans, param->data_col_device, param->output_gemm_device, M, N, K);
//         launch_reshape_kernel(param->output_gemm_device, param->pout, n, k, outh, outw);
//     }
// }

// void convolutionForward(void *p)
// {
//     mykernelParamType *param = (mykernelParamType *)p;
//     executeConvAlgos(param);
// }

// unsigned int getAlgos(int n, int c, int h, int w, int k, int r, int s)
// {
//     if (n == 64 && c == 256 && h == 14 && w == 14 && k == 256 && r == 3 && s == 3)
//     {
//         // return IM2COL_GEMM_1BATCH;
//         return MMA_NAIVE;
//     }
//     else if (n == 256 && c == 192 && h == 14 && w == 14 && k == 192 && r == 3 && s == 3)
//     {
//         return IM2COL_GEMM_1BATCH_64;
//         // return IMPL_GEMM;
//         // return IM2COL_GEMM_COMMON;
//         // return MMA_NAIVE;
//     }
//     else if (n == 16 && c == 256 && h == 26 && w == 26 && k == 512 && r == 3 && s == 3)
//     {
//         // return IM2COL_GEMM_1BATCH_64;
//         // return IMPL_GEMM;
//         // return IM2COL_GEMM_COMMON;
//         return MMA_NAIVE;
//     }
//     else if (n == 32 && c == 256 && h == 14 && w == 14 && k == 256 && r == 3 && s == 3)
//     {
//         // return IM2COL_GEMM_1BATCH;
//         // return IMPL_GEMM;
//         return MMA_NAIVE;
//     }
//     else if (n == 2 && c == 1280 && h == 16 && w == 16 && k == 1280 && r == 3 && s == 3)
//     {
//         // return IM2COL_GEMM_1BATCH;
//         // return IMPL_GEMM;
//         return MMA_NAIVE;
//     }
//     else if (n == 2 && c == 960 && h == 64 && w == 64 && k == 32 && r == 3 && s == 3)
//     {
//         return MMA_NAIVE;
//     }
//     else
//     {
//         return IMPL_GEMM;
//     }
// }
