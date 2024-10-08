#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>  
#include "conv2d.h"



extern "C" __global__ void myHgemmV3Aligned(
    __half * __restrict__ A, __half * __restrict__ B, __half * __restrict__ C,
    const int M, const int N, const int K
) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    // 使用双倍的share memory来预取数据，
    // 在计算数据之前加载下一次循环用到的数据（从 global Memory 加载到 Shared Memory)
    // GPU无法乱序执行，必须在计算之前就进行数据的加载
    __shared__ float s_a[2][BK][BM];   
    __shared__ float s_b[2][BK][BN];

    __half r_load_a_test[4];
    __half r_load_b_test[4];

    float r_load_a[4];
    float r_load_b[4];

    float r_comp_a[TM];     //  存储从s_a取出的TM长度的向量
    float r_comp_b[TN];     //  存储从s_b取出的TN长度的向量

    float r_c[TM][TN] = {0.0};
    __half r_c_store[8];

    int load_a_smem_m = tid >> 1;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    // 第一次把数据写进share memory中
    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);

        HALF2(r_load_a_test[0]) = HALF2(A[load_a_gmem_addr]);
        HALF2(r_load_a_test[2]) = HALF2(A[load_a_gmem_addr + 2]);
    
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        /* FLOAT4(r_load_b[0]) = FLOAT4(B[load_b_gmem_addr]); */
      //  if (load_b_gmem_k < K && load_b_gmem_n + 3 < N) {
            // HALF4(r_load_b[0]) = HALF4(B[load_b_gmem_addr]);

        HALF2(r_load_b_test[0]) = HALF2(B[load_b_gmem_addr]);
        HALF2(r_load_b_test[2]) = HALF2(B[load_b_gmem_addr + 2]);
        r_load_b[0] = r_load_b_test[0];
        r_load_b[1] = r_load_b_test[1];
        r_load_b[2] = r_load_b_test[2];
        r_load_b[3] = r_load_b_test[3];

        s_a[0][load_a_smem_k    ][load_a_smem_m] = r_load_a_test[0];
        s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a_test[1];
        s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a_test[2];
        s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a_test[3];

        FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);
    }

    __syncthreads();
    

    for (int bk = 1; bk < (K + BK - 1) / BK ; bk++) {

        int smem_sel = (bk - 1) & 1;   // 当前循环计算需要使用的share memory序号
        int smem_next = bk & 1;
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);


        HALF2(r_load_a_test[0]) = HALF2(A[load_a_gmem_addr]);
        HALF2(r_load_a_test[2]) = HALF2(A[load_a_gmem_addr + 2]);

        HALF2(r_load_b_test[0]) = HALF2(B[load_b_gmem_addr]);
        HALF2(r_load_b_test[2]) = HALF2(B[load_b_gmem_addr + 2]);

        r_load_b[0] = r_load_b_test[0];
        r_load_b[1] = r_load_b_test[1];
        r_load_b[2] = r_load_b_test[2];
        r_load_b[3] = r_load_b_test[3];


        // 还有这里的同步指令不能使用了，我们希望加载与计算能够并行执行

        // 计算预取的数据
        #pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            // 从共享内存取出两个向量

            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);  

            // 计算外积，注意这里的矩阵内存位置不是连续的，不能直接写回C
            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                   // r_c[tm][tn] += __half2float(r_comp_a[tm] * r_comp_b[tn]);
                   r_c[tm][tn] += (r_comp_a[tm] * r_comp_b[tn]);
                }
            }
        }

        // 把加载的数据从寄存器中写入共享内存中
        // 这部分的STS指令会等待LDG指令写回后再继续发射执行，所以不能放在计算部分之前
        s_a[smem_next][load_a_smem_k    ][load_a_smem_m] = r_load_a_test[0];
        s_a[smem_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a_test[1];
        s_a[smem_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a_test[2];
        s_a[smem_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a_test[3];
        FLOAT4(s_b[smem_next][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);
        // s_b[smem_next][load_b_smem_k][load_b_smem_n    ] = r_load_b[0];
        // s_b[smem_next][load_b_smem_k][load_b_smem_n + 1] = r_load_b[1];
        // s_b[smem_next][load_b_smem_k][load_b_smem_n + 2] = r_load_b[2];
        // s_b[smem_next][load_b_smem_k][load_b_smem_n + 3] = r_load_b[3];
        
        __syncthreads();
    }
    
    // 计算最后一次循环
    int smem_sel = ((K + BK - 1) / BK - 1) & 1;  
    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
        // 从共享内存取出两个向量
        // HALF4(r_comp_a[0]) = HALF4(s_a[smem_sel][tk][ty * TM / 2]);
        // HALF4(r_comp_a[4]) = HALF4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
        // HALF4(r_comp_b[0]) = HALF4(s_b[smem_sel][tk][tx * TN / 2]);
        // HALF4(r_comp_b[4]) = HALF4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);
        // for (int l = 0;l < 4;l++){
        //     r_comp_a[l] = s_a[smem_sel][tk][ty * TM / 2 + l];
        // }
        // for (int l = 0;l < 4;l++){
        //     r_comp_a[l + 4] = s_a[smem_sel][tk][ty * TM / 2 + l + BM / 2];
        // }
        // for (int l = 0;l < 4;l++){
        //     r_comp_b[l] = s_b[smem_sel][tk][tx * TN / 2 + l];
        // }
        // for (int l = 0;l < 4;l++){
        //     r_comp_b[l + 4] = s_b[smem_sel][tk][tx * TN / 2 + l + BN / 2];
        // }

        FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2]);
        FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
        FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2]);
        FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

        // 计算外积，注意这里的矩阵内存位置不是连续的，不能直接写回C
        #pragma unroll
        for (int tm = 0; tm < TM; tm++) {
            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                //r_c[tm][tn] += __half2float(r_comp_a[tm] * r_comp_b[tn]);
                r_c[tm][tn] += (r_comp_a[tm] * r_comp_b[tn]);
            }
        }
    }



    // 把r_c矩阵根据空间变换写回矩阵C
    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);

      //  if (store_c_gmem_m < M) {
      //      if (store_c_gmem_n + 3 < N) {
                // HALF4(C[store_c_gmem_addr]) = HALF4(r_c[i][0]);
            r_c_store[0] = r_c[i][0];
            r_c_store[1] = r_c[i][1];
            r_c_store[2] = r_c[i][2];
            r_c_store[3] = r_c[i][3];
            HALF2(C[store_c_gmem_addr]) = HALF2(r_c_store[0]);
            HALF2(C[store_c_gmem_addr+2]) = HALF2(r_c_store[2]);


            /*     C[store_c_gmem_addr] = r_c[i][0];
                C[store_c_gmem_addr + 1] = r_c[i][1];
                C[store_c_gmem_addr + 2] = r_c[i][2];
                C[store_c_gmem_addr + 3] = r_c[i][3]; */

    //            if (store_c_gmem_n + 3 + BN  / 2 < N) {
                    // HALF4(C[store_c_gmem_addr + BN / 2]) = HALF4(r_c[i][4]);
            r_c_store[4] = r_c[i][4];
            r_c_store[5] = r_c[i][5];
            r_c_store[6] = r_c[i][6];
            r_c_store[7] = r_c[i][7];

            HALF2(C[store_c_gmem_addr + BN/2]) = HALF2(r_c_store[4]);
            HALF2(C[store_c_gmem_addr + BN/2 + 2]) = HALF2(r_c_store[6]);
   /*                  C[store_c_gmem_addr + BN / 2    ] = r_c[i][4];
                    C[store_c_gmem_addr + BN / 2 + 1] = r_c[i][5];
                    C[store_c_gmem_addr + BN / 2 + 2] = r_c[i][6];
                    C[store_c_gmem_addr + BN / 2 + 3] = r_c[i][7]; */
   //             }
   //             else if (store_c_gmem_n + BN / 2 < N) {
   //                 for (int k = 0; k < 4; k++) {
   //                     if (store_c_gmem_n + k + BN / 2 < N) {
   //                         C[store_c_gmem_addr + BN / 2 + k] = r_c[i + TM / 2][k];
   //                    }
   //                 }
   //             }
   //         }

   /*         else {
                for (int k = 0; k < 4; k++) {
                    if (store_c_gmem_n + k < N) {
                        C[store_c_gmem_addr + k] = r_c[i][k];
                    }
                }

            }*/
   //     }
        // FLOAT4(C[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        // FLOAT4(C[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }

    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + BM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
 
    //    if (store_c_gmem_m < M && store_c_gmem_n + 3 < N) {
            // HALF4(C[store_c_gmem_addr]) = HALF4(r_c[i + TM / 2][0]);
            r_c_store[0] = r_c[i + TM/2][0];
            r_c_store[1] = r_c[i + TM/2][1];
            r_c_store[2] = r_c[i + TM/2][2];
            r_c_store[3] = r_c[i + TM/2][3];
            HALF2(C[store_c_gmem_addr]) = HALF2(r_c_store[0]);
            HALF2(C[store_c_gmem_addr+2]) = HALF2(r_c_store[2]);

            // C[store_c_gmem_addr] = r_c[i + TM / 2][0];
            // C[store_c_gmem_addr + 1] = r_c[i + TM / 2][1];
            // C[store_c_gmem_addr + 2] = r_c[i + TM / 2][2];
            // C[store_c_gmem_addr + 3] = r_c[i + TM / 2][3];
   //     } 
   /*     else if (store_c_gmem_m < M) {

            for (int k = 0; k < 4; k++) {
                if (store_c_gmem_n + k < N) {
                    C[store_c_gmem_addr + k] = r_c[i + TM / 2][k];
                    break;
                }
            }

        }*/

   //     if (store_c_gmem_m < M && store_c_gmem_n + 3 < N) {
            // HALF4(C[store_c_gmem_addr + BN / 2]) = HALF4(r_c[i + TM / 2][4]);
            r_c_store[4] = r_c[i + TM/2][4];
            r_c_store[5] = r_c[i + TM/2][5];
            r_c_store[6] = r_c[i + TM/2][6];
            r_c_store[7] = r_c[i + TM/2][7];
            HALF2(C[store_c_gmem_addr + BN / 2]) = HALF2(r_c_store[4]);
            HALF2(C[store_c_gmem_addr + BN / 2 +2]) = HALF2(r_c_store[6]);


            //C[store_c_gmem_addr + BN / 2] = r_c[i + TM / 2][4];
            //C[store_c_gmem_addr + BN / 2 + 1] = r_c[i + TM / 2][5];
            //C[store_c_gmem_addr + BN / 2 + 2] = r_c[i + TM / 2][6];
            //C[store_c_gmem_addr + BN / 2 + 3] = r_c[i + TM / 2][7];
  //      } 
  /*      else if (store_c_gmem_m < M) {

            for (int k = 0; k < 4; k++) {
                if (store_c_gmem_n + k < N) {
                    C[store_c_gmem_addr + k + BN / 2] = r_c[i + TM / 2][k + 4];
                    break;
                }
            }

        }*/
    }
}

// extern "C" __global__ void im2col_batch_kernel(const _Float16* data_im, int n, int channels, int height, int width,
//                                     int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w,
//                                     int output_h, int output_w, _Float16* data_col) {
//     data_col[0] = 0.0;
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     int crs = blockIdx.y;  // 展开的卷积核索引，表示 (c * r * s)
//     int c = crs / (kernel_h * kernel_w);  // 计算通道索引
//     int kh = (crs % (kernel_h * kernel_w)) / kernel_w;  // 计算卷积核的行索引
//     int kw = (crs % (kernel_h * kernel_w)) % kernel_w;  // 计算卷积核的列索引

//     if (index < n * output_h * output_w) {
//         int b = index / (output_h * output_w);  // 计算批次索引
//         int oh = (index % (output_h * output_w)) / output_w;  // 计算输出行索引
//         int ow = (index % (output_h * output_w)) % output_w;  // 计算输出列索引

//         // 计算输入矩阵中的行和列位置，考虑padding
//         int im_row = kh - pad_h + oh * stride_h;
//         int im_col = kw - pad_w + ow * stride_w;

//         // 计算最终在 data_col 中的偏移
//         int offset_col = crs * n * output_h * output_w + b * output_h * output_w + oh * output_w + ow;

//         if (im_row >= 0 && im_row < height && im_col >= 0 && im_col < width) {
//             // 从输入矩阵中读取值并赋值到输出矩阵中
//             data_col[offset_col] = data_im[(b * channels + c) * height * width + im_row * width + im_col];
//         } else {
//             // 填充0值
//             data_col[offset_col] = 0.0;
//         }
//     }
// }

// void im2col_batch_hip(const _Float16* data_im_device, int n, int channels, int height, int width,
//                       int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w,
//                       _Float16* data_col_device) {
//     int output_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
//     int output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;

//     // 每个block处理1024个index
//     dim3 blockSize(1024);  // 每个block中的线程数
//     // grid的x维度为处理n * output_h * output_w的大小，y维度为crs
//     dim3 gridSize((n * output_h * output_w + blockSize.x - 1) / blockSize.x, channels * kernel_h * kernel_w);

//     // 启动CUDA核函数
//     hipLaunchKernelGGL(im2col_batch_kernel, gridSize, blockSize, 0, 0,
//     data_im_device, n, channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, output_h, output_w, data_col_device);

// }





/*选手需要返回自定义kernel入参结构体的size*/
int getParamsize(__in__ problem_t* problem, __out__ int* paramSize)
{
    *paramSize = sizeof(mykernelParamType);
    return 0;
}

/*选手需要返回自己优化的kernel的grid信息与kernel函数的指针*/
int getkernelInfo(__in__ problem_t* problem, __out__ kernelInfo_t* kernelInfo, __in_out__ void* param) {

    mykernelParamType* pArgs = (mykernelParamType*)param;
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


void executeConvAlgos(mykernelParamType* param) {

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
    unsigned int N = n*outh*outw;
    unsigned int K = c*r*s;
    if(algo == DIRECT_CONV) {
        dim3 group1((outh*outw + 15) / 16, (k + 15) / 16, n);
        dim3 thread1(16, 16, 1);
        directConvolution<<<group1, thread1>>>(*param);
        //hipExtLaunchKernel((void*)directConvolution,group1,thread1,(void**)param,128,0,0,0,0);
    }else if(algo == IM2COL_GEMM_1BATCH) {
        unsigned int BM = 128;
        unsigned int BN = 128;
        unsigned int TM = 8;
        unsigned int TN = 8;
        dim3 thread1(BN / TN, BM / TM);
        dim3 group1((N + BN - 1) / BN, (M + BM - 1) / BM); 

        launch_im2col_r_1_c_n_kernel(param->pin, n, c, h, w, r, s, p, q, u, v, param->data_col_device);
        myHgemmV3Aligned<<<group1, thread1>>>((__half*)param->pweight,
                                                (__half*)param->data_col_device,
                                                (__half*)param->output_gemm_device,
                                                k, outh * outw * n,
                                                c*r*s);
        launch_reshape_kernel(param->output_gemm_device, param->pout, n, k, outh, outw);

    } else if(algo == IM2COL_GEMM_NBATCH) {

    } else if(algo == IMPL_GEMM) {
        launch_implicit_gemm(outh, outw, k, n, param);
    } else if(algo == WINOGRAD) {

    } else if(algo == IM2COL_GEMM_COMMON) {
        // launch_im2col_r_1_c_n_kernel(param->pin, n, c, h, w, r, s, p, q, u, v, param->data_col_device);
        // const int TM = 4;
        // const int TN = 4;
        // const int TK = 2;
        // const int padding = 4;
        // // int thread_size = (m * n + reg_size * reg_size - 1) / (reg_size * reg_size);
        // int thread_size = min(64 * 64, M*N*K);
        // dim3 block((M + 64 - 1) / 64, (N + 64 - 1) / 64);
        // dim3 thread((64 * 64 + TM * TN - 1) / (TM * TN));
        // int shared_size = sizeof(float) * (64 * (16 + padding) + 16 * (64 + padding));
        // gemm_kernel8_4<<<block, thread, shared_size>>>((__half*)param->pweight, (__half*)param->data_col_device, (__half*)param->output_gemm_device, M, N, K);

        // launch_reshape_kernel(param->output_gemm_device, param->pout, n, k, outh, outw);
    } else if(algo == IM2COL_GEMM_1BATCH_64) {

        launch_im2col_r_1_c_n_kernel(param->pin, n, c, h, w, r, s, p, q, u, v, param->data_col_device);

        // GEMM_64x64x8_v3<<<group1, thread1>>>((__half *)param->pweight,
        //                                         (__half *)param->data_col_device,
        //                                         (__half *)param->output_gemm_device,
        //                                         k, outh * outw * n,
        //                                         c*r*s);
        launch_gemm_64x64x8_fp32((__half *)param->pweight, (__half *)param->data_col_device, (__half *)param->output_gemm_device, M, N, K);

        launch_reshape_kernel(param->output_gemm_device, param->pout, n, k, outh, outw);
    } else if(algo == MMA_NAIVE) {
        launch_im2col_r_1_c_n_kernel(param->pin, n, c, h, w, r, s, p, q, u, v, param->data_col_device);
        launch_transpose_kernel(param->pweight, param->pweight_trans, M, K);
        launch_gemm_32x32x16_fp16(param->pweight_trans, param->data_col_device, param->output_gemm_device, M, N, K);
        launch_reshape_kernel(param->output_gemm_device, param->pout, n, k, outh, outw);        
    }
}

void convolutionForward(void* p) {
    mykernelParamType* param = (mykernelParamType*)p;
    executeConvAlgos(param);
}

unsigned int getAlgos(int n, int c, int h, int w, int k, int r, int s) {
    if(n == 64 && c == 256 && h == 14 && w == 14 && k == 256 && r == 3 && s == 3) {
            //return IM2COL_GEMM_1BATCH;
            return MMA_NAIVE;
        } else if (n == 256 && c == 192 && h == 14 && w == 14 && k == 192 && r == 3 && s == 3) {
            return IM2COL_GEMM_1BATCH_64;
            //return IMPL_GEMM;
            //return IM2COL_GEMM_COMMON;
            //return MMA_NAIVE;
        } else if(n == 16 && c == 256 && h == 26 && w == 26 && k == 512 && r == 3 && s == 3) {
            //return IM2COL_GEMM_1BATCH_64;
            //return IMPL_GEMM;
            //return IM2COL_GEMM_COMMON;
            return MMA_NAIVE;
        } else if(n == 32 && c == 256 && h == 14 && w == 14 && k == 256 && r == 3 && s == 3) {
            //return IM2COL_GEMM_1BATCH;
            //return IMPL_GEMM;
            return MMA_NAIVE;
        } else if(n == 2 && c == 1280 && h ==16 && w == 16 && k == 1280 && r == 3 && s == 3) {
            //return IM2COL_GEMM_1BATCH;
            //return IMPL_GEMM;
            return MMA_NAIVE;
        } else if(n == 2 && c == 960 && h == 64 && w == 64 && k == 32 && r == 3 && s == 3) {
            return MMA_NAIVE;
        } else {
            return IMPL_GEMM;
        }
}
