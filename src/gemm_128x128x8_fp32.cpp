#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include "conv2d.h"

extern "C" __global__ void gemm_128x128x8_fp32(
    __half *__restrict__ A, __half *__restrict__ B, __half *__restrict__ C,
    const int M, const int N, const int K)
{

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

    float r_comp_a[TM]; //  存储从s_a取出的TM长度的向量
    float r_comp_b[TN]; //  存储从s_b取出的TN长度的向量

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

        s_a[0][load_a_smem_k][load_a_smem_m] = r_load_a_test[0];
        s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a_test[1];
        s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a_test[2];
        s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a_test[3];

        FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);
    }

    __syncthreads();

    for (int bk = 1; bk < (K + BK - 1) / BK; bk++)
    {

        int smem_sel = (bk - 1) & 1; // 当前循环计算需要使用的share memory序号
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
        for (int tk = 0; tk < BK; tk++)
        {
            // 从共享内存取出两个向量

            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

// 计算外积，注意这里的矩阵内存位置不是连续的，不能直接写回C
#pragma unroll
            for (int tm = 0; tm < TM; tm++)
            {
#pragma unroll
                for (int tn = 0; tn < TN; tn++)
                {
                    // r_c[tm][tn] += __half2float(r_comp_a[tm] * r_comp_b[tn]);
                    r_c[tm][tn] += (r_comp_a[tm] * r_comp_b[tn]);
                }
            }
        }

        // 把加载的数据从寄存器中写入共享内存中
        // 这部分的STS指令会等待LDG指令写回后再继续发射执行，所以不能放在计算部分之前
        s_a[smem_next][load_a_smem_k][load_a_smem_m] = r_load_a_test[0];
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
    for (int tk = 0; tk < BK; tk++)
    {
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
        for (int tm = 0; tm < TM; tm++)
        {
#pragma unroll
            for (int tn = 0; tn < TN; tn++)
            {
                // r_c[tm][tn] += __half2float(r_comp_a[tm] * r_comp_b[tn]);
                r_c[tm][tn] += (r_comp_a[tm] * r_comp_b[tn]);
            }
        }
    }

// 把r_c矩阵根据空间变换写回矩阵C
#pragma unroll
    for (int i = 0; i < TM / 2; i++)
    {
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
        HALF2(C[store_c_gmem_addr + 2]) = HALF2(r_c_store[2]);

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

        HALF2(C[store_c_gmem_addr + BN / 2]) = HALF2(r_c_store[4]);
        HALF2(C[store_c_gmem_addr + BN / 2 + 2]) = HALF2(r_c_store[6]);
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
    for (int i = 0; i < TM / 2; i++)
    {
        int store_c_gmem_m = by * BM + ty * TM / 2 + BM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);

        //    if (store_c_gmem_m < M && store_c_gmem_n + 3 < N) {
        // HALF4(C[store_c_gmem_addr]) = HALF4(r_c[i + TM / 2][0]);
        r_c_store[0] = r_c[i + TM / 2][0];
        r_c_store[1] = r_c[i + TM / 2][1];
        r_c_store[2] = r_c[i + TM / 2][2];
        r_c_store[3] = r_c[i + TM / 2][3];
        HALF2(C[store_c_gmem_addr]) = HALF2(r_c_store[0]);
        HALF2(C[store_c_gmem_addr + 2]) = HALF2(r_c_store[2]);

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
        r_c_store[4] = r_c[i + TM / 2][4];
        r_c_store[5] = r_c[i + TM / 2][5];
        r_c_store[6] = r_c[i + TM / 2][6];
        r_c_store[7] = r_c[i + TM / 2][7];
        HALF2(C[store_c_gmem_addr + BN / 2]) = HALF2(r_c_store[4]);
        HALF2(C[store_c_gmem_addr + BN / 2 + 2]) = HALF2(r_c_store[6]);

        // C[store_c_gmem_addr + BN / 2] = r_c[i + TM / 2][4];
        // C[store_c_gmem_addr + BN / 2 + 1] = r_c[i + TM / 2][5];
        // C[store_c_gmem_addr + BN / 2 + 2] = r_c[i + TM / 2][6];
        // C[store_c_gmem_addr + BN / 2 + 3] = r_c[i + TM / 2][7];
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

void launch_gemm_128x128x8_fp32(__half *__restrict__ A, __half *__restrict__ B, __half *__restrict__ C,
                                const int M, const int N, const int K)
{
    unsigned int BM = 128;
    unsigned int BN = 128;
    unsigned int TM = 8;
    unsigned int TN = 8;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(BN / TN, BM / TM);
    gemm_128x128x8_fp32<<<grid, block>>>(A, B, C, M, N, K);
}