#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>  
#include "conv2d.h"

extern "C" __global__ void gemm_64x64x8_fp32(
    __half * __restrict__ A, __half * __restrict__ B, __half * __restrict__ C,
    const int M, const int N, const int K) {
    const int BM = 64;
    const int BN = 64;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[2][BK][BM];   
    __shared__ float s_b[2][BK][BN];

    float r_load_a[8];
    float r_load_b[8];
    float r_comp_a[TM];    
    float r_comp_b[TN];     
    __half r_load_a_tmp[8];
    __half r_load_b_tmp[8];

    float r_c[TM][TN] = {0.0};
    __half r_c_store[8];

    int load_a_smem_m = tid;           
    int load_a_smem_k = 0;     
    int load_b_smem_k = tid >> 3;           
    int load_b_smem_n = (tid & 7) << 3;    

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    // 第一次把数据写进share memory中
    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);

        HALF2(r_load_a_tmp[0]) = HALF2(A[load_a_gmem_addr]);
        HALF2(r_load_a_tmp[2]) = HALF2(A[load_a_gmem_addr + 2]);
        HALF2(r_load_a_tmp[4]) = HALF2(A[load_a_gmem_addr + 4]);
        HALF2(r_load_a_tmp[6]) = HALF2(A[load_a_gmem_addr + 6]);

        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);

        HALF2(r_load_b_tmp[0]) = HALF2(B[load_b_gmem_addr]);
        HALF2(r_load_b_tmp[2]) = HALF2(B[load_b_gmem_addr + 2]);
        HALF2(r_load_b_tmp[4]) = HALF2(B[load_b_gmem_addr + 4]);
        HALF2(r_load_b_tmp[6]) = HALF2(B[load_b_gmem_addr + 6]);
        r_load_b[0] = r_load_b_tmp[0];
        r_load_b[1] = r_load_b_tmp[1];
        r_load_b[2] = r_load_b_tmp[2];
        r_load_b[3] = r_load_b_tmp[3];
        r_load_b[4] = r_load_b_tmp[4];
        r_load_b[5] = r_load_b_tmp[5];
        r_load_b[6] = r_load_b_tmp[6];
        r_load_b[7] = r_load_b_tmp[7];

        s_a[0][0][load_a_smem_m] = r_load_a_tmp[0];
        s_a[0][1][load_a_smem_m] = r_load_a_tmp[1];
        s_a[0][2][load_a_smem_m] = r_load_a_tmp[2];
        s_a[0][3][load_a_smem_m] = r_load_a_tmp[3];
        s_a[0][4][load_a_smem_m] = r_load_a_tmp[4];
        s_a[0][5][load_a_smem_m] = r_load_a_tmp[5];
        s_a[0][6][load_a_smem_m] = r_load_a_tmp[6];
        s_a[0][7][load_a_smem_m] = r_load_a_tmp[7];

        FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);
        FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n + 4]) = FLOAT4(r_load_b[4]);
    }

    __syncthreads();


    for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
        int smem_sel = (bk - 1) & 1;   // 当前循环计算需要使用的share memory序号
        int smem_next = bk & 1;

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);

        HALF2(r_load_a_tmp[0]) = HALF2(A[load_a_gmem_addr]);
        HALF2(r_load_a_tmp[2]) = HALF2(A[load_a_gmem_addr + 2]);
        HALF2(r_load_a_tmp[4]) = HALF2(A[load_a_gmem_addr + 4]);
        HALF2(r_load_a_tmp[6]) = HALF2(A[load_a_gmem_addr + 6]);

        HALF2(r_load_b_tmp[0]) = HALF2(B[load_b_gmem_addr]);
        HALF2(r_load_b_tmp[2]) = HALF2(B[load_b_gmem_addr + 2]);
        HALF2(r_load_b_tmp[4]) = HALF2(B[load_b_gmem_addr + 4]);
        HALF2(r_load_b_tmp[6]) = HALF2(B[load_b_gmem_addr + 6]);
        r_load_b[0] = r_load_b_tmp[0];
        r_load_b[1] = r_load_b_tmp[1];
        r_load_b[2] = r_load_b_tmp[2];
        r_load_b[3] = r_load_b_tmp[3];
        r_load_b[4] = r_load_b_tmp[4];
        r_load_b[5] = r_load_b_tmp[5];
        r_load_b[6] = r_load_b_tmp[6];
        r_load_b[7] = r_load_b_tmp[7];
           
       for(int tk = 0; tk < BK; tk++) {
          FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + 0]);
          FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2 + 0]);

          FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + 0]);
          FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2 + 0]);

          // 计算外积，注意这里的矩阵内存位置不是连续的，不能直接写回C
          #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                }
            }
       }
       s_a[smem_next][0][load_a_smem_m] = r_load_a_tmp[0];
       s_a[smem_next][1][load_a_smem_m] = r_load_a_tmp[1];
       s_a[smem_next][2][load_a_smem_m] = r_load_a_tmp[2];
       s_a[smem_next][3][load_a_smem_m] = r_load_a_tmp[3];
       s_a[smem_next][4][load_a_smem_m] = r_load_a_tmp[4];
       s_a[smem_next][5][load_a_smem_m] = r_load_a_tmp[5];
       s_a[smem_next][6][load_a_smem_m] = r_load_a_tmp[6];
       s_a[smem_next][7][load_a_smem_m] = r_load_a_tmp[7];

       FLOAT4(s_b[smem_next][load_b_smem_k][load_b_smem_n + 0]) = FLOAT4(r_load_b[0]);
       FLOAT4(s_b[smem_next][load_b_smem_k][load_b_smem_n + 4]) = FLOAT4(r_load_b[4]);
        __syncthreads();
    }

    int smem_sel = ((K + BK - 1) / BK - 1) & 1;
    #pragma unroll
    for(int tk = 0; tk < BK; tk++) {
        
        FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + 0]);
        FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2 + 0]);

        FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + 0]);
        FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2 + 0]);

         // 计算外积，注意这里的矩阵内存位置不是连续的，不能直接写回C
        #pragma unroll
        for (int tm = 0; tm < TM; tm++) {
            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
            }
        }
    }


    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);

        r_c_store[0] = r_c[i][0];
        r_c_store[1] = r_c[i][1];
        r_c_store[2] = r_c[i][2];
        r_c_store[3] = r_c[i][3];
        r_c_store[4] = r_c[i][4];
        r_c_store[5] = r_c[i][5];
        r_c_store[6] = r_c[i][6];
        r_c_store[7] = r_c[i][7];

        HALF2(C[store_c_gmem_addr]) = HALF2(r_c_store[0]);
        HALF2(C[store_c_gmem_addr + 2]) = HALF2(r_c_store[2]);
        HALF2(C[store_c_gmem_addr + BN / 2]) = HALF2(r_c_store[4]);
        HALF2(C[store_c_gmem_addr + BN / 2 + 2]) = HALF2(r_c_store[6]);

        /* C[store_c_gmem_addr] = r_c[i][0];
        C[store_c_gmem_addr + 1] = r_c[i][1];
        C[store_c_gmem_addr + 2] = r_c[i][2];
        C[store_c_gmem_addr + 3] = r_c[i][3];
        C[store_c_gmem_addr + BN / 2] = r_c[i][4];
        C[store_c_gmem_addr + BN / 2 + 1] = r_c[i][5];
        C[store_c_gmem_addr + BN / 2 + 2] = r_c[i][6];
        C[store_c_gmem_addr + BN / 2 + 3] = r_c[i][7]; */
    }

    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + BM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);

        r_c_store[0] = r_c[i + TM / 2][0];
        r_c_store[1] = r_c[i + TM / 2][1];
        r_c_store[2] = r_c[i + TM / 2][2];
        r_c_store[3] = r_c[i + TM / 2][3];
        r_c_store[4] = r_c[i + TM / 2][4];
        r_c_store[5] = r_c[i + TM / 2][5];
        r_c_store[6] = r_c[i + TM / 2][6];
        r_c_store[7] = r_c[i + TM / 2][7];

        HALF2(C[store_c_gmem_addr]) = HALF2(r_c_store[0]);
        HALF2(C[store_c_gmem_addr + 2]) = HALF2(r_c_store[2]);
        HALF2(C[store_c_gmem_addr + BN / 2]) = HALF2(r_c_store[4]);
        HALF2(C[store_c_gmem_addr + BN / 2 + 2]) = HALF2(r_c_store[6]);

        /* C[store_c_gmem_addr] = r_c[i + TM / 2][0];
        C[store_c_gmem_addr + 1] = r_c[i + TM / 2][1];
        C[store_c_gmem_addr + 2] = r_c[i + TM / 2][2];
        C[store_c_gmem_addr + 3] = r_c[i + TM / 2][3];
        C[store_c_gmem_addr + BN / 2] = r_c[i + TM / 2][4];
        C[store_c_gmem_addr + BN / 2 + 1] = r_c[i + TM / 2][5];
        C[store_c_gmem_addr + BN / 2 + 2] = r_c[i + TM / 2][6];
        C[store_c_gmem_addr + BN / 2 + 3] = r_c[i + TM / 2][7]; */
    }
}

void launch_gemm_64x64x8_fp32(  __half * __restrict__ A, __half * __restrict__ B, __half * __restrict__ C,
    const int M, const int N, const int K) {
    unsigned int BM = 64;
    unsigned int BN = 64;
    unsigned int TM = 8;
    unsigned int TN = 8;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM); 
    dim3 block(BN / TN, BM / TM);
    gemm_64x64x8_fp32<<<grid, block>>>(A, B, C, M, N, K);
}