#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>  
#include "conv2d.h"


extern "C" __global__ void gemm_32x32x16_fp16(_Float16* __restrict__ A, _Float16* __restrict__ B, _Float16* C,
                                         const int M, const int N, const int K) {
    const int lane_id = threadIdx.x;
    const int K_tiles = K / MMA_K;

    const int warp_row = blockIdx.y * MMA_M;
    const int warp_col = blockIdx.x * MMA_N;
    
    __shared__ _Float16 smem[2 *MMA_K * MMA_M];
    
    RegisterUnion fragA, fragB;
    float4_ fragC00, fragC01, fragC10, fragC11;

    fragC00 = {0, 0, 0, 0};
    fragC01 = {0, 0, 0, 0};
    fragC10 = {0, 0, 0, 0};
    fragC11 = {0, 0, 0, 0};
    
   #pragma unroll
    for (int i = 0; i < K_tiles; i++) {
	uint32_t gmem_offsetA = (i * MMA_K + (lane_id / 4)) * M + warp_row + ((lane_id & 3) << 3);
        uint32_t gmem_offsetB = (i * MMA_K + (lane_id / 4)) * N + warp_col + ((lane_id & 3) << 3);
        uint32_t lds_write_offset = lane_id << 3;
        uint32_t lds_read_A_offset = (lane_id << 3) * sizeof(_Float16);
        uint32_t lds_read_B_offset = lds_read_A_offset + 16 * 32 * sizeof(_Float16);
        
        *((int4 *)(&smem[lds_write_offset])) = *((int4 *)(&A[gmem_offsetA]));
        *((int4 *)(&smem[lds_write_offset + MMA_M * MMA_K])) = *(int4 *)(&B[gmem_offsetB]);
			/*
        smem[lds_write_offset + 0] = A[gmem_offsetA + 0];
        smem[lds_write_offset + 1] = A[gmem_offsetA + 1];
        smem[lds_write_offset + 2] = A[gmem_offsetA + 2];
        smem[lds_write_offset + 3] = A[gmem_offsetA + 3];
        smem[lds_write_offset + 4] = A[gmem_offsetA + 4];
        smem[lds_write_offset + 5] = A[gmem_offsetA + 5];
        smem[lds_write_offset + 6] = A[gmem_offsetA + 6];
        smem[lds_write_offset + 7] = A[gmem_offsetA + 7];

        smem[lds_write_offset + MMA_M * MMA_K + 0] = B[gmem_offsetB + 0];
        smem[lds_write_offset + MMA_M * MMA_K + 1] = B[gmem_offsetB + 1];
        smem[lds_write_offset + MMA_M * MMA_K + 2] = B[gmem_offsetB + 2];
        smem[lds_write_offset + MMA_M * MMA_K + 3] = B[gmem_offsetB + 3];
        smem[lds_write_offset + MMA_M * MMA_K + 4] = B[gmem_offsetB + 4];
        smem[lds_write_offset + MMA_M * MMA_K + 5] = B[gmem_offsetB + 5];
        smem[lds_write_offset + MMA_M * MMA_K + 6] = B[gmem_offsetB + 6];
        smem[lds_write_offset + MMA_M * MMA_K + 7] = B[gmem_offsetB + 7];*/

        asm volatile("s_waitcnt lgkmcnt(0)\n\t");
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragA.vector8), "+v"(lds_read_A_offset));
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragB.vector8), "+v"(lds_read_B_offset));
        asm volatile("s_waitcnt lgkmcnt(0)\n\t");

        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC00), "+v"(fragA.vector_front), "+v"(fragB.vector_front));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC01), "+v"(fragA.vector_rear), "+v"(fragB.vector_front));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC10), "+v"(fragA.vector_front), "+v"(fragB.vector_rear));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(fragC11), "+v"(fragA.vector_rear), "+v"(fragB.vector_rear));
    }

    
    uint32_t output_row = blockIdx.y * MMA_M + (lane_id & 15);
    uint32_t output_col = blockIdx.x * MMA_N + (lane_id >> 4);
    C[N * output_row + output_col] = fragC00.x;
    C[N * output_row + output_col + 4] = fragC00.y;
    C[N * output_row + output_col + 8] = fragC00.z;
    C[N * output_row + output_col + 12] = fragC00.w;

    C[N * (output_row + 16) + output_col] = fragC01.x;
    C[N * (output_row + 16) + output_col + 4] = fragC01.y;
    C[N * (output_row + 16) + output_col + 8] = fragC01.z;
    C[N * (output_row + 16) + output_col + 12] = fragC01.w;

    C[N * output_row + output_col + 16 ] = fragC10.x;
    C[N * output_row + output_col + 16 + 4] = fragC10.y;
    C[N * output_row + output_col + 16 + 8] = fragC10.z;
    C[N * output_row + output_col + 16 + 12] = fragC10.w;

    C[N * (output_row + 16) + output_col + 16] = fragC11.x;
    C[N * (output_row + 16) + output_col + 16 + 4] = fragC11.y;
    C[N * (output_row + 16) + output_col + 16 + 8] = fragC11.z;
    C[N * (output_row + 16) + output_col + 16 + 12] = fragC11.w;
}

void launch_gemm_32x32x16_fp16(mykernelParamType *param,
                              int expand_row, int expand_col) {
    
    _Float16* A = param->pweight_trans;
    _Float16* B = param->data_col_device;
    _Float16* C = param->output_gemm_device;
    int M = param->k;
    int N = expand_col * param->Oh * param->Ow;
    int K = param->c * param->r * param->s;
    int span_B = K * N;
    int span_C = M * N;

    dim3 grid(N / MMA_N, M / MMA_M);
    dim3 block(64);

    for(int i = 0; i < expand_row; i++) 
        gemm_32x32x16_fp16 <<<grid, block>>> (A, 
                                            B + i * span_B, 
                                            C + i * span_C, 
                                            M, N, K);
}