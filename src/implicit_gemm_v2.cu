#include <cuda_runtime.h>
#include <cuda_fp16.h> 
#include "conv2d.h"

extern "C" __global__ void implicit_gemm_v2(mykernelParamType param)
{
    const int ohow = blockIdx.x*16 + threadIdx.x;  // Oh*Ow维度
    const int k = blockIdx.y*16 + threadIdx.y;       // 输出通道
    const int n = blockIdx.z;                       // 批次索引
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ float sh_input[16][16];
    __shared__ float sh_weight[16][16];

    const int oh = ohow / param.Ow;
    const int ow = ohow % param.Ow;
    
    const int ih_start = oh*param.u - param.p;
    const int iw_start = ow*param.v - param.q;

    float sum = 0.0f;

    const int n_offset = n * param.c * param.h * param.w;
    const int k_offset = k * param.c * param.r * param.s;

    const int crs_total = param.c * param.r * param.s;

    for(int base=0; base<crs_total; base+=16){
        int crs_idx = base + tx;
        sh_weight[ty][tx] = (crs_idx < crs_total) ? 
                          param.pweight[k_offset + crs_idx] : 0.0f;

        const int c = (base + ty) / (param.r*param.s);
        const int r = (base + ty) % (param.r*param.s) / param.s;
        const int s = (base + ty) % (param.r*param.s) % param.s;
        
        const int ih = ih_start + r;
        const int iw = iw_start + s;

        if(ih>=0 && iw>=0 && ih<param.h && iw<param.w){
            const int i_addr = c*param.h*param.w + ih*param.w + iw;
            sh_input[ty][tx] = param.pin[n_offset + i_addr];
        } else {
            sh_input[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for(int i=0; i<16; i++){
            sum += sh_input[i][tx] * sh_weight[ty][i];
        }

        __syncthreads();
    }

    /**** 结果写入 ****/
    if(ohow < param.Oh*param.Ow && k < param.k){
        const int o_addr = n*param.k*param.Oh*param.Ow + 
                         k*param.Oh*param.Ow + ohow;
        param.pout[o_addr] = sum;
    }
}


void launch_implicit_gemm_v2(unsigned int outh, unsigned int outw, unsigned int k, unsigned int n, mykernelParamType* param) {
    int blockx = (param->Oh * param->Ow + 15) / 16 ;
    int blocky = (param->k + 15) / 16;
    int blockz = param->n;
    int threadx = 16;        
    int thready = 16;
    int threadz = 1;
    dim3 block(threadx, thready, threadz);
    dim3 grid(blockx, blocky, blockz);
    implicit_gemm_v2<<<grid, block>>>(*param);
}