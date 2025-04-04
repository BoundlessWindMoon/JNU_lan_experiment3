#include <cuda_runtime.h>
#include <cuda_fp16.h> 
#include "conv2d.h"

extern "C" __global__ void implicit_gemm_v0(mykernelParamType param)
{ 
    int OhOw = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z;
    if(OhOw >= param.Oh*param.Ow || k >= param.k || n >= param.n)
        return;

    int oh = OhOw / param.Ow;
    int ow = OhOw % param.Ow;
    int input_addr, weight_addr, output_addr;
    float sum = 0.0;
    output_addr = n*(param.k*param.Oh*param.Ow) + k*(param.Oh*param.Ow) + OhOw;

    for(int r = 0; r < param.r; r++) {
        for(int s = 0; s < param.s; s++) {
            int ih = oh * param.u - param.p + r;
            int iw = ow * param.v - param.q + s;

            if (ih >= 0 && ih < param.h && iw >= 0 && iw < param.w) {
                for(int c = 0; c < param.c; c++) {
                    input_addr = n * param.c * param.h * param.w 
                               + c * param.h * param.w 
                               + ih * param.w 
                               + iw;
                    
                    weight_addr = k * param.c * param.r * param.s 
                                + c * param.r * param.s 
                                + r * param.s 
                                + s;

                    sum += param.pin[input_addr] * param.pweight[weight_addr];
                }
            }
        }
    }
    param.pout[output_addr] = sum;
}

void launch_implicit_gemm_v0(unsigned int outh, unsigned int outw, unsigned int k, unsigned int n, mykernelParamType* param) {
    int blockx = (param->Oh * param->Ow + 15) / 16 ;
    int blocky = (param->k + 15) / 16;
    int blockz = param->n;
    int threadx = 16;        
    int thready = 16;
    int threadz = 1;
    dim3 block(threadx, thready, threadz);
    dim3 grid(blockx, blocky, blockz);
    implicit_gemm_v0<<<grid, block>>>(*param);
}