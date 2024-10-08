#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>  
#include "conv2d.h"

/*选手自己实现的kernel*/
extern "C" __global__ void directConvolution(mykernelParamType param) __attribute__((amdgpu_flat_work_group_size(1,256)))
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;
    
    if(x >= param.Oh*param.Ow || y >= param.k || z >= param.n)
    {
        return;
    }
    
    
    //当前线程处理的数据点在oh、ow上的坐标
    int posOh = x/param.Ow;
    int posOw = x%param.Ow;
        
    int posh_ori = posOh*param.u - param.p;
    int posw_ori = posOw*param.v - param.q;
    
    float sum = 0.0;

    int inOffset = z*param.c*param.h*param.w + posh_ori*param.w + posw_ori;
    int weiOffset = y*param.c*param.r*param.s;
    int inChannelOffset = param.h*param.w;
    int weightChannelOffset = param.r*param.s;
    
    for(int i = 0; i < param.r; i++)
    {
        for(int j = 0; j < param.s; j++)
        {
            int posh_real = posh_ori + i;
            int posw_real = posw_ori + j;            
            
            if(posh_real>=0 && posw_real>=0 && posw_real<param.w && posh_real<param.h)
            {
                int inOffsetTmp = inOffset;
                int weiOffsetTmp = weiOffset;
                for(int channel = 0; channel<param.c; channel++)
                {
                    sum += (float)(param.pin[inOffsetTmp + i*param.w + j] * param.pweight[weiOffsetTmp + i*param.s + j]);
                    inOffsetTmp += inChannelOffset;
                    weiOffsetTmp += weightChannelOffset;
                }               
            }
        }
    }   

    //计算输出偏移
    int outOffset = z*param.k*param.Oh*param.Ow + y*param.Oh*param.Ow + x;
    param.pout[outOffset] = (_Float16)sum;
}