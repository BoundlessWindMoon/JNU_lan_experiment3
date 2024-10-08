#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>  
#include "conv2d.h"

extern "C" __global__ void implicit_gemm(mykernelParamType param)
{ 
    uint32_t tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // warp tile, z字排布
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t mma_tid_x = (lane_id / 2) % 8;
    const uint32_t mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);

    // 每个线程需要负责一个 8 * 8 的矩阵， 实际上这里划分为 4个 4 * 4 的矩阵
    uint32_t input_lds_addr = (warp_id % 2) * (8 * 8) + mma_tid_x * 4 ;
    uint32_t weight_lds_addr = (warp_id / 2) * (8 * 4) + mma_tid_y * 4;
    int y = weight_lds_addr + by * 128;
    int x = input_lds_addr + bx * 128;

    // share memory buffer, 每个线程需要负责加载 4 * 2 数据
    __shared__ float shm_weight[2][8 * 132];    // 列主序 shm_weight[4][32][8]
    __shared__ float shm_input[2][128 * 8];   // 行主序 shm_input[8][4][32]

    uint32_t weight_sts_addr = (tx % 8) * 132 + (tx / 8) * 4 ;  // shm_weight[:4][tx / 8][tx % 8]
    uint32_t input_sts_addr = (tx / 32) * 128 + (tx % 32);  // shm_input[tx / 32][：4][tx % 32]
    
    // 当前线程加载的数据点在输入矩阵 Oh 和 Ow 上的坐标, 注意和上面的矩阵的对应关系
    int pos_ori_h[4];
    int pos_ori_w[4];
    # pragma unroll
    for (int i = 0; i < 4; i++) {
        pos_ori_h[i] = ((bx * 128 + tx % 32 + i * 32) / param.Ow) * param.u - param.p;
        pos_ori_w[i] = ((bx * 128 + tx % 32 + i * 32) % param.Ow) * param.v - param.q;
    }

    // 计算对应加载数据所在矩阵的偏移
    int z = blockIdx.z;
    int input_offset = z * param.h * param.w * param.c;
    int weight_offset = (by * 128 + tx / 8 * 4) * param.c * param.r * param.s;
    int input_channel_size = param.h * param.w;
    int weight_channel_size = param.r * param.s;
    int kernel_size = param.c * weight_channel_size;


    // 初始化 输出矩阵 , 中间矩阵
    int write_flag = 1;
    float weight_temp[2][8];
    float input_temp[2][8];
    float output_temp[8][8];
 
    // float weight_temp[8];
    // float input_temp[8];
    // float output_temp[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
    #pragma unroll
        for (int j = 0; j < 8; j++) {
            output_temp[i][j] = 0.0;
        }
    }

    float weight_ldg_reg[4];
    float input_ldg_reg[4];

    int crs = 0;
    int weight_offset_tmp = crs + tx % 8;
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        if ( weight_offset_tmp < kernel_size && by * 128 + tx / 8 * 4 + i < param.k ) {
            weight_ldg_reg[i] = param.pweight[weight_offset + weight_offset_tmp + i * kernel_size];
        } 
        else {
            weight_ldg_reg[i] = 0.0;
        }   
        // weight_ldg_reg[i] = param.pweight[weight_offset + weight_offset_tmp + i * kernel_size]; // 不清楚为什么不判断越界也可以
    }

    int cur_c = (crs + tx / 32) / weight_channel_size;
    int cur_ih = ((crs + tx / 32) % weight_channel_size) / param.s;
    int cur_iw = ((crs + tx / 32) % weight_channel_size) % param.s;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int cur_h = pos_ori_h[i] + cur_ih;
        int cur_w = pos_ori_w[i] + cur_iw;
        int input_offset_tmp = cur_c * input_channel_size + cur_h * param.w + cur_w;

        if (cur_h >= 0 && cur_w >= 0 && cur_h < param.h && cur_w < param.w) {
            // shm_input[input_sts_addr + i * 32] = param.pin[input_offset_tmp + input_offset];
            input_ldg_reg[i] = param.pin[input_offset_tmp + input_offset];
        }
        else {
            input_ldg_reg[i] = 0.0;
        }
    }

    // sts
    for (int i = 0; i < 4; i++){
        shm_input[0][input_sts_addr + i * 32] = input_ldg_reg[i];
        shm_weight[0][weight_sts_addr + i] = weight_ldg_reg[i];
    }
    __syncthreads();

    //lds stage , for subcrs = 0
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        input_temp[0][i] = shm_input[0][input_lds_addr + i];
        input_temp[0][i + 4] = shm_input[0][input_lds_addr + i + 32];
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        weight_temp[0][i] = shm_weight[0][weight_lds_addr + i];
        weight_temp[0][i + 4] = shm_weight[0][weight_lds_addr + i + 16];
    }
    
    // 主循环，注意每个循环内 负责一个 CRS tile 的计算，以及下一个循环需要的数据ldf + sts + (下个循环第一个lds使用的)
    // 
    // main loop
    for (crs = 0; crs < kernel_size; crs += 8) {
        // 加载数据 
        // ldg stage
        int weight_offset_tmp = crs + 8 + tx % 8;
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            if ( weight_offset_tmp < kernel_size && by * 128 + tx / 8 * 4 + i < param.k ) {
                weight_ldg_reg[i] = param.pweight[weight_offset + weight_offset_tmp + i * kernel_size];
            } 
            else {
                weight_ldg_reg[i] = 0.0;
            }   
            // weight_ldg_reg[i] = param.pweight[weight_offset + weight_offset_tmp + i * kernel_size]; // 不清楚为什么不判断越界也可以
        }

        int cur_c = (crs + 8 + tx / 32) / weight_channel_size;
        int cur_ih = ((crs + 8 + tx / 32) % weight_channel_size) / param.s;
        int cur_iw = ((crs + 8 + tx / 32) % weight_channel_size) % param.s;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int cur_h = pos_ori_h[i] + cur_ih;
            int cur_w = pos_ori_w[i] + cur_iw;
            int input_offset_tmp = cur_c * input_channel_size + cur_h * param.w + cur_w;

            if (cur_h >= 0 && cur_w >= 0 && cur_h < param.h && cur_w < param.w) {
                // shm_input[input_sts_addr + i * 32] = param.pin[input_offset_tmp + input_offset];
                input_ldg_reg[i] = param.pin[input_offset_tmp + input_offset];
            }
            else {
                input_ldg_reg[i] = 0.0;
            }
        }

        int load_flag = write_flag ^ 1; // 对应这个循环计算使用的数据标志位
        #pragma unroll
        for (int subcrs = 0; subcrs < 8 - 1; subcrs++) {
            // lds下个循环使用的数据
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                weight_temp[(subcrs + 1) % 2][i] = shm_weight[load_flag][weight_lds_addr + (subcrs + 1) * 132 + i]; 
                weight_temp[(subcrs + 1) % 2][i + 4] = shm_weight[load_flag][weight_lds_addr + (subcrs + 1) * 132 + i + 16]; 
            }

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                input_temp[(subcrs + 1) % 2][i] = shm_input[load_flag][input_lds_addr + (subcrs + 1) * 128 + i]; 
                input_temp[(subcrs + 1) % 2][i + 4] = shm_input[load_flag][input_lds_addr + (subcrs + 1) * 128 + i + 32];
            }

            // compute
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    // if (z == 0 && y + i == 0 && ((x + j == 128 && j < 4) || (x + j - 4 == 128 + 32 && j >= 4))) {
                    //     printf("step %d : %.9f += %.9f * %.9f\n",subcrs + crs, output_temp[i][j], input_temp[subcrs % 2][j] , weight_temp[subcrs % 2][i]);
                    // }
                    output_temp[i][j] += input_temp[subcrs % 2][j] * weight_temp[subcrs % 2][i];
                }
            }
        }
        
        for (int i = 0; i < 4; i++)
        {
            shm_weight[write_flag][weight_sts_addr + i] = weight_ldg_reg[i];
            shm_input[write_flag][input_sts_addr + i * 32] = input_ldg_reg[i];
        }

        __syncthreads();  // 必须等待数据加载完成，不然lds会出错

        write_flag = write_flag ^ 1;

        // lds下个循环使用的数据/*选手自己实现的kernel*/

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            weight_temp[0][i] = shm_weight[load_flag ^ 1][weight_lds_addr + i]; 
            weight_temp[0][i + 4] = shm_weight[load_flag ^ 1][weight_lds_addr + i + 16]; 
        }

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            input_temp[0][i] = shm_input[load_flag ^ 1][input_lds_addr + i]; 
            input_temp[0][i + 4] = shm_input[load_flag ^ 1][input_lds_addr + i + 32];
        }

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                // if (z == 0 && y + i == 0 && ((x + j == 128 && j < 4) || (x + j - 4 == 128 + 32 && j >= 4))){
                //     printf("step %d : %.9f += %.9f * %.9f\n",7 + crs, output_temp[i][j], input_temp[1][j] , weight_temp[1][i]);
                // } 
                output_temp[i][j] += input_temp[1][j] * weight_temp[1][i];
            }
        }

    }

    // 计算输出偏移
    int output_offset;
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            output_offset = z * param.Oh * param.Ow * param.k + (y + i) * param.Oh * param.Ow + x + j;
            if ((x + j) < param.Ow * param.Oh && (y + i) < param.k)
            {
                param.pout[output_offset] = (_Float16)output_temp[i][j];
            }

            output_offset = z * param.Oh * param.Ow * param.k + (y + i) * param.Oh * param.Ow + x + j + 32;
            if ((x + j + 32) < param.Ow * param.Oh && (y + i) < param.k)
            {
                param.pout[output_offset] = (_Float16)output_temp[i][j + 4];
            }

            output_offset = z * param.Oh * param.Ow * param.k + (y + i + 16) * param.Oh * param.Ow + x + j;
            if ((x + j) < param.Ow * param.Oh && (y + i + 16) < param.k)
            {
                param.pout[output_offset] = (_Float16)output_temp[i + 4][j];
            }

            output_offset = z * param.Oh * param.Ow * param.k + (y + i + 16) * param.Oh * param.Ow + x + j + 32;
            if ((x + j + 32) < param.Ow * param.Oh && (y + i + 16) < param.k)
            {
                param.pout[output_offset] = (_Float16)output_temp[i + 4][j + 4];
            }
        }
    }
}

void launch_implicit_gemm(unsigned int outh, unsigned int outw, unsigned int k, unsigned int n, mykernelParamType* param) {
    dim3 grid((outh*outw + 15) / 16, (k + 15) / 16, n);
    dim3 block(16, 16, 1);
    directConvolution<<<grid, block>>>(*param);
}