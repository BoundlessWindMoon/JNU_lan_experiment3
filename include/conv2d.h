#ifndef __CONV2D_FP16_FWD_HEADER__
#define __CONV2D_FP16_FWD_HEADER__
// #include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#define __in__
#define __out__
#define __in_out__

#define DIRECT_CONV 1
#define IMPL_GEMM 2
#define IM2COL_GEMM_1BATCH 3
#define IM2COL_GEMM_NBATCH 4
#define WINOGRAD 5
#define IM2COL_GEMM_COMMON 6
#define IM2COL_GEMM_1BATCH_64 7
#define MMA_NAIVE 8

#define MMA_M 32 
#define MMA_N 32
#define MMA_K 16
#define WARP_SIZE 64

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
#define HALF2(pointer)  ((reinterpret_cast<__half2*>(&(pointer))[0]))

typedef struct
{
    float*   pweight_trans;
    float*   data_col_device;
    float*   output_gemm_device;
    float*   in;                             //输入数据地址
    float*   weight;                         //权值数据地址
    float*   out;                            //输出数据地址
    unsigned int      n;                              //batch szie              default value 1
    unsigned int      c;                              //channel number          default value 32
    unsigned int      h;                              //数据高                  default value 32
    unsigned int      w;                              //数据宽                  default value 32
    unsigned int      k;                              //卷积核数量              default value 32
    unsigned int      r;                              //卷积核高                default value 1
    unsigned int      s;                              //卷积核宽                default value 1
    unsigned int      u;                              //卷积在高方向上的步长     default value 1
    unsigned int      v;                              //卷积在宽方向上的步长     default value 1
    unsigned int      p;                              //卷积在高方向上的补边     default value 0
    unsigned int      q;                              //卷积在宽方向上的补边     default value 0
}problem_t;

typedef struct
{
    /*
    void*                       kernelPtr;   
    unsigned int         block1_x;                    //blockx  number
    unsigned int         block1_y;                    //blocky  number
    unsigned int         block1_z;                    //blockz  number
    unsigned int         thread1_x;                   //threadx number per block
    unsigned int         thread1_y;                   //thready executeConvAlogsumber
    unsigned int         block2_z;                    //blockz  number
    unsigned int         thread2_x;                   //threadx number per block
    unsigned int         thread2_y;                   //thready number per block
    unsigned int         thread2_z;                   //threadz number per block

    void*       kernelPtr3;
    unsigned int         block3_x;                    //blockx  number
    unsigned int         block3_y;                    //blocky  number
    unsigned int         block3_z;                    //blockz  number
    unsigned int         thread3_x;                   //threadx number per block
    unsigned int         thread3_y;                   //thready number per block
    unsigned int         thread3_z;                   //threadz number per block

    unsigned int         dynmicLdsSize;             //动态分配的lds大小，如果不使用动态分配的lds，则该值为0；
    */    
}kernelInfo_t;

typedef struct mykernelParamType
{
    float*         pweight_trans;
    float*         data_col_device;                       
    float*         output_gemm_device;
    float*         pin;                            //输入数据地址
    float*         pout;                           //输出数据地址
    float*         pweight;                        //权值数据地址
    unsigned int      n;                              //batch szie            
    unsigned int      c;                              //channel number        
    unsigned int      h;                              //数据高                
    unsigned int      w;                              //数据宽                
    unsigned int      k;                              //卷积核数量            
    unsigned int      r;                              //卷积核高              
    unsigned int      s;                              //卷积核宽              
    unsigned int      u;                              //卷积在高方向上的步长  
    unsigned int      v;                              //卷积在宽方向上的步长  
    unsigned int      p;                              //卷积在高方向上的补边  
    unsigned int      q;                              //卷积在宽方向上的补边  
    unsigned int      Oh;                             //卷积在高方向上输出大小    
    unsigned int      Ow;                             //卷积在宽方向上输出大小
    unsigned int      revs6;                          //预留
    unsigned int      revs7;                          //预留
}mykernelParamType; 


typedef struct convPlanType{
    char name[32];
    void (*conv_init)(mykernelParamType*);
    void (*conv_run)(mykernelParamType*);
    void (*conv_exit)(mykernelParamType*);
}convPlanType;

typedef struct convParamType
{
    unsigned int      n;                              //batch szie            
    unsigned int      c;                              //channel number        
    unsigned int      h;                              //数据高                
    unsigned int      w;                              //数据宽                
    unsigned int      k;                              //卷积核数量            
    unsigned int      r;                              //卷积核高              
    unsigned int      s;                              //卷积核宽              
    unsigned int      u;                              //卷积在高方向上的步长  
    unsigned int      v;                              //卷积在宽方向上的步长  
    unsigned int      p;                              //卷积在高方向上的补边  
    unsigned int      q;                              //卷积在宽方向上的补边  
}convParamType;

#define PARAM_EQUAL(ref, in) (ref.n == in.n) &&\
                            (ref.c == in.c) &&\
                            (ref.h == in.h) &&\
                            (ref.w == in.w) &&\
                            (ref.k == in.k) &&\
                            (ref.r == in.r) &&\
                            (ref.s == in.s) &&\
                            (ref.u == in.u) &&\
                            (ref.v == in.v) &&\
                            (ref.p == in.p) &&\
                            (ref.q == in.q)


// typedef _Float16 _Float16_8 __attribute__((ext_vector_type(8)));
// typedef _Float16 _Float16_4 __attribute__((ext_vector_type(4)));
typedef float float4_ __attribute__((ext_vector_type(4)));

// union RegisterUnion
// {
//   _Float16_8 vector8;
//   struct
//   {
//     _Float16_4 vector_front;
//     _Float16_4 vector_rear;
//   };
// };

convPlanType scheduler(problem_t *problem, mykernelParamType * param);

int getParamsize(__in__ problem_t* problem, __out__ int* paramSize);

int getkernelInfo(__in__ problem_t* problem, __out__  kernelInfo_t* kernelInfo, __in_out__ void* param);
                   
extern "C" __global__ void directConvolution(mykernelParamType param) __attribute__((amdgpu_flat_work_group_size(1,256)));
void launch_implicit_gemm(unsigned int outh, unsigned int outw, unsigned int k, unsigned int n, mykernelParamType* param);  
void launch_implicit_gemm_v0(unsigned int outh, unsigned int outw, unsigned int k, unsigned int n, mykernelParamType* param);
void launch_implicit_gemm_v1(unsigned int outh, unsigned int outw, unsigned int k, unsigned int n, mykernelParamType* param);
void launch_implicit_gemm_v2(unsigned int outh, unsigned int outw, unsigned int k, unsigned int n, mykernelParamType* param);
#endif
