#ifndef __CONV2D_FP16_FWD_HEADER__
#define __CONV2D_FP16_FWD_HEADER__
#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
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
    _Float16*   in;                             //输入数据地址
    _Float16*   weight;                         //权值数据地址
    _Float16*   weight_trans;
    _Float16*   out;                            //输出数据地址
    _Float16*   data_col_device;
    _Float16*   output_gemm_device;
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
    unsigned int      algo;
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
    _Float16*         pin;                            //输入数据地址
    _Float16*         pweight;                        //权值数据地址
    _Float16*         pweight_trans;
    _Float16*         pout;                           //输出数据地址
    _Float16*         data_col_device;                          //预留
    _Float16*         output_gemm_device;
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
    unsigned int      algo;                          //预留
    unsigned int      revs6;                          //预留
    unsigned int      revs7;                          //预留
}mykernelParamType;         
typedef _Float16 _Float16_8 __attribute__((ext_vector_type(8)));
typedef _Float16 _Float16_4 __attribute__((ext_vector_type(4)));
typedef float float4_ __attribute__((ext_vector_type(4)));

union RegisterUnion
{
  _Float16_8 vector8;
  struct
  {
    _Float16_4 vector_front;
    _Float16_4 vector_rear;
  };
};

unsigned int getAlgos(int n, int c, int h, int w, int k, int r, int s);
int getParamsize(__in__ problem_t* problem, __out__ int* paramSize);
int getkernelInfo(__in__ problem_t* problem, __out__  kernelInfo_t* kernelInfo, __in_out__ void* param);
void convolutionForward(void* param);
void executeConvAlogs(mykernelParamType* param);
void launch_reshape_kernel(const _Float16* output_gemm_device, _Float16* output_gemm_device_rearrange,
                 int n, int k, int output_h, int output_w);
void im2col_batch_hip(const _Float16* data_im_device, int n, int channels, int height, int width,
                      int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w,
                      _Float16* data_col_device);                 
                 
extern "C" __global__ void implicitGemm(mykernelParamType param);
extern "C" __global__ void directConvolution(mykernelParamType param) __attribute__((amdgpu_flat_work_group_size(1,256)));

extern "C" __global__ void __launch_bounds__(1024) reshape_kernel(const _Float16* output_gemm_device, _Float16* output_gemm_device_rearrange,
                               			  int n, int k, int output_h, int output_w);
extern "C" __global__ void __launch_bounds__(1024) im2col_batch_kernel(const _Float16* data_im, int n, int channels, int height, int width,
                                    		   int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w,
                                    		   int output_h, int output_w, _Float16* data_col);
extern "C" __global__ void myHgemmV3Aligned(__half * __restrict__ A, __half * __restrict__ B, __half * __restrict__ C,
    										const int M, const int N, const int K);
extern "C" __global__ void myHgemmV1Aligned(__half * __restrict__ A, __half * __restrict__ B, __half * __restrict__ C,
    										const int M, const int N, const int K);
extern "C" __global__ void gemm_kernel8_4(__half *d_A, __half *d_B, __half *d_C, int M, int N, int K);

// extern "C" __global__ void GEMM_64x64x8_v3(__half * __restrict__ A, __half * __restrict__ B, __half * __restrict__ C,
//                                         const int M, const int N, const int K);

void launch_gemm_64x64x8_fp32(  __half * __restrict__ A, __half * __restrict__ B, __half * __restrict__ C,
    const int M, const int N, const int K);
    
extern "C" __global__ void im2col_kernel(const _Float16* data_im, int n, int channels, int height, int width,
                              int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w,
                              int output_h, int output_w, _Float16* data_col);

extern "C" __global__ void transpose_kernel(_Float16* A, _Float16* At, int m, int k);

void launch_gemm_32x32x16_fp16(_Float16* __restrict__ A, _Float16* __restrict__ B, _Float16* C, const int M, const int N, const int K);
#endif
