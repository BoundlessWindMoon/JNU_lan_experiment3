
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>  // 添加bool头文件

typedef struct mykernelParamType {
    float*         pin;        // 输入数据地址
    float*         pout;       // 输出数据地址
    float*         pweight;    // 权值数据地址
    unsigned int   n;          // batch size
    unsigned int   c;          // 输入通道数
    unsigned int   h;          // 输入高度
    unsigned int   w;          // 输入宽度
    unsigned int   k;          // 输出通道数（卷积核数量）
    unsigned int   r;          // 卷积核高度
    unsigned int   s;          // 卷积核宽度
    unsigned int   u;          // 高度方向步长
    unsigned int   v;          // 宽度方向步长
    unsigned int   p;          // 高度方向补边
    unsigned int   q;          // 宽度方向补边
    unsigned int   Oh;         // 输出高度
    unsigned int   Ow;         // 输出宽度
    unsigned int   revs6;
    unsigned int   revs7;
} mykernelParamType;


void conv_implicit_unimplement(mykernelParamType param)
{ 
    // 框架代码无需修改
    int M = param.k;
    int N = param.n * param.Oh * param.Ow;
    int K = param.c * param.r * param.s;  // 修正拼写错误 prarm->param
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            float sum = 0.0;
            for(int k = 0; k < K; k++) {
                // sum += param.pin[]  * param.pweight[];
            }
            // param.pout[] = sum;
        }
    }
}

void conv_implicit(mykernelParamType param) {
    int M = param.k;
    int N = param.n * param.Oh * param.Ow;
    int K = param.c * param.r * param.s;
    int k, n, oh, ow, c, r, s; 
    int input_addr, weight_addr, output_addr;

    for (int i = 0; i < M; i++) {
        k = i; 
        for (int j = 0; j < N; j++) {
            n = j / (param.Oh * param.Ow);
            oh = (j % (param.Oh * param.Ow)) / param.Ow;
            ow = j % param.Ow;
            float sum = 0.0f;

            /* NKOHOW 布局输出地址 */
            output_addr = n * param.k * param.Oh * param.Ow  // N维度步长
                        + k * param.Oh * param.Ow            // K维度步长
                        + oh * param.Ow                      // H维度步长
                        + ow;                                // W维度步长

            for (int kk = 0; kk < K; kk++) { 
                c = kk / (param.r * param.s);
                r = (kk % (param.r * param.s)) / param.s;
                s = (kk % (param.r * param.s)) % param.s;

                int ih = oh * param.u - param.p + r;
                int iw = ow * param.v - param.q + s;

                if (ih >= 0 && ih < param.h && iw >= 0 && iw < param.w) {
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
            param.pout[output_addr] = sum;
        }
    }
}

void conv_direct(mykernelParamType param) {
    for (unsigned int n = 0; n < param.n; ++n) {                
        for (unsigned int k = 0; k < param.k; ++k) {             
            for (unsigned int oh = 0; oh < param.Oh; ++oh) {     
                for (unsigned int ow = 0; ow < param.Ow; ++ow) { 
                    float sum = 0.0f;
                    
                    for (unsigned int c = 0; c < param.c; ++c) {
                        for (unsigned int r_idx = 0; r_idx < param.r; ++r_idx) {
                            for (unsigned int s_idx = 0; s_idx < param.s; ++s_idx) {

                                int ih = oh * param.u - param.p + r_idx;
                                int iw = ow * param.v - param.q + s_idx;
                                
                                if (ih >= 0 && ih < param.h && iw >= 0 && iw < param.w) {
                                    size_t in_idx = n * param.c * param.h * param.w 
                                                  + c * param.h * param.w 
                                                  + ih * param.w 
                                                  + iw;
                                    
                                    size_t weight_idx = k * param.c * param.r * param.s 
                                                      + c * param.r * param.s 
                                                      + r_idx * param.s 
                                                      + s_idx;
                                    
                                    sum += param.pin[in_idx] * param.pweight[weight_idx];
                                }
                            }
                        }
                    }
                    
                    /* NKOHOW 布局输出地址 */
                    size_t out_idx = n * param.k * param.Oh * param.Ow 
                                   + k * param.Oh * param.Ow 
                                   + oh * param.Ow 
                                   + ow;
                    param.pout[out_idx] = sum;
                }
            }
        }
    }
}


// 结果比较函数
bool compare_output(float* out1, float* out2, size_t size, float epsilon = 1e-4) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(out1[i] - out2[i]) > epsilon) {
            printf("Mismatch at index %zu: %.6f vs %.6f\n", i, out1[i], out2[i]);
            return false;
        }
    }
    return true;
}

// 测试用例
int main() {
    // 参数配置（示例）
    mykernelParamType param = {
        .n = 1,         // batch=1
        .c = 3,         // 输入通道=3
        .h = 32,        // 输入高度=32
        .w = 32,        // 输入宽度=32
        .k = 64,        // 输出通道=64
        .r = 3,         // 卷积核高=3
        .s = 3,         // 卷积核宽=3
        .u = 1,         // 高度步长=1
        .v = 1,         // 宽度步长=1
        .p = 1,         // 高度补边=1
        .q = 1,         // 宽度补边=1
        .Oh = 32,       // 输出高度=32（根据公式计算）
        .Ow = 32        // 输出宽度=32
    };

    // 计算数据尺寸
    size_t input_size = param.n * param.c * param.h * param.w;
    size_t weight_size = param.k * param.c * param.r * param.s;
    size_t output_size = param.n * param.k * param.Oh * param.Ow;

    // 分配内存
    float* input = (float*)malloc(input_size * sizeof(float));
    float* weight = (float*)malloc(weight_size * sizeof(float));
    float* output_implicit = (float*)malloc(output_size * sizeof(float));
    float* output_direct = (float*)malloc(output_size * sizeof(float));

    // 初始化数据（随机值）
    for (size_t i = 0; i < input_size; ++i) input[i] = (float)rand() / RAND_MAX;
    for (size_t i = 0; i < weight_size; ++i) weight[i] = (float)rand() / RAND_MAX;

    // 设置参数指针
    param.pin = input;
    param.pweight = weight;

    // 执行隐式卷积
    param.pout = output_implicit;
    conv_implicit(param);

    // 执行直接卷积
    param.pout = output_direct;
    conv_direct(param);

    // 结果比较
    bool is_match = compare_output(output_implicit, output_direct, output_size);
    printf("Results match: %s\n", is_match ? "Yes" : "No");

    // 清理内存
    free(input);
    free(weight);
    free(output_implicit);
    free(output_direct);

    return 0;
}
