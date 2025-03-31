#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
// #include <cuda/cuda_ext.h>
#include "verfiy.h"
#include "conv2d.h"

int main(int argc, char **argv)
{
    int n = atoi(argv[1]);
    int c = atoi(argv[2]);
    int h = atoi(argv[3]);
    int w = atoi(argv[4]);
    int k = atoi(argv[5]);
    int r = atoi(argv[6]);
    int s = atoi(argv[7]);
    int u = atoi(argv[8]);
    int v = atoi(argv[9]);
    int p = atoi(argv[10]);
    int q = atoi(argv[11]);
    int outh = (h - r + 2 * p) / u + 1;
    int outw = (w - s + 2 * q) / v + 1;


    float *pIn = (float *)malloc(n * c * h * w * sizeof(float));
    float *pWeight = (float *)malloc(k * c * r * s * sizeof(float));
    float *pOut = (float *)malloc(n * k * outh * outw * sizeof(float));
    float *pOut_host = (float *)malloc(n * k * outh * outw * sizeof(float));
    float *pIn_device, *pWeight_device, *pOut_device;
    cudaMalloc((void **)&pIn_device, n * c * h * w * sizeof(float));
    cudaMalloc((void **)&pWeight_device, k * c * r * s * sizeof(float));
    cudaMalloc((void **)&pOut_device, n * k * outh * outw * sizeof(float));

    for (int i = 0; i < n * c * h * w; i++)
    {
        pIn[i] = (rand() % 255) / 255.0;
    }

    for (int i = 0; i < k * c * r * s; i++)
    {
        pWeight[i] = (rand() % 255) / 255.0;
    }

    for (int i = 0; i < n * k * outh * outw; i++)
    {
        pOut[i] = 0.0;
        pOut_host[i] = 0.0;
    }
    cudaMemcpy(pIn_device, pIn, n * c * h * w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(pWeight_device, pWeight, k * c * r * s * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(pOut_device, pOut, n * k * outh * outw * sizeof(float), cudaMemcpyHostToDevice);
    /********************step 1*****************************/

    problem_t problem;
    int paramSize;
    kernelInfo_t kernelInfo;

    problem.in = pIn_device;
    problem.out = pOut_device;
    problem.weight = pWeight_device;
    problem.n = n;
    problem.c = c;
    problem.h = h;
    problem.w = w;
    problem.k = k;
    problem.r = r;
    problem.s = s;
    problem.u = u;
    problem.v = v;
    problem.p = p;
    problem.q = q;

    
    /**********************************step 2****************************/
    getParamsize(&problem, &paramSize);
    printf("\nparamsize:%d\n", paramSize);
    void *param = malloc(paramSize);

    getkernelInfo(&problem, &kernelInfo, param);
    convPlanType current_plan = scheduler(&problem, (mykernelParamType *)param);

    current_plan.conv_init((mykernelParamType *)param);
    current_plan.conv_run((mykernelParamType *)param);
    cudaMemcpy(pOut_host, pOut_device, n * k * outh * outw * sizeof(float), cudaMemcpyDeviceToHost);

    /*******************************cost time test************************************/
    float time_elapsed_optim = 0.0;
    float time_elapsed_baseline = 0.0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    int iternum = 100;
    for (int i = 0; i < iternum; i++)
    {
        // convolutionForward((mykernelParamType *)param); 
        current_plan.conv_run((mykernelParamType *)param);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed_optim, start, stop);
    printf("time: %f us\n", time_elapsed_optim * 1000 / iternum);
    

#ifndef TEST
    printf("===============start verfiy=====================\n");
    cudaEventRecord(start, 0);
    conv2dcpu(pIn, pWeight, pOut, n, c, h, w, k, r, s, u, v, p, q);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed_baseline, start, stop);

    int error = 0;
    for (int i = 0; i < n * k * outh * outw; i++)
    {
        float device_out = pOut_host[i];
        if ((fabs(pOut_host[i] - pOut[i])) / pOut_host[i] > 0.01 || isnan(device_out) || isinf(device_out))
        {
            printf("error, postion:%d, gpuvalue:%f, cpuvalue:%f\n", i, (float)pOut_host[i], (float)pOut[i]);
            error++;
            break;
        }
    }

    printf("error = %d\n", error); 
    float score = get_score(error, time_elapsed_optim, time_elapsed_baseline);
    printf("score = %.1f\n", score);
    
    printf("===============finish verfiy====================\n");
#endif
    cudaFree(pIn_device);
    cudaFree(pWeight_device);
    cudaFree(pOut_device);

    free(pIn);
    free(pWeight);
    free(pOut);
    free(pOut_host);

    current_plan.conv_exit((mykernelParamType *)param);
    free(param);
    return 0;
}
