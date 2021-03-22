#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include "device_launch_parameters.h"
#include <complex>
#include <device_functions.h>
#include <cuComplex.h>
#include <chrono>
#include <iostream>
#pragma comment(lib,"cufft.lib")

using namespace std;

__global__
void Complex_mult(cufftComplex * c, const cufftComplex * a, const cufftComplex * b) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    c[i].x = (a[i].x * b[i].x - a[i].y * (-b[i].y));
    c[i].y = (a[i].x * (-b[i].y) + a[i].y * b[i].x);
}

std::chrono::time_point<std::chrono::high_resolution_clock> now()
{
    return std::chrono::high_resolution_clock::now();
}

template <typename T>
double milliseconds(T t)
{
    return (double) std::chrono::duration_cast<std::chrono::nanoseconds>(t).count() / 1000000;
}

extern "C"
cufftComplex* FFT_GPU(cufftComplex* signal1, cufftComplex* signal2, int len_c)
{
    cufftComplex* GPU_data_first;
    cufftComplex* GPU_data_second;
    auto t1 = now();
    cudaMalloc((void**)&GPU_data_first, len_c * sizeof(cufftComplex));
    cudaMalloc((void**)&GPU_data_second, len_c * sizeof(cufftComplex));
    cufftHandle plan1;
    cufftPlan1d(&plan1, len_c, CUFFT_C2C, 1);


    auto t2 = now();
    cout<<"VIDEO MEM: "<< milliseconds(t2-t1)<<" ms"<<endl;
    cudaMemcpy(GPU_data_first, signal1, len_c * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(GPU_data_second, signal2, len_c * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    auto t3 = now();
    cout<<"COPY DATA: "<< milliseconds(t3-t2)<<" ms"<<endl;

    auto t3_1 = now();

    //cudaDeviceSynchronize();
    cufftExecC2C(plan1, (cufftComplex*)GPU_data_first, (cufftComplex*)GPU_data_first, CUFFT_FORWARD);
    cufftExecC2C(plan1, (cufftComplex*)GPU_data_second, (cufftComplex*)GPU_data_second, CUFFT_FORWARD);
    auto t4 = now();
    cout<<"FFT: "<< milliseconds(t4-t3_1)<<" ms"<<endl;
    //cufftDestroy(plan1); // освобождение памяти

    cufftComplex* Mult_result;
    cudaMalloc((void**)&Mult_result, len_c * sizeof(cufftComplex));

    auto t6 = now();

    Complex_mult <<<256, 192>>>(Mult_result, GPU_data_first, GPU_data_second);

    auto t7 = now();

    cout<<"cMULT: "<< milliseconds(t7-t6)<<" ms"<<endl;
    cudaFree(GPU_data_first);
    cudaFree(GPU_data_second);
    cufftHandle plan3;

    auto t8 = now();
   // cufftPlan1d(&plan3, len_c, CUFFT_C2C, 1);
    cufftExecC2C(plan1, (cufftComplex*)Mult_result, (cufftComplex*)Mult_result, CUFFT_INVERSE);
    auto t9 = now();
    cout<<"IFFT: "<< milliseconds(t9-t8)<<" ms"<<endl;

    cufftComplex* result_of_IFFT = new cufftComplex[len_c];
    auto t10 = now();
    cudaMemcpy(result_of_IFFT, Mult_result, sizeof(cufftComplex) * (len_c), cudaMemcpyDeviceToHost);
    auto t11 = now();
    cout<<"COPY FROM VIDEO: "<< milliseconds(t11-t10)<<" ms"<<endl;
    cudaFree(Mult_result);

    return result_of_IFFT;
}
