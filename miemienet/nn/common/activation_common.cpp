#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include "../../framework/config.h"
#include "../../framework/tensor.h"
#include "../activation.h"
#include "activation_common.h"

#if BACKEND_X86
#include <immintrin.h>
#endif // BACKEND_X86

#if BACKEND_ARM
//#include <arm_neon.h>
#endif // BACKEND_ARM


NS_MM_F_BEGIN

template<typename data_t>
void leakyrelu_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, float alpha, int num){
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num; i++) {
        if (x[i] > static_cast<data_t>(0.f))
        {
            y[i] = x[i];
        }
        else
        {
            y[i] = x[i] * alpha;
        }
    }
}

template<typename data_t>
void leakyrelu_x86_kernel(const int num_threads_, const data_t* x, data_t* y, float alpha, int num){

    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num; i++) {
        if (x[i] > static_cast<data_t>(0.f))
        {
            y[i] = x[i];
        }
        else
        {
            y[i] = x[i] * alpha;
        }
    }

//    const int BLOCK = 128;
//    #pragma omp parallel for num_threads(num_threads_)
//    for (int b = 0; b < num; b+=BLOCK) {
//        for (int i = 0; i < BLOCK; i++) {
//            int j = i + b;
//            if (x[j] > static_cast<data_t>(0.f))
//            {
//                y[j] = x[j];
//            }
//            else
//            {
//                y[j] = x[j] * alpha;
//            }
//        }
//    }


//    int H = 256;
//    int W = 256*256;
//    #pragma omp parallel for num_threads(num_threads_)
//    for (int h = 0; h < H; h+=BLOCK) {
//        for (int w = 0; w < W; w+=BLOCK) {
//            for (int i = 0; i < BLOCK; i++) {
//                for (int j = 0; j < BLOCK; j++) {
//                    float val = x[((h + j) * W) + (w + i)];
//                    if (val > static_cast<data_t>(0.f))
//                    {
//                        y[((h + j) * W) + (w + i)] = val;
//                    }
//                    else
//                    {
//                        y[((h + j) * W) + (w + i)] = val * alpha;
//                    }
//                }
//            }
//        }
//    }
}

template<typename data_t>
void leakyrelu_grad_cpp_kernel(const int num_threads_, const data_t* dy, const data_t* x, data_t* dx, float alpha, int num){
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num; i++) {
        if (x[i] > static_cast<data_t>(0.f))
        {
            dx[i] = dy[i];
        }
        else
        {
            dx[i] = dy[i] * alpha;
        }
    }
}

template<typename data_t>
void relu_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num){
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num; i++) {
        if (x[i] > static_cast<data_t>(0.f))
        {
            y[i] = x[i];
        }
        else
        {
            y[i] = static_cast<data_t>(0.f);
        }
    }
}

template<typename data_t>
void relu_grad_cpp_kernel(const int num_threads_, const data_t* dy, const data_t* x, data_t* dx, int num){
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num; i++) {
        if (x[i] > static_cast<data_t>(0.f))
        {
            dx[i] = dy[i];
        }
        else
        {
            dx[i] = static_cast<data_t>(0.f);
        }
    }
}

// expf instead of exp should be used for float type, complement
// and register float kernel separatelly
template<typename data_t>
void sigmoid_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num){
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num; i++) {
        y[i] = static_cast<float>(1.f / (1.f + expf(-x[i])));
    }
}

template<typename data_t>
void sigmoid_grad_cpp_kernel(const int num_threads_, const data_t* dy, const data_t* y, data_t* dx, int num){
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num; i++) {
        dx[i] = dy[i] * y[i] * (1.f - y[i]);
    }
}

template<typename data_t>
void tanh_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num){
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num; i++) {
        y[i] = tanhf(x[i]);
    }
}

template<typename data_t>
void tanh_grad_cpp_kernel(const int num_threads_, const data_t* dy, const data_t* y, data_t* dx, int num){
    #pragma omp parallel for num_threa