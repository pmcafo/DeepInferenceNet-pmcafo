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
//                    if (val > sta