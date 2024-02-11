
//#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <chrono>
#include <immintrin.h>

float calc_diff(float* x, float* y, int numel)
{
    float diff = 0.f;
    float M = 1.f;
//    float M = 1.f / (float)numel;
    for (int i = 0; i < numel; i++)
    {
        diff += (x[i] - y[i]) * (x[i] - y[i]) * M;
    }
    return diff;
}




template<typename data_t>
void transpose2d_10_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int H, int W) {
    // y[w][h] = x[h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            y[(w * H) + h] = x[(h * W) + w];
        }
    }
}


void matmul_true_cpp_kernel(const int num_threads_, const float* input, const float* weight, float* output, int batch_size, int ch_in, int ch_out) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int bs = 0; bs < batch_size; bs++) {
        for (int oc = 0; oc < ch_out; oc++) {
            for (int ic = 0; ic < ch_in; ic++) {
                output[bs * ch_out + oc] += input[bs * ch_in + ic] * weight[ic * ch_out + oc];
            }
        }
    }
}


template<typename data_t>
void matmul_cpp_kernel(const int num_threads_, const data_t* input, const data_t* weight, data_t* output, int batch_size, int ch_in, int ch_out) {

    // 以这种方式遍历时，左矩阵是一行一行地遍历，右矩阵是一行一行地遍历，cache命中率高。
//    #pragma omp parallel for num_threads(num_threads_)
//    for (int bs = 0; bs < batch_size; bs++) {
//        for (int ic = 0; ic < ch_in; ic++) {
//            const float _a = input[bs * ch_in + ic];
//            for (int oc = 0; oc < ch_out; oc++) {
//                output[bs * ch_out + oc] += _a * weight[ic * ch_out + oc];
//            }
//        }
//    }


    int elempack = 8;
    #pragma omp parallel for num_threads(num_threads_)
    for (int bs = 0; bs < batch_size; bs++) {
        const float* w_ptr = weight;
        const float* x_ptr = input + bs * ch_in;
        for (int ic = 0; ic < ch_in; ic++) {
            float* out_ptr = output + bs * ch_out;
            __m256 _a8 = _mm256_broadcast_ss(x_ptr);
            int oc = 0;
            for (; oc + (elempack - 1) < ch_out; oc += elempack) {
                __m256 _b = _mm256_loadu_ps(w_ptr);
                __m256 _out = _mm256_loadu_ps(out_ptr);
                _out = _mm256_fmadd_ps(_a8, _b, _out);
                _mm256_storeu_ps(out_ptr, _out);
                w_ptr += elempack;
                out_ptr += elempack;
            }
            __m128 _a4 = _mm_broadcast_ss(x_ptr);
            for (; oc + 3 < ch_out; oc += 4) {
                __m128 _b = _mm_load_ps(w_ptr);
                __m128 _out = _mm_load_ps(out_ptr);
                _out = _mm_fmadd_ps(_a4, _b, _out);
                _mm_store_ps(out_ptr, _out);
                w_ptr += 4;
                out_ptr += 4;
            }
            for (; oc < ch_out; oc++) {
                output[bs * ch_out + oc] += input[bs * ch_in + ic] * weight[ic * ch_out + oc];
                w_ptr += 1;
                out_ptr += 1;
            }
            x_ptr++;
        }
    }
}



template<typename data_t>
void matmul_block_pack4_cpp_kernel(const int num_threads_, const data_t* input, const data_t* weight, data_t* output, int batch_size, int ch_in, int ch_out) {

    #pragma omp parallel for num_threads(num_threads_)
    for (int bs = 0; bs < batch_size; bs++) {
        for (int ic = 0; ic < ch_in; ic+=4) {
            const float _a = input[bs * ch_in + ic];
            const float _b = input[bs * ch_in + ic + 1];
            const float _c = input[bs * ch_in + ic + 2];
            const float _d = input[bs * ch_in + ic + 3];
            for (int oc = 0; oc < ch_out; oc+=4) {
                output[bs * ch_out + oc] += _a * weight[ic * ch_out + oc] + \
                                            _b * weight[(ic + 1) * ch_out + oc] + \
                                            _c * weight[(ic + 2) * ch_out + oc] + \
                                            _d * weight[(ic + 3) * ch_out + oc];
                output[bs * ch_out + oc + 1] += _a * weight[ic * ch_out + oc + 1] + \
                                                _b * weight[(ic + 1) * ch_out + oc + 1] + \
                                                _c * weight[(ic + 2) * ch_out + oc + 1] + \
                                                _d * weight[(ic + 3) * ch_out + oc + 1];
                output[bs * ch_out + oc + 2] += _a * weight[ic * ch_out + oc + 2] + \
                                                _b * weight[(ic + 1) * ch_out + oc + 2] + \
                                                _c * weight[(ic + 2) * ch_out + oc + 2] + \
                                                _d * weight[(ic + 3) * ch_out + oc + 2];
                output[bs * ch_out + oc + 3] += _a * weight[ic * ch_out + oc + 3] + \
                                                _b * weight[(ic + 1) * ch_out + oc + 3] + \
                                                _c * weight[(ic + 2) * ch_out + oc + 3] + \
                                                _d * weight[(ic + 3) * ch_out + oc + 3];
            }
        }
    }
}

template<typename data_t>
void matmul_block_pack4all_cpp_kernel(const int num_threads_, const data_t* input, const data_t* weight, data_t* output, int batch_size, int ch_in, int ch_out) {

    #pragma omp parallel for num_threads(num_threads_)
    for (int bs = 0; bs < batch_size; bs+=4) {
        for (int ic = 0; ic < ch_in; ic+=4) {
            const float _a0 = input[bs * ch_in + ic];
            const float _b0 = input[bs * ch_in + ic + 1];
            const float _c0 = input[bs * ch_in + ic + 2];
            const float _d0 = input[bs * ch_in + ic + 3];
            const float _a1 = input[(bs + 1) * ch_in + ic];
            const float _b1 = input[(bs + 1) * ch_in + ic + 1];
            const float _c1 = input[(bs + 1) * ch_in + ic + 2];
            const float _d1 = input[(bs + 1) * ch_in + ic + 3];
            const float _a2 = input[(bs + 2) * ch_in + ic];
            const float _b2 = input[(bs + 2) * ch_in + ic + 1];
            const float _c2 = input[(bs + 2) * ch_in + ic + 2];
            const float _d2 = input[(bs + 2) * ch_in + ic + 3];
            const float _a3 = input[(bs + 3) * ch_in + ic];
            const float _b3 = input[(bs + 3) * ch_in + ic + 1];
            const float _c3 = input[(bs + 3) * ch_in + ic + 2];
            const float _d3 = input[(bs + 3) * ch_in + ic + 3];
            for (int oc = 0; oc < ch_out; oc+=4) {
                output[bs * ch_out + oc] += _a0 * weight[ic * ch_out + oc] + \
                                            _b0 * weight[(ic + 1) * ch_out + oc] + \
                                            _c0 * weight[(ic + 2) * ch_out + oc] + \
                                            _d0 * weight[(ic + 3) * ch_out + oc];
                output[bs * ch_out + oc + 1] += _a0 * weight[ic * ch_out + oc + 1] + \
                                                _b0 * weight[(ic + 1) * ch_out + oc + 1] + \
                                                _c0 * weight[(ic + 2) * ch_out + oc + 1] + \
                                                _d0 * weight[(ic + 3) * ch_out + oc + 1];
                output[bs * ch_out + oc + 2] += _a0 * weight[ic * ch_out + oc + 2] + \
                                                _b0 * weight[(ic + 1) * ch_out + oc + 2] + \
                                                _c0 * weight[(ic + 2) * ch_out + oc + 2] + \
                                                _d0 * weight[(ic + 3) * ch_out + oc + 2];
                output[bs * ch_out + oc + 3] += _a0 * weight[ic * ch_out + oc + 3] + \
                                                _b0 * weight[(ic + 1) * ch_out + oc + 3] + \
                                                _c0 * weight[(ic + 2) * ch_out + oc + 3] + \
                                                _d0 * weight[(ic + 3) * ch_out + oc + 3];

                output[(bs + 1) * ch_out + oc] += _a1 * weight[ic * ch_out + oc] + \
                                            _b1 * weight[(ic + 1) * ch_out + oc] + \
                                            _c1 * weight[(ic + 2) * ch_out + oc] + \
                                            _d1 * weight[(ic + 3) * ch_out + oc];
                output[(bs + 1) * ch_out + oc + 1] += _a1 * weight[ic * ch_out + oc + 1] + \
                                                _b1 * weight[(ic + 1) * ch_out + oc + 1] + \
                                                _c1 * weight[(ic + 2) * ch_out + oc + 1] + \
                                                _d1 * weight[(ic + 3) * ch_out + oc + 1];
                output[(bs + 1) * ch_out + oc + 2] += _a1 * weight[ic * ch_out + oc + 2] + \
                                                _b1 * weight[(ic + 1) * ch_out + oc + 2] + \
                                                _c1 * weight[(ic + 2) * ch_out + oc + 2] + \
                                                _d1 * weight[(ic + 3) * ch_out + oc + 2];
                output[(bs + 1) * ch_out + oc + 3] += _a1 * weight[ic * ch_out + oc + 3] + \
                                                _b1 * weight[(ic + 1) * ch_out + oc + 3] + \
                                                _c1 * weight[(ic + 2) * ch_out + oc + 3] + \
                                                _d1 * weight[(ic + 3) * ch_out + oc + 3];

                output[(bs + 2) * ch_out + oc] += _a2 * weight[ic * ch_out + oc] + \
                                            _b2 * weight[(ic + 1) * ch_out + oc] + \
                                            _c2 * weight[(ic + 2) * ch_out + oc] + \
                                            _d2 * weight[(ic + 3) * ch_out + oc];
                output[(bs + 2) * ch_out + oc + 1] += _a2 * weight[ic * ch_out + oc + 1] + \
                                                _b2 * weight[(ic + 1) * ch_out + oc + 1] + \
                                                _c2 * weight[(ic + 2) * ch_out + oc + 1] + \
                                                _d2 * weight[(ic + 3) * ch_out + oc + 1];
                output[(bs + 2) * ch_out + oc + 2] += _a2 * weight[ic * ch_out + oc + 2] + \
                                                _b2 * weight[(ic + 1) * ch_out + oc + 2] + \
                                                _c2 * weight[(ic + 2) * ch_out + oc + 2] + \
                                                _d2 * weight[(ic + 3) * ch_out + oc + 2];
                output[(bs + 2) * ch_out + oc + 3] += _a2 * weight[ic * ch_out + oc + 3] + \
                                                _b2 * weight[(ic + 1) * ch_out + oc + 3] + \
                                                _c2 * weight[(ic + 2) * ch_out + oc + 3] + \
                                                _d2 * weight[(ic + 3) * ch_out + oc + 3];

                output[(bs + 3) * ch_out + oc] += _a3 * weight[ic * ch_out + oc] + \
                                            _b3 * weight[(ic + 1) * ch_out + oc] + \
                                            _c3 * weight[(ic + 2) * ch_out + oc] + \
                                            _d3 * weight[(ic + 3) * ch_out + oc];
                output[(bs + 3) * ch_out + oc + 1] += _a3 * weight[ic * ch_out + oc + 1] + \
                                                _b3 * weight[(ic + 1) * ch_out + oc + 1] + \
                                                _c3 * weight[(ic + 2) * ch_out + oc + 1] + \
                                                _d3 * weight[(ic + 3) * ch_out + oc + 1];
                output[(bs + 3) * ch_out + oc + 2] += _a3 * weight[ic * ch_out + oc + 2] + \
                                                _b3 * weight[(ic + 1) * ch_out + oc + 2] + \
                                                _c3 * weight[(ic + 2) * ch_out + oc + 2] + \
                                                _d3 * weight[(ic + 3) * ch_out + oc + 2];
                output[(bs + 3) * ch_out + oc + 3] += _a3 * weight[ic * ch_out + oc + 3] + \
                                                _b3 * weight[(ic + 1) * ch_out + oc + 3] + \
                                                _c3 * weight[(ic + 2) * ch_out + oc + 3] + \
                                                _d3 * weight[(ic + 3) * ch_out + oc + 3];
            }
        }
    }
}



template<typename data_t>
void matmul_block_pack4all_ptr_cpp_kernel(const int num_threads_, const data_t* input, const data_t* weight, data_t* output, int batch_size, int ch_in, int ch_out) {

    #pragma omp parallel for num_threads(num_threads_)
    for (int bs = 0; bs < batch_size; bs+=4) {
        for (int ic = 0; ic < ch_in; ic+=4) {
            const float _a0 = input[bs * ch_in + ic];
            const float _b0 = input[bs * ch_in + ic + 1];
            const float _c0 = input[bs * ch_in + ic + 2];
            const float _d0 = input[bs * ch_in + ic + 3];
            const float _a1 = input[(bs + 1) * ch_in + ic];
            const float _b1 = input[(bs + 1) * ch_in + ic + 1];
            const float _c1 = input[(bs + 1) * ch_in + ic + 2];
            const float _d1 = input[(bs + 1) * ch_in + ic + 3];
            const float _a2 = input[(bs + 2) * ch_in + ic];
            const float _b2 = input[(bs + 2) * ch_in + ic + 1];
            const float _c2 = input[(bs + 2) * ch_in + ic + 2];
            const float _d2 = input[(bs + 2) * ch_in + ic + 3];
            const float _a3 = input[(bs + 3) * ch_in + ic];
            const float _b3 = input[(bs + 3) * ch_in + ic + 1];
            const float _c3 = input[(bs + 3) * ch_in + ic + 2];
            const float _d3 = input[(bs + 3) * ch_in + ic + 3];
            for (int oc = 0; oc < ch_out; oc+=4) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                const float* w_ptr1 = weight + (ic + 1) * ch_out + oc;
                const float* w_ptr2 = weight + (ic + 2) * ch_out + oc;
                const float* w_ptr3 = weight + (ic + 3) * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;
                float* y_ptr1 = output + (bs + 1) * ch_out + oc;
                float* y_ptr2 = output + (bs + 2) * ch_out + oc;
                float* y_ptr3 = output + (bs + 3) * ch_out + oc;


                *(y_ptr0 + 0) += _a0 * *(w_ptr0 + 0) + \
                                 _b0 * *(w_ptr1 + 0) + \
                                 _c0 * *(w_ptr2 + 0) + \
                                 _d0 * *(w_ptr3 + 0);
                *(y_ptr0 + 1) += _a0 * *(w_ptr0 + 1) + \
                                 _b0 * *(w_ptr1 + 1) + \
                                 _c0 * *(w_ptr2 + 1) + \
                                 _d0 * *(w_ptr3 + 1);
                *(y_ptr0 + 2) += _a0 * *(w_ptr0 + 2) + \
                                 _b0 * *(w_ptr1 + 2) + \
                                 _c0 * *(w_ptr2 + 2) + \
                                 _d0 * *(w_ptr3 + 2);
                *(y_ptr0 + 3) += _a0 * *(w_ptr0 + 3) + \
                                 _b0 * *(w_ptr1 + 3) + \
                                 _c0 * *(w_ptr2 + 3) + \
                                 _d0 * *(w_ptr3 + 3);

                *(y_ptr1 + 0) += _a1 * *(w_ptr0 + 0) + \
                                 _b1 * *(w_ptr1 + 0) + \
                                 _c1 * *(w_ptr2 + 0) + \
                                 _d1 * *(w_ptr3 + 0);
                *(y_ptr1 + 1) += _a1 * *(w_ptr0 + 1) + \
                                 _b1 * *(w_ptr1 + 1) + \
                                 _c1 * *(w_ptr2 + 1) + \
                                 _d1 * *(w_ptr3 + 1);
                *(y_ptr1 + 2) += _a1 * *(w_ptr0 + 2) + \
                                 _b1 * *(w_ptr1 + 2) + \
                                 _c1 * *(w_ptr2 + 2) + \
                                 _d1 * *(w_ptr3 + 2);
                *(y_ptr1 + 3) += _a1 * *(w_ptr0 + 3) + \
                                 _b1 * *(w_ptr1 + 3) + \
                                 _c1 * *(w_ptr2 + 3) + \
                                 _d1 * *(w_ptr3 + 3);

                *(y_ptr2 + 0) += _a2 * *(w_ptr0 + 0) + \
                                 _b2 * *(w_ptr1 + 0) + \
                                 _c2 * *(w_ptr2 + 0) + \
                                 _d2 * *(w_ptr3 + 0);
                *(y_ptr2 + 1) += _a2 * *(w_ptr0 + 1) + \
                                 _b2 * *(w_ptr1 + 1) + \
                                 _c2 * *(w_ptr2 + 1) + \
                                 _d2 * *(w_ptr3 + 1);
                *(y_ptr2 + 2) += _a2 * *(w_ptr0 + 2) + \
                                 _b2 * *(w_ptr1 + 2) + \
                                 _c2 * *(w_ptr2 + 2) + \
                                 _d2 * *(w_ptr3 + 2);
                *(y_ptr2 + 3) += _a2 * *(w_ptr0 + 3) + \
                                 _b2 * *(w_ptr1 + 3) + \
                                 _c2 * *(w_ptr2 + 3) + \
                                 _d2 * *(w_ptr3 + 3);

                *(y_ptr3 + 0) += _a3 * *(w_ptr0 + 0) + \
                                 _b3 * *(w_ptr1 + 0) + \
                                 _c3 * *(w_ptr2 + 0) + \
                                 _d3 * *(w_ptr3 + 0);
                *(y_ptr3 + 1) += _a3 * *(w_ptr0 + 1) + \
                                 _b3 * *(w_ptr1 + 1) + \
                                 _c3 * *(w_ptr2 + 1) + \
                                 _d3 * *(w_ptr3 + 1);
                *(y_ptr3 + 2) += _a3 * *(w_ptr0 + 2) + \
                                 _b3 * *(w_ptr1 + 2) + \
                                 _c3 * *(w_ptr2 + 2) + \
                                 _d3 * *(w_ptr3 + 2);
                *(y_ptr3 + 3) += _a3 * *(w_ptr0 + 3) + \
                                 _b3 * *(w_ptr1 + 3) + \
                                 _c3 * *(w_ptr2 + 3) + \
                                 _d3 * *(w_ptr3 + 3);
            }
        }
    }
}



template<typename data_t>
void matmul_block_pack4all_SIMD_cpp_kernel(const int num_threads_, const data_t* input, const data_t* weight, data_t* output, int batch_size, int ch_in, int ch_out) {

    #pragma omp parallel for num_threads(num_threads_)
    for (int bs = 0; bs < batch_size; bs+=4) {
        for (int ic = 0; ic < ch_in; ic+=4) {
//            const float _a0 = input[bs * ch_in + ic];
//            const float _b0 = input[bs * ch_in + ic + 1];
//            const float _c0 = input[bs * ch_in + ic + 2];
//            const float _d0 = input[bs * ch_in + ic + 3];
//            const float _a1 = input[(bs + 1) * ch_in + ic];
//            const float _b1 = input[(bs + 1) * ch_in + ic + 1];
//            const float _c1 = input[(bs + 1) * ch_in + ic + 2];
//            const float _d1 = input[(bs + 1) * ch_in + ic + 3];
//            const float _a2 = input[(bs + 2) * ch_in + ic];
//            const float _b2 = input[(bs + 2) * ch_in + ic + 1];
//            const float _c2 = input[(bs + 2) * ch_in + ic + 2];
//            const float _d2 = input[(bs + 2) * ch_in + ic + 3];
//            const float _a3 = input[(bs + 3) * ch_in + ic];
//            const float _b3 = input[(bs + 3) * ch_in + ic + 1];
//            const float _c3 = input[(bs + 3) * ch_in + ic + 2];
//            const float _d3 = input[(bs + 3) * ch_in + ic + 3];
            __m128 _a0 = _mm_broadcast_ss(input + bs * ch_in + ic);
            __m128 _b0 = _mm_broadcast_ss(input + bs * ch_in + ic + 1);
            __m128 _c0 = _mm_broadcast_ss(input + bs * ch_in + ic + 2);
            __m128 _d0 = _mm_broadcast_ss(input + bs * ch_in + ic + 3);
            __m128 _a1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic);
            __m128 _b1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 1);
            __m128 _c1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 2);
            __m128 _d1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 3);
            __m128 _a2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic);
            __m128 _b2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 1);
            __m128 _c2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 2);
            __m128 _d2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 3);
            __m128 _a3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic);
            __m128 _b3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 1);
            __m128 _c3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 2);
            __m128 _d3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 3);
            for (int oc = 0; oc < ch_out; oc+=4) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                const float* w_ptr1 = weight + (ic + 1) * ch_out + oc;
                const float* w_ptr2 = weight + (ic + 2) * ch_out + oc;
                const float* w_ptr3 = weight + (ic + 3) * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;
                float* y_ptr1 = output + (bs + 1) * ch_out + oc;
                float* y_ptr2 = output + (bs + 2) * ch_out + oc;
                float* y_ptr3 = output + (bs + 3) * ch_out + oc;


                __m128 _w0 = _mm_load_ps(w_ptr0);
                __m128 _w1 = _mm_load_ps(w_ptr1);
                __m128 _w2 = _mm_load_ps(w_ptr2);
                __m128 _w3 = _mm_load_ps(w_ptr3);

//                __m256 _y1 = _mm256_loadu_ps(y_ptr1);
//                __m256 _y2 = _mm256_loadu_ps(y_ptr2);
//                __m256 _y3 = _mm256_loadu_ps(y_ptr3);


//                w_ptr0 += 8;
//                w_ptr1 += 8;
//                w_ptr2 += 8;
//                w_ptr3 += 8;

                // 下面这4句代码一共有16个积。
                // 观察到 _w2 的4个元素*(w_ptr2 + 0)、*(w_ptr2 + 1)、*(w_ptr2 + 2)、*(w_ptr2 + 3)都是和 _c0 相乘
                // 所以 _c0 经过广播 _mm_broadcast_ss() 获得
//                *(y_ptr0 + 0) += _a0 * *(w_ptr0 + 0) + \
//                                 _b0 * *(w_ptr1 + 0) + \
//                                 _c0 * *(w_ptr2 + 0) + \
//                                 _d0 * *(w_ptr3 + 0);
//                *(y_ptr0 + 1) += _a0 * *(w_ptr0 + 1) + \
//                                 _b0 * *(w_ptr1 + 1) + \
//                                 _c0 * *(w_ptr2 + 1) + \
//                                 _d0 * *(w_ptr3 + 1);
//                *(y_ptr0 + 2) += _a0 * *(w_ptr0 + 2) + \
//                                 _b0 * *(w_ptr1 + 2) + \
//                                 _c0 * *(w_ptr2 + 2) + \
//                                 _d0 * *(w_ptr3 + 2);
//                *(y_ptr0 + 3) += _a0 * *(w_ptr0 + 3) + \
//                                 _b0 * *(w_ptr1 + 3) + \
//                                 _c0 * *(w_ptr2 + 3) + \
//                                 _d0 * *(w_ptr3 + 3);

                // 现在拿到了16个积。
//                __m128 _a0w0 = _mm_mul_ps(_a0, _w0);
//                __m128 _b0w1 = _mm_mul_ps(_b0, _w1);
//                __m128 _c0w2 = _mm_mul_ps(_c0, _w2);   // _w2 的4个元素 都是和 _c0 相乘
//                __m128 _d0w3 = _mm_mul_ps(_d0, _w3);
                // 16个积累加到_y0的4个元素里。_a0w0、_b0w1、_c0w2、_d0w3逐元素相加，累加到_y0里
                __m128 _y0 = _mm_load_ps(y_ptr0);
                _y0 = _mm_fmadd_ps(_a0, _w0, _y0);
                _y0 = _mm_fmadd_ps(_b0, _w1, _y0);
                _y0 = _mm_fmadd_ps(_c0, _w2, _y0);
                _y0 = _mm_fmadd_ps(_d0, _w3, _y0);
                _mm_store_ps(y_ptr0, _y0);

                __m128 _y1 = _mm_load_ps(y_ptr1);
                _y1 = _mm_fmadd_ps(_a1, _w0, _y1);
                _y1 = _mm_fmadd_ps(_b1, _w1, _y1);
                _y1 = _mm_fmadd_ps(_c1, _w2, _y1);
                _y1 = _mm_fmadd_ps(_d1, _w3, _y1);
                _mm_store_ps(y_ptr1, _y1);

                __m128 _y2 = _mm_load_ps(y_ptr2);
                _y2 = _mm_fmadd_ps(_a2, _w0, _y2);
                _y2 = _mm_fmadd_ps(_b2, _w1, _y2);
                _y2 = _mm_fmadd_ps(_c2, _w2, _y2);
                _y2 = _mm_fmadd_ps(_d2, _w3, _y2);
                _mm_store_ps(y_ptr2, _y2);

                __m128 _y3 = _mm_load_ps(y_ptr3);
                _y3 = _mm_fmadd_ps(_a3, _w0, _y3);
                _y3 = _mm_fmadd_ps(_b3, _w1, _y3);
                _y3 = _mm_fmadd_ps(_c3, _w2, _y3);
                _y3 = _mm_fmadd_ps(_d3, _w3, _y3);
                _mm_store_ps(y_ptr3, _y3);
            }
        }
    }
}



template<typename data_t>
void matmul_block_pack_8_8_8_SIMD_no_consider_mod_x86_kernel(const int num_threads_, const data_t* input, const data_t* weight, data_t* output, int batch_size, int ch_in, int ch_out) {

    #pragma omp parallel for num_threads(num_threads_)
    for (int bs = 0; bs < batch_size; bs+=8) {
        for (int ic = 0; ic < ch_in; ic+=8) {
            __m256 _a0 = _mm256_broadcast_ss(input + bs * ch_in + ic);
            __m256 _b0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 1);
            __m256 _c0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 2);
            __m256 _d0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 3);
            __m256 _e0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 4);
            __m256 _f0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 5);
            __m256 _g0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 6);
            __m256 _h0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 7);

            __m256 _a1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic);
            __m256 _b1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 1);
            __m256 _c1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 2);
            __m256 _d1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 3);
            __m256 _e1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 4);
            __m256 _f1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 5);
            __m256 _g1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 6);
            __m256 _h1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 7);

            __m256 _a2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic);
            __m256 _b2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 1);
            __m256 _c2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 2);
            __m256 _d2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 3);
            __m256 _e2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 4);
            __m256 _f2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 5);
            __m256 _g2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 6);
            __m256 _h2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 7);

            __m256 _a3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic);
            __m256 _b3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 1);
            __m256 _c3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 2);
            __m256 _d3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 3);
            __m256 _e3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 4);
            __m256 _f3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 5);
            __m256 _g3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 6);
            __m256 _h3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 7);

            __m256 _a4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic);
            __m256 _b4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic + 1);
            __m256 _c4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic + 2);
            __m256 _d4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic + 3);
            __m256 _e4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic + 4);
            __m256 _f4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic + 5);
            __m256 _g4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic + 6);
            __m256 _h4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic + 7);

            __m256 _a5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic);
            __m256 _b5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic + 1);
            __m256 _c5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic + 2);
            __m256 _d5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic + 3);
            __m256 _e5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic + 4);
            __m256 _f5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic + 5);
            __m256 _g5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic + 6);
            __m256 _h5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic + 7);

            __m256 _a6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic);
            __m256 _b6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic + 1);
            __m256 _c6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic + 2);
            __m256 _d6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic + 3);
            __m256 _e6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic + 4);
            __m256 _f6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic + 5);
            __m256 _g6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic + 6);
            __m256 _h6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic + 7);

            __m256 _a7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic);
            __m256 _b7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic + 1);
            __m256 _c7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic + 2);
            __m256 _d7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic + 3);
            __m256 _e7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic + 4);
            __m256 _f7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic + 5);
            __m256 _g7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic + 6);
            __m256 _h7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic + 7);

            for (int oc = 0; oc < ch_out; oc+=8) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                const float* w_ptr1 = weight + (ic + 1) * ch_out + oc;
                const float* w_ptr2 = weight + (ic + 2) * ch_out + oc;
                const float* w_ptr3 = weight + (ic + 3) * ch_out + oc;
                const float* w_ptr4 = weight + (ic + 4) * ch_out + oc;
                const float* w_ptr5 = weight + (ic + 5) * ch_out + oc;
                const float* w_ptr6 = weight + (ic + 6) * ch_out + oc;
                const float* w_ptr7 = weight + (ic + 7) * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;
                float* y_ptr1 = output + (bs + 1) * ch_out + oc;
                float* y_ptr2 = output + (bs + 2) * ch_out + oc;
                float* y_ptr3 = output + (bs + 3) * ch_out + oc;
                float* y_ptr4 = output + (bs + 4) * ch_out + oc;
                float* y_ptr5 = output + (bs + 5) * ch_out + oc;
                float* y_ptr6 = output + (bs + 6) * ch_out + oc;
                float* y_ptr7 = output + (bs + 7) * ch_out + oc;

                __m256 _w0 = _mm256_loadu_ps(w_ptr0);
                __m256 _w1 = _mm256_loadu_ps(w_ptr1);
                __m256 _w2 = _mm256_loadu_ps(w_ptr2);
                __m256 _w3 = _mm256_loadu_ps(w_ptr3);
                __m256 _w4 = _mm256_loadu_ps(w_ptr4);
                __m256 _w5 = _mm256_loadu_ps(w_ptr5);
                __m256 _w6 = _mm256_loadu_ps(w_ptr6);
                __m256 _w7 = _mm256_loadu_ps(w_ptr7);


                __m256 _y0 = _mm256_loadu_ps(y_ptr0);
                _y0 = _mm256_fmadd_ps(_a0, _w0, _y0);
                _y0 = _mm256_fmadd_ps(_b0, _w1, _y0);
                _y0 = _mm256_fmadd_ps(_c0, _w2, _y0);
                _y0 = _mm256_fmadd_ps(_d0, _w3, _y0);
                _y0 = _mm256_fmadd_ps(_e0, _w4, _y0);
                _y0 = _mm256_fmadd_ps(_f0, _w5, _y0);
                _y0 = _mm256_fmadd_ps(_g0, _w6, _y0);
                _y0 = _mm256_fmadd_ps(_h0, _w7, _y0);
                _mm256_storeu_ps(y_ptr0, _y0);

                __m256 _y1 = _mm256_loadu_ps(y_ptr1);
                _y1 = _mm256_fmadd_ps(_a1, _w0, _y1);
                _y1 = _mm256_fmadd_ps(_b1, _w1, _y1);
                _y1 = _mm256_fmadd_ps(_c1, _w2, _y1);
                _y1 = _mm256_fmadd_ps(_d1, _w3, _y1);
                _y1 = _mm256_fmadd_ps(_e1, _w4, _y1);
                _y1 = _mm256_fmadd_ps(_f1, _w5, _y1);
                _y1 = _mm256_fmadd_ps(_g1, _w6, _y1);
                _y1 = _mm256_fmadd_ps(_h1, _w7, _y1);
                _mm256_storeu_ps(y_ptr1, _y1);

                __m256 _y2 = _mm256_loadu_ps(y_ptr2);
                _y2 = _mm256_fmadd_ps(_a2, _w0, _y2);
                _y2 = _mm256_fmadd_ps(_b2, _w1, _y2);
                _y2 = _mm256_fmadd_ps(_c2, _w2, _y2);
                _y2 = _mm256_fmadd_ps(_d2, _w3, _y2);
                _y2 = _mm256_fmadd_ps(_e2, _w4, _y2);
                _y2 = _mm256_fmadd_ps(_f2, _w5, _y2);
                _y2 = _mm256_fmadd_ps(_g2, _w6, _y2);
                _y2 = _mm256_fmadd_ps(_h2, _w7, _y2);
                _mm256_storeu_ps(y_ptr2, _y2);

                __m256 _y3 = _mm256_loadu_ps(y_ptr3);
                _y3 = _mm256_fmadd_ps(_a3, _w0, _y3);
                _y3 = _mm256_fmadd_ps(_b3, _w1, _y3);
                _y3 = _mm256_fmadd_ps(_c3, _w2, _y3);
                _y3 = _mm256_fmadd_ps(_d3, _w3, _y3);
                _y3 = _mm256_fmadd_ps(_e3, _w4, _y3);
                _y3 = _mm256_fmadd_ps(_f3, _w5, _y3);
                _y3 = _mm256_fmadd_ps(_g3, _w6, _y3);
                _y3 = _mm256_fmadd_ps(_h3, _w7, _y3);
                _mm256_storeu_ps(y_ptr3, _y3);

                __m256 _y4 = _mm256_loadu_ps(y_ptr4);
                _y4 = _mm256_fmadd_ps(_a4, _w0, _y4);
                _y4 = _mm256_fmadd_ps(_b4, _w1, _y4);
                _y4 = _mm256_fmadd_ps(_c4, _w2, _y4);
                _y4 = _mm256_fmadd_ps(_d4, _w3, _y4);
                _y4 = _mm256_fmadd_ps(_e4, _w4, _y4);
                _y4 = _mm256_fmadd_ps(_f4, _w5, _y4);
                _y4 = _mm256_fmadd_ps(_g4, _w6, _y4);
                _y4 = _mm256_fmadd_ps(_h4, _w7, _y4);
                _mm256_storeu_ps(y_ptr4, _y4);

                __m256 _y5 = _mm256_loadu_ps(y_ptr5);
                _y5 = _mm256_fmadd_ps(_a5, _w0, _y5);
                _y5 = _mm256_fmadd_ps(_b5, _w1, _y5);
                _y5 = _mm256_fmadd_ps(_c5, _w2, _y5);
                _y5 = _mm256_fmadd_ps(_d5, _w3, _y5);
                _y5 = _mm256_fmadd_ps(_e5, _w4, _y5);
                _y5 = _mm256_fmadd_ps(_f5, _w5, _y5);
                _y5 = _mm256_fmadd_ps(_g5, _w6, _y5);
                _y5 = _mm256_fmadd_ps(_h5, _w7, _y5);
                _mm256_storeu_ps(y_ptr5, _y5);

                __m256 _y6 = _mm256_loadu_ps(y_ptr6);
                _y6 = _mm256_fmadd_ps(_a6, _w0, _y6);
                _y6 = _mm256_fmadd_ps(_b6, _w1, _y6);
                _y6 = _mm256_fmadd_ps(_c6, _w2, _y6);
                _y6 = _mm256_fmadd_ps(_d6, _w3, _y6);
                _y6 = _mm256_fmadd_ps(_e6, _w4, _y6);
                _y6 = _mm256_fmadd_ps(_f6, _w5, _y6);
                _y6 = _mm256_fmadd_ps(_g6, _w6, _y6);
                _y6 = _mm256_fmadd_ps(_h6, _w7, _y6);
                _mm256_storeu_ps(y_ptr6, _y6);

                __m256 _y7 = _mm256_loadu_ps(y_ptr7);
                _y7 = _mm256_fmadd_ps(_a7, _w0, _y7);
                _y7 = _mm256_fmadd_ps(_b7, _w1, _y7);
                _y7 = _mm256_fmadd_ps(_c7, _w2, _y7);
                _y7 = _mm256_fmadd_ps(_d7, _w3, _y7);
                _y7 = _mm256_fmadd_ps(_e7, _w4, _y7);
                _y7 = _mm256_fmadd_ps(_f7, _w5, _y7);
                _y7 = _mm256_fmadd_ps(_g7, _w6, _y7);
                _y7 = _mm256_fmadd_ps(_h7, _w7, _y7);
                _mm256_storeu_ps(y_ptr7, _y7);
            }
        }
    }
}





template<typename data_t>
void matmul_block_pack_8_8_8_SIMD_consider_mod_x86_kernel(const int num_threads_, const data_t* input, const data_t* weight, data_t* output, int batch_size, int ch_in, int ch_out) {
    const int B_mod = batch_size % 8;
    const int I_mod = ch_in % 8;
    const int O_mod = ch_out % 8;
    const int pack4_offset = (batch_size / 8) * 8;
    const int pack1_offset = (batch_size / 4) * 4;
    #pragma omp parallel for num_threads(num_threads_)
    for (int bs = 0; bs < pack4_offset; bs+=8) {
        int ic = 0;
        for (; ic + 7 < ch_in; ic+=8) {
            __m256 _a0 = _mm256_broadcast_ss(input + bs * ch_in + ic);
            __m256 _b0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 1);
            __m256 _c0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 2);
            __m256 _d0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 3);
            __m256 _e0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 4);
            __m256 _f0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 5);
            __m256 _g0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 6);
            __m256 _h0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 7);

            __m256 _a1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic);
            __m256 _b1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 1);
            __m256 _c1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 2);
            __m256 _d1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 3);
            __m256 _e1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 4);
            __m256 _f1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 5);
            __m256 _g1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 6);
            __m256 _h1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 7);

            __m256 _a2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic);
            __m256 _b2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 1);
            __m256 _c2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 2);
            __m256 _d2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 3);
            __m256 _e2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 4);
            __m256 _f2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 5);
            __m256 _g2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 6);
            __m256 _h2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 7);

            __m256 _a3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic);
            __m256 _b3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 1);
            __m256 _c3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 2);
            __m256 _d3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 3);
            __m256 _e3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 4);
            __m256 _f3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 5);
            __m256 _g3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 6);
            __m256 _h3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 7);

            __m256 _a4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic);
            __m256 _b4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic + 1);
            __m256 _c4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic + 2);
            __m256 _d4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic + 3);
            __m256 _e4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic + 4);
            __m256 _f4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic + 5);
            __m256 _g4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic + 6);
            __m256 _h4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic + 7);

            __m256 _a5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic);
            __m256 _b5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic + 1);
            __m256 _c5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic + 2);
            __m256 _d5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic + 3);
            __m256 _e5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic + 4);
            __m256 _f5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic + 5);
            __m256 _g5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic + 6);
            __m256 _h5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic + 7);

            __m256 _a6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic);
            __m256 _b6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic + 1);
            __m256 _c6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic + 2);
            __m256 _d6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic + 3);
            __m256 _e6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic + 4);
            __m256 _f6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic + 5);
            __m256 _g6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic + 6);
            __m256 _h6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic + 7);

            __m256 _a7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic);
            __m256 _b7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic + 1);
            __m256 _c7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic + 2);
            __m256 _d7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic + 3);
            __m256 _e7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic + 4);
            __m256 _f7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic + 5);
            __m256 _g7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic + 6);
            __m256 _h7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic + 7);

            int oc = 0;
            for (; oc + 7 < ch_out; oc+=8) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                const float* w_ptr1 = weight + (ic + 1) * ch_out + oc;
                const float* w_ptr2 = weight + (ic + 2) * ch_out + oc;
                const float* w_ptr3 = weight + (ic + 3) * ch_out + oc;
                const float* w_ptr4 = weight + (ic + 4) * ch_out + oc;
                const float* w_ptr5 = weight + (ic + 5) * ch_out + oc;
                const float* w_ptr6 = weight + (ic + 6) * ch_out + oc;
                const float* w_ptr7 = weight + (ic + 7) * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;
                float* y_ptr1 = output + (bs + 1) * ch_out + oc;
                float* y_ptr2 = output + (bs + 2) * ch_out + oc;
                float* y_ptr3 = output + (bs + 3) * ch_out + oc;
                float* y_ptr4 = output + (bs + 4) * ch_out + oc;
                float* y_ptr5 = output + (bs + 5) * ch_out + oc;
                float* y_ptr6 = output + (bs + 6) * ch_out + oc;
                float* y_ptr7 = output + (bs + 7) * ch_out + oc;

                __m256 _w0 = _mm256_loadu_ps(w_ptr0);
                __m256 _w1 = _mm256_loadu_ps(w_ptr1);
                __m256 _w2 = _mm256_loadu_ps(w_ptr2);
                __m256 _w3 = _mm256_loadu_ps(w_ptr3);
                __m256 _w4 = _mm256_loadu_ps(w_ptr4);
                __m256 _w5 = _mm256_loadu_ps(w_ptr5);
                __m256 _w6 = _mm256_loadu_ps(w_ptr6);
                __m256 _w7 = _mm256_loadu_ps(w_ptr7);


                __m256 _y0 = _mm256_loadu_ps(y_ptr0);
                _y0 = _mm256_fmadd_ps(_a0, _w0, _y0);
                _y0 = _mm256_fmadd_ps(_b0, _w1, _y0);
                _y0 = _mm256_fmadd_ps(_c0, _w2, _y0);
                _y0 = _mm256_fmadd_ps(_d0, _w3, _y0);
                _y0 = _mm256_fmadd_ps(_e0, _w4, _y0);
                _y0 = _mm256_fmadd_ps(_f0, _w5, _y0);
                _y0 = _mm256_fmadd_ps(_g0, _w6, _y0);
                _y0 = _mm256_fmadd_ps(_h0, _w7, _y0);
                _mm256_storeu_ps(y_ptr0, _y0);

                __m256 _y1 = _mm256_loadu_ps(y_ptr1);
                _y1 = _mm256_fmadd_ps(_a1, _w0, _y1);
                _y1 = _mm256_fmadd_ps(_b1, _w1, _y1);
                _y1 = _mm256_fmadd_ps(_c1, _w2, _y1);
                _y1 = _mm256_fmadd_ps(_d1, _w3, _y1);
                _y1 = _mm256_fmadd_ps(_e1, _w4, _y1);
                _y1 = _mm256_fmadd_ps(_f1, _w5, _y1);
                _y1 = _mm256_fmadd_ps(_g1, _w6, _y1);
                _y1 = _mm256_fmadd_ps(_h1, _w7, _y1);
                _mm256_storeu_ps(y_ptr1, _y1);

                __m256 _y2 = _mm256_loadu_ps(y_ptr2);
                _y2 = _mm256_fmadd_ps(_a2, _w0, _y2);
                _y2 = _mm256_fmadd_ps(_b2, _w1, _y2);
                _y2 = _mm256_fmadd_ps(_c2, _w2, _y2);
                _y2 = _mm256_fmadd_ps(_d2, _w3, _y2);
                _y2 = _mm256_fmadd_ps(_e2, _w4, _y2);
                _y2 = _mm256_fmadd_ps(_f2, _w5, _y2);
                _y2 = _mm256_fmadd_ps(_g2, _w6, _y2);
                _y2 = _mm256_fmadd_ps(_h2, _w7, _y2);
                _mm256_storeu_ps(y_ptr2, _y2);

                __m256 _y3 = _mm256_loadu_ps(y_ptr3);
                _y3 = _mm256_fmadd_ps(_a3, _w0, _y3);
                _y3 = _mm256_fmadd_ps(_b3, _w1, _y3);
                _y3 = _mm256_fmadd_ps(_c3, _w2, _y3);
                _y3 = _mm256_fmadd_ps(_d3, _w3, _y3);
                _y3 = _mm256_fmadd_ps(_e3, _w4, _y3);
                _y3 = _mm256_fmadd_ps(_f3, _w5, _y3);
                _y3 = _mm256_fmadd_ps(_g3, _w6, _y3);
                _y3 = _mm256_fmadd_ps(_h3, _w7, _y3);
                _mm256_storeu_ps(y_ptr3, _y3);

                __m256 _y4 = _mm256_loadu_ps(y_ptr4);
                _y4 = _mm256_fmadd_ps(_a4, _w0, _y4);
                _y4 = _mm256_fmadd_ps(_b4, _w1, _y4);
                _y4 = _mm256_fmadd_ps(_c4, _w2, _y4);
                _y4 = _mm256_fmadd_ps(_d4, _w3, _y4);
                _y4 = _mm256_fmadd_ps(_e4, _w4, _y4);
                _y4 = _mm256_fmadd_ps(_f4, _w5, _y4);
                _y4 = _mm256_fmadd_ps(_g4, _w6, _y4);
                _y4 = _mm256_fmadd_ps(_h4, _w7, _y4);
                _mm256_storeu_ps(y_ptr4, _y4);

                __m256 _y5 = _mm256_loadu_ps(y_ptr5);
                _y5 = _mm256_fmadd_ps(_a5, _w0, _y5);
                _y5 = _mm256_fmadd_ps(_b5, _w1, _y5);
                _y5 = _mm256_fmadd_ps(_c5, _w2, _y5);
                _y5 = _mm256_fmadd_ps(_d5, _w3, _y5);
                _y5 = _mm256_fmadd_ps(_e5, _w4, _y5);
                _y5 = _mm256_fmadd_ps(_f5, _w5, _y5);
                _y5 = _mm256_fmadd_ps(_g5, _w6, _y5);
                _y5 = _mm256_fmadd_ps(_h5, _w7, _y5);
                _mm256_storeu_ps(y_ptr5, _y5);

                __m256 _y6 = _mm256_loadu_ps(y_ptr6);
                _y6 = _mm256_fmadd_ps(_a6, _w0, _y6);
                _y6 = _mm256_fmadd_ps(_b6, _w1, _y6);
                _y6 = _mm256_fmadd_ps(_c6, _w2, _y6);
                _y6 = _mm256_fmadd_ps(_d6, _w3, _y6);
                _y6 = _mm256_fmadd_ps(_e6, _w4, _y6);
                _y6 = _mm256_fmadd_ps(_f6, _w5, _y6);
                _y6 = _mm256_fmadd_ps(_g6, _w6, _y6);
                _y6 = _mm256_fmadd_ps(_h6, _w7, _y6);
                _mm256_storeu_ps(y_ptr6, _y6);

                __m256 _y7 = _mm256_loadu_ps(y_ptr7);
                _y7 = _mm256_fmadd_ps(_a7, _w0, _y7);
                _y7 = _mm256_fmadd_ps(_b7, _w1, _y7);
                _y7 = _mm256_fmadd_ps(_c7, _w2, _y7);
                _y7 = _mm256_fmadd_ps(_d7, _w3, _y7);
                _y7 = _mm256_fmadd_ps(_e7, _w4, _y7);
                _y7 = _mm256_fmadd_ps(_f7, _w5, _y7);
                _y7 = _mm256_fmadd_ps(_g7, _w6, _y7);
                _y7 = _mm256_fmadd_ps(_h7, _w7, _y7);
                _mm256_storeu_ps(y_ptr7, _y7);
            }
            for (; oc + 3 < ch_out; oc+=4) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                const float* w_ptr1 = weight + (ic + 1) * ch_out + oc;
                const float* w_ptr2 = weight + (ic + 2) * ch_out + oc;
                const float* w_ptr3 = weight + (ic + 3) * ch_out + oc;
                const float* w_ptr4 = weight + (ic + 4) * ch_out + oc;
                const float* w_ptr5 = weight + (ic + 5) * ch_out + oc;
                const float* w_ptr6 = weight + (ic + 6) * ch_out + oc;
                const float* w_ptr7 = weight + (ic + 7) * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;
                float* y_ptr1 = output + (bs + 1) * ch_out + oc;
                float* y_ptr2 = output + (bs + 2) * ch_out + oc;
                float* y_ptr3 = output + (bs + 3) * ch_out + oc;
                float* y_ptr4 = output + (bs + 4) * ch_out + oc;
                float* y_ptr5 = output + (bs + 5) * ch_out + oc;
                float* y_ptr6 = output + (bs + 6) * ch_out + oc;
                float* y_ptr7 = output + (bs + 7) * ch_out + oc;

                __m128 _w0 = _mm_load_ps(w_ptr0);
                __m128 _w1 = _mm_load_ps(w_ptr1);
                __m128 _w2 = _mm_load_ps(w_ptr2);
                __m128 _w3 = _mm_load_ps(w_ptr3);
                __m128 _w4 = _mm_load_ps(w_ptr4);
                __m128 _w5 = _mm_load_ps(w_ptr5);
                __m128 _w6 = _mm_load_ps(w_ptr6);
                __m128 _w7 = _mm_load_ps(w_ptr7);


                __m128 __a0 = _mm_broadcast_ss(input + bs * ch_in + ic);
                __m128 __b0 = _mm_broadcast_ss(input + bs * ch_in + ic + 1);
                __m128 __c0 = _mm_broadcast_ss(input + bs * ch_in + ic + 2);
                __m128 __d0 = _mm_broadcast_ss(input + bs * ch_in + ic + 3);
                __m128 __e0 = _mm_broadcast_ss(input + bs * ch_in + ic + 4);
                __m128 __f0 = _mm_broadcast_ss(input + bs * ch_in + ic + 5);
                __m128 __g0 = _mm_broadcast_ss(input + bs * ch_in + ic + 6);
                __m128 __h0 = _mm_broadcast_ss(input + bs * ch_in + ic + 7);

                __m128 __a1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic);
                __m128 __b1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 1);
                __m128 __c1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 2);
                __m128 __d1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 3);
                __m128 __e1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 4);
                __m128 __f1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 5);
                __m128 __g1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 6);
                __m128 __h1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 7);

                __m128 __a2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic);
                __m128 __b2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 1);
                __m128 __c2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 2);
                __m128 __d2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 3);
                __m128 __e2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 4);
                __m128 __f2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 5);
                __m128 __g2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 6);
                __m128 __h2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 7);

                __m128 __a3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic);
                __m128 __b3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 1);
                __m128 __c3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 2);
                __m128 __d3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 3);
                __m128 __e3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 4);
                __m128 __f3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 5);
                __m128 __g3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 6);
                __m128 __h3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 7);

                __m128 __a4 = _mm_broadcast_ss(input + (bs + 4) * ch_in + ic);
                __m128 __b4 = _mm_broadcast_ss(input + (bs + 4) * ch_in + ic + 1);
                __m128 __c4 = _mm_broadcast_ss(input + (bs + 4) * ch_in + ic + 2);
                __m128 __d4 = _mm_broadcast_ss(input + (bs + 4) * ch_in + ic + 3);
                __m128 __e4 = _mm_broadcast_ss(input + (bs + 4) * ch_in + ic + 4);
                __m128 __f4 = _mm_broadcast_ss(input + (bs + 4) * ch_in + ic + 5);
                __m128 __g4 = _mm_broadcast_ss(input + (bs + 4) * ch_in + ic + 6);
                __m128 __h4 = _mm_broadcast_ss(input + (bs + 4) * ch_in + ic + 7);

                __m128 __a5 = _mm_broadcast_ss(input + (bs + 5) * ch_in + ic);
                __m128 __b5 = _mm_broadcast_ss(input + (bs + 5) * ch_in + ic + 1);
                __m128 __c5 = _mm_broadcast_ss(input + (bs + 5) * ch_in + ic + 2);
                __m128 __d5 = _mm_broadcast_ss(input + (bs + 5) * ch_in + ic + 3);
                __m128 __e5 = _mm_broadcast_ss(input + (bs + 5) * ch_in + ic + 4);
                __m128 __f5 = _mm_broadcast_ss(input + (bs + 5) * ch_in + ic + 5);
                __m128 __g5 = _mm_broadcast_ss(input + (bs + 5) * ch_in + ic + 6);
                __m128 __h5 = _mm_broadcast_ss(input + (bs + 5) * ch_in + ic + 7);

                __m128 __a6 = _mm_broadcast_ss(input + (bs + 6) * ch_in + ic);
                __m128 __b6 = _mm_broadcast_ss(input + (bs + 6) * ch_in + ic + 1);
                __m128 __c6 = _mm_broadcast_ss(input + (bs + 6) * ch_in + ic + 2);
                __m128 __d6 = _mm_broadcast_ss(input + (bs + 6) * ch_in + ic + 3);
                __m128 __e6 = _mm_broadcast_ss(input + (bs + 6) * ch_in + ic + 4);
                __m128 __f6 = _mm_broadcast_ss(input + (bs + 6) * ch_in + ic + 5);
                __m128 __g6 = _mm_broadcast_ss(input + (bs + 6) * ch_in + ic + 6);
                __m128 __h6 = _mm_broadcast_ss(input + (bs + 6) * ch_in + ic + 7);

                __m128 __a7 = _mm_broadcast_ss(input + (bs + 7) * ch_in + ic);
                __m128 __b7 = _mm_broadcast_ss(input + (bs + 7) * ch_in + ic + 1);
                __m128 __c7 = _mm_broadcast_ss(input + (bs + 7) * ch_in + ic + 2);
                __m128 __d7 = _mm_broadcast_ss(input + (bs + 7) * ch_in + ic + 3);
                __m128 __e7 = _mm_broadcast_ss(input + (bs + 7) * ch_in + ic + 4);
                __m128 __f7 = _mm_broadcast_ss(input + (bs + 7) * ch_in + ic + 5);
                __m128 __g7 = _mm_broadcast_ss(input + (bs + 7) * ch_in + ic + 6);
                __m128 __h7 = _mm_broadcast_ss(input + (bs + 7) * ch_in + ic + 7);


                __m128 _y0 = _mm_load_ps(y_ptr0);
                _y0 = _mm_fmadd_ps(__a0, _w0, _y0);
                _y0 = _mm_fmadd_ps(__b0, _w1, _y0);
                _y0 = _mm_fmadd_ps(__c0, _w2, _y0);
                _y0 = _mm_fmadd_ps(__d0, _w3, _y0);
                _y0 = _mm_fmadd_ps(__e0, _w4, _y0);
                _y0 = _mm_fmadd_ps(__f0, _w5, _y0);
                _y0 = _mm_fmadd_ps(__g0, _w6, _y0);
                _y0 = _mm_fmadd_ps(__h0, _w7, _y0);
                _mm_store_ps(y_ptr0, _y0);

                __m128 _y1 = _mm_load_ps(y_ptr1);
                _y1 = _mm_fmadd_ps(__a1, _w0, _y1);
                _y1 = _mm_fmadd_ps(__b1, _w1, _y1);
                _y1 = _mm_fmadd_ps(__c1, _w2, _y1);
                _y1 = _mm_fmadd_ps(__d1, _w3, _y1);
                _y1 = _mm_fmadd_ps(__e1, _w4, _y1);
                _y1 = _mm_fmadd_ps(__f1, _w5, _y1);
                _y1 = _mm_fmadd_ps(__g1, _w6, _y1);
                _y1 = _mm_fmadd_ps(__h1, _w7, _y1);
                _mm_store_ps(y_ptr1, _y1);

                __m128 _y2 = _mm_load_ps(y_ptr2);
                _y2 = _mm_fmadd_ps(__a2, _w0, _y2);
                _y2 = _mm_fmadd_ps(__b2, _w1, _y2);
                _y2 = _mm_fmadd_ps(__c2, _w2, _y2);
                _y2 = _mm_fmadd_ps(__d2, _w3, _y2);
                _y2 = _mm_fmadd_ps(__e2, _w4, _y2);
                _y2 = _mm_fmadd_ps(__f2, _w5, _y2);
                _y2 = _mm_fmadd_ps(__g2, _w6, _y2);
                _y2 = _mm_fmadd_ps(__h2, _w7, _y2);
                _mm_store_ps(y_ptr2, _y2);

                __m128 _y3 = _mm_load_ps(y_ptr3);
                _y3 = _mm_fmadd_ps(__a3, _w0, _y3);
                _y3 = _mm_fmadd_ps(__b3, _w1, _y3);
                _y3 = _mm_fmadd_ps(__c3, _w2, _y3);
                _y3 = _mm_fmadd_ps(__d3, _w3, _y3);
                _y3 = _mm_fmadd_ps(__e3, _w4, _y3);
                _y3 = _mm_fmadd_ps(__f3, _w5, _y3);
                _y3 = _mm_fmadd_ps(__g3, _w6, _y3);
                _y3 = _mm_fmadd_ps(__h3, _w7, _y3);
                _mm_store_ps(y_ptr3, _y3);

                __m128 _y4 = _mm_load_ps(y_ptr4);
                _y4 = _mm_fmadd_ps(__a4, _w0, _y4);
                _y4 = _mm_fmadd_ps(__b4, _w1, _y4);
                _y4 = _mm_fmadd_ps(__c4, _w2, _y4);
                _y4 = _mm_fmadd_ps(__d4, _w3, _y4);
                _y4 = _mm_fmadd_ps(__e4, _w4, _y4);
                _y4 = _mm_fmadd_ps(__f4, _w5, _y4);
                _y4 = _mm_fmadd_ps(__g4, _w6, _y4);
                _y4 = _mm_fmadd_ps(__h4, _w7, _y4);
                _mm_store_ps(y_ptr4, _y4);

                __m128 _y5 = _mm_load_ps(y_ptr5);
                _y5 = _mm_fmadd_ps(__a5, _w0, _y5);
                _y5 = _mm_fmadd_ps(__b5, _w1, _y5);
                _y5 = _mm_fmadd_ps(__c5, _w2, _y5);
                _y5 = _mm_fmadd_ps(__d5, _w3, _y5);
                _y5 = _mm_fmadd_ps(__e5, _w4, _y5);
                _y5 = _mm_fmadd_ps(__f5, _w5, _y5);
                _y5 = _mm_fmadd_ps(__g5, _w6, _y5);
                _y5 = _mm_fmadd_ps(__h5, _w7, _y5);
                _mm_store_ps(y_ptr5, _y5);

                __m128 _y6 = _mm_load_ps(y_ptr6);
                _y6 = _mm_fmadd_ps(__a6, _w0, _y6);
                _y6 = _mm_fmadd_ps(__b6, _w1, _y6);
                _y6 = _mm_fmadd_ps(__c6, _w2, _y6);
                _y6 = _mm_fmadd_ps(__d6, _w3, _y6);
                _y6 = _mm_fmadd_ps(__e6, _w4, _y6);
                _y6 = _mm_fmadd_ps(__f6, _w5, _y6);
                _y6 = _mm_fmadd_ps(__g6, _w6, _y6);
                _y6 = _mm_fmadd_ps(__h6, _w7, _y6);
                _mm_store_ps(y_ptr6, _y6);

                __m128 _y7 = _mm_load_ps(y_ptr7);
                _y7 = _mm_fmadd_ps(__a7, _w0, _y7);
                _y7 = _mm_fmadd_ps(__b7, _w1, _y7);
                _y7 = _mm_fmadd_ps(__c7, _w2, _y7);
                _y7 = _mm_fmadd_ps(__d7, _w3, _y7);
                _y7 = _mm_fmadd_ps(__e7, _w4, _y7);
                _y7 = _mm_fmadd_ps(__f7, _w5, _y7);
                _y7 = _mm_fmadd_ps(__g7, _w6, _y7);
                _y7 = _mm_fmadd_ps(__h7, _w7, _y7);
                _mm_store_ps(y_ptr7, _y7);
            }
            for (; oc < ch_out; oc++) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                const float* w_ptr1 = weight + (ic + 1) * ch_out + oc;
                const float* w_ptr2 = weight + (ic + 2) * ch_out + oc;
                const float* w_ptr3 = weight + (ic + 3) * ch_out + oc;
                const float* w_ptr4 = weight + (ic + 4) * ch_out + oc;
                const float* w_ptr5 = weight + (ic + 5) * ch_out + oc;
                const float* w_ptr6 = weight + (ic + 6) * ch_out + oc;
                const float* w_ptr7 = weight + (ic + 7) * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;
                float* y_ptr1 = output + (bs + 1) * ch_out + oc;
                float* y_ptr2 = output + (bs + 2) * ch_out + oc;
                float* y_ptr3 = output + (bs + 3) * ch_out + oc;
                float* y_ptr4 = output + (bs + 4) * ch_out + oc;
                float* y_ptr5 = output + (bs + 5) * ch_out + oc;
                float* y_ptr6 = output + (bs + 6) * ch_out + oc;
                float* y_ptr7 = output + (bs + 7) * ch_out + oc;

                float _w0 = *(w_ptr0);
                float _w1 = *(w_ptr1);
                float _w2 = *(w_ptr2);
                float _w3 = *(w_ptr3);
                float _w4 = *(w_ptr4);
                float _w5 = *(w_ptr5);
                float _w6 = *(w_ptr6);
                float _w7 = *(w_ptr7);


                float __a0 = *(input + bs * ch_in + ic);
                float __b0 = *(input + bs * ch_in + ic + 1);
                float __c0 = *(input + bs * ch_in + ic + 2);
                float __d0 = *(input + bs * ch_in + ic + 3);
                float __e0 = *(input + bs * ch_in + ic + 4);
                float __f0 = *(input + bs * ch_in + ic + 5);
                float __g0 = *(input + bs * ch_in + ic + 6);
                float __h0 = *(input + bs * ch_in + ic + 7);

                float __a1 = *(input + (bs + 1) * ch_in + ic);
                float __b1 = *(input + (bs + 1) * ch_in + ic + 1);
                float __c1 = *(input + (bs + 1) * ch_in + ic + 2);
                float __d1 = *(input + (bs + 1) * ch_in + ic + 3);
                float __e1 = *(input + (bs + 1) * ch_in + ic + 4);
                float __f1 = *(input + (bs + 1) * ch_in + ic + 5);
                float __g1 = *(input + (bs + 1) * ch_in + ic + 6);
                float __h1 = *(input + (bs + 1) * ch_in + ic + 7);

                float __a2 = *(input + (bs + 2) * ch_in + ic);
                float __b2 = *(input + (bs + 2) * ch_in + ic + 1);
                float __c2 = *(input + (bs + 2) * ch_in + ic + 2);
                float __d2 = *(input + (bs + 2) * ch_in + ic + 3);
                float __e2 = *(input + (bs + 2) * ch_in + ic + 4);
                float __f2 = *(input + (bs + 2) * ch_in + ic + 5);
                float __g2 = *(input + (bs + 2) * ch_in + ic + 6);
                float __h2 = *(input + (bs + 2) * ch_in + ic + 7);

                float __a3 = *(input + (bs + 3) * ch_in + ic);
                float __b3 = *(input + (bs + 3) * ch_in + ic + 1);
                float __c3 = *(input + (bs + 3) * ch_in + ic + 2);
                float __d3 = *(input + (bs + 3) * ch_in + ic + 3);
                float __e3 = *(input + (bs + 3) * ch_in + ic + 4);
                float __f3 = *(input + (bs + 3) * ch_in + ic + 5);
                float __g3 = *(input + (bs + 3) * ch_in + ic + 6);
                float __h3 = *(input + (bs + 3) * ch_in + ic + 7);

                float __a4 = *(input + (bs + 4) * ch_in + ic);
                float __b4 = *(input + (bs + 4) * ch_in + ic + 1);
                float __c4 = *(input + (bs + 4) * ch_in + ic + 2);
                float __d4 = *(input + (bs + 4) * ch_in + ic + 3);
                float __e4 = *(input + (bs + 4) * ch_in + ic + 4);
                float __f4 = *(input + (bs + 4) * ch_in + ic + 5);
                float __g4 = *(input + (bs + 4) * ch_in + ic + 6);
                float __h4 = *(input + (bs + 4) * ch_in + ic + 7);

                float __a5 = *(input + (bs + 5) * ch_in + ic);
                float __b5 = *(input + (bs + 5) * ch_in + ic + 1);
                float __c5 = *(input + (bs + 5) * ch_in + ic + 2);
                float __d5 = *(input + (bs + 5) * ch_in + ic + 3);
                float __e5 = *(input + (bs + 5) * ch_in + ic + 4);
                float __f5 = *(input + (bs + 5) * ch_in + ic + 5);
                float __g5 = *(input + (bs + 5) * ch_in + ic + 6);
                float __h5 = *(input + (bs + 5) * ch_in + ic + 7);

                float __a6 = *(input + (bs + 6) * ch_in + ic);
                float __b6 = *(input + (bs + 6) * ch_in + ic + 1);
                float __c6 = *(input + (bs + 6) * ch_in + ic + 2);
                float __d6 = *(input + (bs + 6) * ch_in + ic + 3);
                float __e6 = *(input + (bs + 6) * ch_in + ic + 4);
                float __f6 = *(input + (bs + 6) * ch_in + ic + 5);
                float __g6 = *(input + (bs + 6) * ch_in + ic + 6);
                float __h6 = *(input + (bs + 6) * ch_in + ic + 7);

                float __a7 = *(input + (bs + 7) * ch_in + ic);
                float __b7 = *(input + (bs + 7) * ch_in + ic + 1);
                float __c7 = *(input + (bs + 7) * ch_in + ic + 2);
                float __d7 = *(input + (bs + 7) * ch_in + ic + 3);
                float __e7 = *(input + (bs + 7) * ch_in + ic + 4);
                float __f7 = *(input + (bs + 7) * ch_in + ic + 5);
                float __g7 = *(input + (bs + 7) * ch_in + ic + 6);
                float __h7 = *(input + (bs + 7) * ch_in + ic + 7);


                float _y0 = *(y_ptr0);
                _y0 += __a0 * _w0;
                _y0 += __b0 * _w1;
                _y0 += __c0 * _w2;
                _y0 += __d0 * _w3;
                _y0 += __e0 * _w4;
                _y0 += __f0 * _w5;
                _y0 += __g0 * _w6;
                _y0 += __h0 * _w7;
                *y_ptr0 = _y0;

                float _y1 = *(y_ptr1);
                _y1 += __a1 * _w0;
                _y1 += __b1 * _w1;
                _y1 += __c1 * _w2;
                _y1 += __d1 * _w3;
                _y1 += __e1 * _w4;
                _y1 += __f1 * _w5;
                _y1 += __g1 * _w6;
                _y1 += __h1 * _w7;
                *y_ptr1 = _y1;

                float _y2 = *(y_ptr2);
                _y2 += __a2 * _w0;
                _y2 += __b2 * _w1;
                _y2 += __c2 * _w2;
                _y2 += __d2 * _w3;
                _y2 += __e2 * _w4;
                _y2 += __f2 * _w5;
                _y2 += __g2 * _w6;
                _y2 += __h2 * _w7;
                *y_ptr2 = _y2;

                float _y3 = *(y_ptr3);
                _y3 += __a3 * _w0;
                _y3 += __b3 * _w1;
                _y3 += __c3 * _w2;
                _y3 += __d3 * _w3;
                _y3 += __e3 * _w4;
                _y3 += __f3 * _w5;
                _y3 += __g3 * _w6;
                _y3 += __h3 * _w7;
                *y_ptr3 = _y3;

                float _y4 = *(y_ptr4);
                _y4 += __a4 * _w0;
                _y4 += __b4 * _w1;
                _y4 += __c4 * _w2;
                _y4 += __d4 * _w3;
                _y4 += __e4 * _w4;
                _y4 += __f4 * _w5;
                _y4 += __g4 * _w6;
                _y4 += __h4 * _w7;
                *y_ptr4 = _y4;

                float _y5 = *(y_ptr5);
                _y5 += __a5 * _w0;
                _y5 += __b5 * _w1;
                _y5 += __c5 * _w2;
                _y5 += __d5 * _w3;
                _y5 += __e5 * _w4;
                _y5 += __f5 * _w5;
                _y5 += __g5 * _w6;
                _y5 += __h5 * _w7;
                *y_ptr5 = _y5;

                float _y6 = *(y_ptr6);
                _y6 += __a6 * _w0;
                _y6 += __b6 * _w1;
                _y6 += __c6 * _w2;
                _y6 += __d6 * _w3;
                _y6 += __e6 * _w4;
                _y6 += __f6 * _w5;
                _y6 += __g6 * _w6;
                _y6 += __h6 * _w7;
                *y_ptr6 = _y6;

                float _y7 = *(y_ptr7);
                _y7 += __a7 * _w0;
                _y7 += __b7 * _w1;
                _y7 += __c7 * _w2;
                _y7 += __d7 * _w3;
                _y7 += __e7 * _w4;
                _y7 += __f7 * _w5;
                _y7 += __g7 * _w6;
                _y7 += __h7 * _w7;
                *y_ptr7 = _y7;
            }
        }
        for (; ic + 3 < ch_in; ic+=4) {
            __m256 _a0 = _mm256_broadcast_ss(input + bs * ch_in + ic);
            __m256 _b0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 1);
            __m256 _c0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 2);
            __m256 _d0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 3);

            __m256 _a1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic);
            __m256 _b1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 1);
            __m256 _c1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 2);
            __m256 _d1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 3);

            __m256 _a2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic);
            __m256 _b2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 1);
            __m256 _c2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 2);
            __m256 _d2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 3);

            __m256 _a3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic);
            __m256 _b3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 1);
            __m256 _c3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 2);
            __m256 _d3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 3);

            __m256 _a4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic);
            __m256 _b4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic + 1);
            __m256 _c4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic + 2);
            __m256 _d4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic + 3);

            __m256 _a5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic);
            __m256 _b5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic + 1);
            __m256 _c5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic + 2);
            __m256 _d5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic + 3);

            __m256 _a6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic);
            __m256 _b6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic + 1);
            __m256 _c6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic + 2);
            __m256 _d6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic + 3);

            __m256 _a7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic);
            __m256 _b7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic + 1);
            __m256 _c7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic + 2);
            __m256 _d7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic + 3);

            int oc = 0;
            for (; oc + 7 < ch_out; oc+=8) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                const float* w_ptr1 = weight + (ic + 1) * ch_out + oc;
                const float* w_ptr2 = weight + (ic + 2) * ch_out + oc;
                const float* w_ptr3 = weight + (ic + 3) * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;
                float* y_ptr1 = output + (bs + 1) * ch_out + oc;
                float* y_ptr2 = output + (bs + 2) * ch_out + oc;
                float* y_ptr3 = output + (bs + 3) * ch_out + oc;
                float* y_ptr4 = output + (bs + 4) * ch_out + oc;
                float* y_ptr5 = output + (bs + 5) * ch_out + oc;
                float* y_ptr6 = output + (bs + 6) * ch_out + oc;
                float* y_ptr7 = output + (bs + 7) * ch_out + oc;

                __m256 _w0 = _mm256_loadu_ps(w_ptr0);
                __m256 _w1 = _mm256_loadu_ps(w_ptr1);
                __m256 _w2 = _mm256_loadu_ps(w_ptr2);
                __m256 _w3 = _mm256_loadu_ps(w_ptr3);


                __m256 _y0 = _mm256_loadu_ps(y_ptr0);
                _y0 = _mm256_fmadd_ps(_a0, _w0, _y0);
                _y0 = _mm256_fmadd_ps(_b0, _w1, _y0);
                _y0 = _mm256_fmadd_ps(_c0, _w2, _y0);
                _y0 = _mm256_fmadd_ps(_d0, _w3, _y0);
                _mm256_storeu_ps(y_ptr0, _y0);

                __m256 _y1 = _mm256_loadu_ps(y_ptr1);
                _y1 = _mm256_fmadd_ps(_a1, _w0, _y1);
                _y1 = _mm256_fmadd_ps(_b1, _w1, _y1);
                _y1 = _mm256_fmadd_ps(_c1, _w2, _y1);
                _y1 = _mm256_fmadd_ps(_d1, _w3, _y1);
                _mm256_storeu_ps(y_ptr1, _y1);

                __m256 _y2 = _mm256_loadu_ps(y_ptr2);
                _y2 = _mm256_fmadd_ps(_a2, _w0, _y2);
                _y2 = _mm256_fmadd_ps(_b2, _w1, _y2);
                _y2 = _mm256_fmadd_ps(_c2, _w2, _y2);
                _y2 = _mm256_fmadd_ps(_d2, _w3, _y2);
                _mm256_storeu_ps(y_ptr2, _y2);

                __m256 _y3 = _mm256_loadu_ps(y_ptr3);
                _y3 = _mm256_fmadd_ps(_a3, _w0, _y3);
                _y3 = _mm256_fmadd_ps(_b3, _w1, _y3);
                _y3 = _mm256_fmadd_ps(_c3, _w2, _y3);
                _y3 = _mm256_fmadd_ps(_d3, _w3, _y3);
                _mm256_storeu_ps(y_ptr3, _y3);

                __m256 _y4 = _mm256_loadu_ps(y_ptr4);
                _y4 = _mm256_fmadd_ps(_a4, _w0, _y4);
                _y4 = _mm256_fmadd_ps(_b4, _w1, _y4);
                _y4 = _mm256_fmadd_ps(_c4, _w2, _y4);
                _y4 = _mm256_fmadd_ps(_d4, _w3, _y4);
                _mm256_storeu_ps(y_ptr4, _y4);

                __m256 _y5 = _mm256_loadu_ps(y_ptr5);
                _y5 = _mm256_fmadd_ps(_a5, _w0, _y5);
                _y5 = _mm256_fmadd_ps(_b5, _w1, _y5);
                _y5 = _mm256_fmadd_ps(_c5, _w2, _y5);
                _y5 = _mm256_fmadd_ps(_d5, _w3, _y5);
                _mm256_storeu_ps(y_ptr5, _y5);

                __m256 _y6 = _mm256_loadu_ps(y_ptr6);
                _y6 = _mm256_fmadd_ps(_a6, _w0, _y6);
                _y6 = _mm256_fmadd_ps(_b6, _w1, _y6);
                _y6 = _mm256_fmadd_ps(_c6, _w2, _y6);
                _y6 = _mm256_fmadd_ps(_d6, _w3, _y6);
                _mm256_storeu_ps(y_ptr6, _y6);

                __m256 _y7 = _mm256_loadu_ps(y_ptr7);
                _y7 = _mm256_fmadd_ps(_a7, _w0, _y7);
                _y7 = _mm256_fmadd_ps(_b7, _w1, _y7);
                _y7 = _mm256_fmadd_ps(_c7, _w2, _y7);
                _y7 = _mm256_fmadd_ps(_d7, _w3, _y7);
                _mm256_storeu_ps(y_ptr7, _y7);
            }
            for (; oc + 3 < ch_out; oc+=4) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                const float* w_ptr1 = weight + (ic + 1) * ch_out + oc;
                const float* w_ptr2 = weight + (ic + 2) * ch_out + oc;
                const float* w_ptr3 = weight + (ic + 3) * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;
                float* y_ptr1 = output + (bs + 1) * ch_out + oc;
                float* y_ptr2 = output + (bs + 2) * ch_out + oc;
                float* y_ptr3 = output + (bs + 3) * ch_out + oc;
                float* y_ptr4 = output + (bs + 4) * ch_out + oc;
                float* y_ptr5 = output + (bs + 5) * ch_out + oc;
                float* y_ptr6 = output + (bs + 6) * ch_out + oc;
                float* y_ptr7 = output + (bs + 7) * ch_out + oc;

                __m128 _w0 = _mm_load_ps(w_ptr0);
                __m128 _w1 = _mm_load_ps(w_ptr1);
                __m128 _w2 = _mm_load_ps(w_ptr2);
                __m128 _w3 = _mm_load_ps(w_ptr3);

                __m128 __a0 = _mm_broadcast_ss(input + bs * ch_in + ic);
                __m128 __b0 = _mm_broadcast_ss(input + bs * ch_in + ic + 1);
                __m128 __c0 = _mm_broadcast_ss(input + bs * ch_in + ic + 2);
                __m128 __d0 = _mm_broadcast_ss(input + bs * ch_in + ic + 3);

                __m128 __a1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic);
                __m128 __b1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 1);
                __m128 __c1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 2);
                __m128 __d1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 3);

                __m128 __a2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic);
                __m128 __b2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 1);
                __m128 __c2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 2);
                __m128 __d2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 3);

                __m128 __a3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic);
                __m128 __b3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 1);
                __m128 __c3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 2);
                __m128 __d3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 3);

                __m128 __a4 = _mm_broadcast_ss(input + (bs + 4) * ch_in + ic);
                __m128 __b4 = _mm_broadcast_ss(input + (bs + 4) * ch_in + ic + 1);
                __m128 __c4 = _mm_broadcast_ss(input + (bs + 4) * ch_in + ic + 2);
                __m128 __d4 = _mm_broadcast_ss(input + (bs + 4) * ch_in + ic + 3);

                __m128 __a5 = _mm_broadcast_ss(input + (bs + 5) * ch_in + ic);
                __m128 __b5 = _mm_broadcast_ss(input + (bs + 5) * ch_in + ic + 1);
                __m128 __c5 = _mm_broadcast_ss(input + (bs + 5) * ch_in + ic + 2);
                __m128 __d5 = _mm_broadcast_ss(input + (bs + 5) * ch_in + ic + 3);

                __m128 __a6 = _mm_broadcast_ss(input + (bs + 6) * ch_in + ic);
                __m128 __b6 = _mm_broadcast_ss(input + (bs + 6) * ch_in + ic + 1);
                __m128 __c6 = _mm_broadcast_ss(input + (bs + 6) * ch_in + ic + 2);
                __m128 __d6 = _mm_broadcast_ss(input + (bs + 6) * ch_in + ic + 3);

                __m128 __a7 = _mm_broadcast_ss(input + (bs + 7) * ch_in + ic);
                __m128 __b7 = _mm_broadcast_ss(input + (bs + 7) * ch_in + ic + 1);
                __m128 __c7 = _mm_broadcast_ss(input + (bs + 7) * ch_in + ic + 2);
                __m128 __d7 = _mm_broadcast_ss(input + (bs + 7) * ch_in + ic + 3);


                __m128 _y0 = _mm_load_ps(y_ptr0);
                _y0 = _mm_fmadd_ps(__a0, _w0, _y0);
                _y0 = _mm_fmadd_ps(__b0, _w1, _y0);
                _y0 = _mm_fmadd_ps(__c0, _w2, _y0);
                _y0 = _mm_fmadd_ps(__d0, _w3, _y0);
                _mm_store_ps(y_ptr0, _y0);

                __m128 _y1 = _mm_load_ps(y_ptr1);
                _y1 = _mm_fmadd_ps(__a1, _w0, _y1);
                _y1 = _mm_fmadd_ps(__b1, _w1, _y1);
                _y1 = _mm_fmadd_ps(__c1, _w2, _y1);
                _y1 = _mm_fmadd_ps(__d1, _w3, _y1);
                _mm_store_ps(y_ptr1, _y1);

                __m128 _y2 = _mm_load_ps(y_ptr2);
                _y2 = _mm_fmadd_ps(__a2, _w0, _y2);
                _y2 = _mm_fmadd_ps(__b2, _w1, _y2);
                _y2 = _mm_fmadd_ps(__c2, _w2, _y2);
                _y2 = _mm_fmadd_ps(__d2, _w3, _y2);
                _mm_store_ps(y_ptr2, _y2);

                __m128 _y3 = _mm_load_ps(y_ptr3);
                _y3 = _mm_fmadd_ps(__a3, _w0, _y3);
                _y3 = _mm_fmadd_ps(__b3, _w1, _y3);
                _y3 = _mm_fmadd_ps(__c3, _w2, _y3);
                _y3 = _mm_fmadd_ps(__d3, _w3, _y3);
                _mm_store_ps(y_ptr3, _y3);

                __m128 _y4 = _mm_load_ps(y_ptr4);
                _y4 = _mm_fmadd_ps(__a4, _w0, _y4);
                _y4 = _mm_fmadd_ps(__b4, _w1, _y4);
                _y4 = _mm_fmadd_ps(__c4, _w2, _y4);
                _y4 = _mm_fmadd_ps(__d4, _w3, _y4);
                _mm_store_ps(y_ptr4, _y4);

                __m128 _y5 = _mm_load_ps(y_ptr5);
                _y5 = _mm_fmadd_ps(__a5, _w0, _y5);
                _y5 = _mm_fmadd_ps(__b5, _w1, _y5);
                _y5 = _mm_fmadd_ps(__c5, _w2, _y5);
                _y5 = _mm_fmadd_ps(__d5, _w3, _y5);
                _mm_store_ps(y_ptr5, _y5);

                __m128 _y6 = _mm_load_ps(y_ptr6);
                _y6 = _mm_fmadd_ps(__a6, _w0, _y6);
                _y6 = _mm_fmadd_ps(__b6, _w1, _y6);
                _y6 = _mm_fmadd_ps(__c6, _w2, _y6);
                _y6 = _mm_fmadd_ps(__d6, _w3, _y6);
                _mm_store_ps(y_ptr6, _y6);

                __m128 _y7 = _mm_load_ps(y_ptr7);
                _y7 = _mm_fmadd_ps(__a7, _w0, _y7);
                _y7 = _mm_fmadd_ps(__b7, _w1, _y7);
                _y7 = _mm_fmadd_ps(__c7, _w2, _y7);
                _y7 = _mm_fmadd_ps(__d7, _w3, _y7);
                _mm_store_ps(y_ptr7, _y7);
            }
            for (; oc < ch_out; oc++) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                const float* w_ptr1 = weight + (ic + 1) * ch_out + oc;
                const float* w_ptr2 = weight + (ic + 2) * ch_out + oc;
                const float* w_ptr3 = weight + (ic + 3) * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;
                float* y_ptr1 = output + (bs + 1) * ch_out + oc;
                float* y_ptr2 = output + (bs + 2) * ch_out + oc;
                float* y_ptr3 = output + (bs + 3) * ch_out + oc;
                float* y_ptr4 = output + (bs + 4) * ch_out + oc;
                float* y_ptr5 = output + (bs + 5) * ch_out + oc;
                float* y_ptr6 = output + (bs + 6) * ch_out + oc;
                float* y_ptr7 = output + (bs + 7) * ch_out + oc;

                float _w0 = *(w_ptr0);
                float _w1 = *(w_ptr1);
                float _w2 = *(w_ptr2);
                float _w3 = *(w_ptr3);

                float __a0 = *(input + bs * ch_in + ic);
                float __b0 = *(input + bs * ch_in + ic + 1);
                float __c0 = *(input + bs * ch_in + ic + 2);
                float __d0 = *(input + bs * ch_in + ic + 3);

                float __a1 = *(input + (bs + 1) * ch_in + ic);
                float __b1 = *(input + (bs + 1) * ch_in + ic + 1);
                float __c1 = *(input + (bs + 1) * ch_in + ic + 2);
                float __d1 = *(input + (bs + 1) * ch_in + ic + 3);

                float __a2 = *(input + (bs + 2) * ch_in + ic);
                float __b2 = *(input + (bs + 2) * ch_in + ic + 1);
                float __c2 = *(input + (bs + 2) * ch_in + ic + 2);
                float __d2 = *(input + (bs + 2) * ch_in + ic + 3);

                float __a3 = *(input + (bs + 3) * ch_in + ic);
                float __b3 = *(input + (bs + 3) * ch_in + ic + 1);
                float __c3 = *(input + (bs + 3) * ch_in + ic + 2);
                float __d3 = *(input + (bs + 3) * ch_in + ic + 3);

                float __a4 = *(input + (bs + 4) * ch_in + ic);
                float __b4 = *(input + (bs + 4) * ch_in + ic + 1);
                float __c4 = *(input + (bs + 4) * ch_in + ic + 2);
                float __d4 = *(input + (bs + 4) * ch_in + ic + 3);

                float __a5 = *(input + (bs + 5) * ch_in + ic);
                float __b5 = *(input + (bs + 5) * ch_in + ic + 1);
                float __c5 = *(input + (bs + 5) * ch_in + ic + 2);
                float __d5 = *(input + (bs + 5) * ch_in + ic + 3);

                float __a6 = *(input + (bs + 6) * ch_in + ic);
                float __b6 = *(input + (bs + 6) * ch_in + ic + 1);
                float __c6 = *(input + (bs + 6) * ch_in + ic + 2);
                float __d6 = *(input + (bs + 6) * ch_in + ic + 3);

                float __a7 = *(input + (bs + 7) * ch_in + ic);
                float __b7 = *(input + (bs + 7) * ch_in + ic + 1);
                float __c7 = *(input + (bs + 7) * ch_in + ic + 2);
                float __d7 = *(input + (bs + 7) * ch_in + ic + 3);


                float _y0 = *(y_ptr0);
                _y0 += __a0 * _w0;
                _y0 += __b0 * _w1;