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
//    float M = 1.f;
    float M = 1.f / (float)numel;
    for (int i = 0; i < numel; i++)
    {
        diff += (x[i] - y[i]) * (x[i] - y[i]) * M;
    }
    return diff;
}

void load_from_txt(char* name, float* data_fp32, int numel, int num_threads_)
{
    FILE* fp = fopen(name, "r");
    if (!fp)
    {
        printf("file %s not exist.\n", name);
        exit(1);
    }

    int bytes = 0;
    bytes = sizeof(float) * numel;
    float* temp = (float*) malloc(bytes);
    const int N = 36;
    char buf[N];
    for (int i = 0; i < numel; i++)
    {
        fgets(buf, N, fp);
        float value = atof(buf);
        temp[i] = value;
    }

    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < numel; i++)
    {
        *(data_fp32 + i) = temp[i];
    }

    free(temp);
    temp = nullptr;
    fclose(fp);
}

void matmul1(float* A, float* B, float* C, int batch_size, int ch_in, int ch_out, int num_threads_)
{
    #pragma omp parallel for num_threads(num_threads_)
    for (int bs = 0; bs < batch_size; bs++) {
        for (int ic = 0; ic < ch_in; ic++) {
            const float _a = A[bs * ch_in + ic];
            for (int oc = 0; oc < ch_out; oc++) {
                C[bs * ch_out + oc] += _a * B[ic * ch_out + oc];
            }
        }
    }
}

void matmul2(float* A, float* B, float* C, int batch_size, int ch_in, int ch_out, int num_threads_)
{
    int elempack = 8;
    #pragma omp parallel for num_threads(num_threads_)
    for (int bs = 0; bs < batch_size; bs++) {
        const float* w_ptr = B;
        const float* x_ptr = A + bs * ch_in;
        for (int ic = 0; ic < ch_in; ic++) {
            float* out_ptr = C + bs * ch_out;
            __m256 _a = _mm256_broadcast_ss(x_ptr);
            for (int oc = 0; oc < ch_out; oc += elempack) {
                __m256 _b = _mm256_loadu_ps(w_ptr);
                __m256 _out = _mm256_loadu_ps(out_ptr);
                _out = _mm256_fmadd_ps(_a, _b, _out);
                _mm256_storeu_ps(out_ptr, _out);
                w_ptr += elempack;
                out_ptr += elempack;
            }
            x_ptr++;
        }
    }
}

int main(int argc, char** argv)
{
/*
g++ test/test2_00000_matmul.cpp -fopenmp -march=native -o test2_00000