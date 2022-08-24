#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include "../../framework/config.h"
#include "../../framework/tensor.h"
#include "../avgpool2d.h"
#include "avgpool2d_common.h"

#include "elementwise_common.h"
#include "matmul_common.h"
//#include "reduce_common.h"

NS_MM_F_BEGIN

template<typename data_t>
void avgpool2d_NCHW_cpp_kernel(const int num_threads_, const data_t* im, data_t* out, int num, int N, int out_H, int out_W, int in_C, int kH, int kW, int H, int W, int stride_h, int stride_w, int padding_h, int padding_w){
// 借鉴自 https://gitee.com/paddlepaddle/Paddle/blob/release/2.0/paddle/fluid/operators/math/pooling.cc
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int ic = 0; ic < in_C; ic++) {
            for (int oh = 0; oh < out_H; oh++) {
                for (int ow = 0; ow < out_W; ow++) {
                    // out.shape = [N, C, out_H, out_W]
                    // 以out（输出）的4D坐标为研究对象，求分量 n ic oh ow

                    // 求出对应池化窗左上角元素的y x坐标
                    int hstart = oh * stride_h - padding_h;
                    int wstart = ow * stride_w - padding_w;
                    int hend = (hstart + kH) > (H + padding_h) ? (H + padding_h) : (hstart + kH);
                    int wend = (wstart + kW) > (W + padding_w) ? (W + padding_w) : (wstart + kW);
                    int pool_size = (hend - hstart) * (wend - wstart);  // 平均池化使用

                    hstart = hstart < 0 ? 0 : hstart;
                    wstart = wstart < 0 ? 0 : wstart;
                    hend = hend > H ? H : hend;
                    wend = wend > W ? W : wend;

                    // 所求元素
                    float ele = 0.f;
                    for (int h = hstart; h < hend; h++) {
                        for (int w = wstart; w < wend; w++) {
                            ele += im[((n * in_C + ic) * H + h) * W + w];
                        }
                    }
                    bool exclusive = false;
                    bool adaptive = false;
                    if (exclusive || adaptive) {
                        pool_size = (hend - hstart) * (wend - wstart);
                    }
                    float factor = 1.f / float(pool_size);
                    out[((n * in_C + ic) * out_H + oh) * out_W + ow] = ele * factor;
                }
            }
        }
    }
}

template<typename data_t>
void avgpool2d_NHWC_cpp_kernel(const int num_threads_, const data_t* im, data_t* out, int num, int N, int out_H, int out_W, int in_C, int kH, int kW, int H, int W, int stride_h, int stride_w, int padding_h, int padding_w){
// 借鉴自 https://gitee.com/paddlepaddle/Paddle/blob/release/2.0/paddle/fluid/operators/math/pooling.cc
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < out_H