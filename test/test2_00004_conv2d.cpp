#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <chrono>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
#include "../miemienet/miemienet.h"
#include "../miemiedet/miemiedet.h"

using namespace miemienet;
using namespace miemiedet;

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


class Model : public Layer
{
public:
    Model(int in_features, int out_features, int kernel_size, int stride, int padding, bool use_bias=true, int groups=1)
    {
        conv = new SNT Conv2d(in_features, out_features, kernel_size, stride, padding, 1, groups, use_bias);
        register_sublayer("conv", conv);
//        act = new SNT Activation("leakyrelu", 0.33, true);
//        register_sublayer("act", act);
    }
    ~Model()
    {
        delete conv;
//        delete act;
    }

    Conv2d* conv;
//    Activation* act;

    virtual Tensor* create_tensors(Tensor* x)
    {
        Tensor* y = conv->create_tensors(x);
//        y = act->create_tensors(y);
        return y;
    }

    virtual Tensor* feed_forward(Tensor* x)
    {
        Tensor* y = conv->feed_forward(x);
//        y = act->feed_forward(y);
        return y;
    }
private:
};




int main(int argc, char** argv)
{
/*
python build.py --platform LINUX