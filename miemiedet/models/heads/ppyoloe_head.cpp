#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <algorithm>
#include "ppyoloe_head.h"

namespace miemiedet {

ESEAttn::ESEAttn(int feat_channels, char* act_name)
{
    this->feat_channels = feat_channels;
    this->act_name = act_name;

    fc = new SNT Conv2d(feat_channels, feat_channels, 1, 1, 0, 1, 1, true);
    register_sublayer("fc", fc);

    this->conv = new SNT ConvBNLayer(feat_channels, feat_channels, 1, 1, 1, 0, act_name);
    register_sublayer("conv", this->conv);
}

ESEAttn::~ESEAttn()
{
    delete fc;
    delete conv;
}

Tensor* ESEAttn::create_tensors(Tensor* feat, Tensor* avg_feat)
{
    // 看PPYOLOEHead的代码，后面还会再次使用feat和avg_feat，所以不能用inpalce（省内存）的方式求输出。
    // avg_feat肯定不会被修改，因为Conv2d层一定会新建结果张量weight。
    // weight进行sigmoid激活可以就地修改，不新建张量。
    // weight 和 feat逐元素相乘，需要新建1个张量feat_weight保存结果。
    Tensor* weight = fc->create_tensors(avg_feat);

    std::vector<int>* _shape = feat->clone_shape();
    Tensor* feat_weight = new SNT Tensor(_shape, FP32, false, false);
    feat_weight->referenceCount++;
    temp_tensors->push_back(feat_weight);

    Tensor* y = conv->create_tensors(feat_weight);
    return y;
}

Tensor* ESEAttn::feed_forward(Tensor* feat, Tensor* avg_feat)
{
    // 看PPYOLOEHead的代码，后面还会再次使用feat和avg_feat，所以不能用inpalce（省内存）的方式求输出。
    // avg_feat肯定不会被修改，因为Conv2d层一定会新建结果张量weight。
 