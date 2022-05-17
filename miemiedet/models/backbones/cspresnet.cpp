
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <algorithm>
#include "cspresnet.h"

namespace miemiedet {

ConvBNLayer::ConvBNLayer(int ch_in, int ch_out, int filter_size, int stride, int groups, int padding, char* act_name)
{
    miemienet::Config* cfg = miemienet::Config::getInstance();
    this->fuse_conv_bn = cfg->fuse_conv_bn;
    this->ch_in = ch_in;
    this->ch_out = ch_out;
    this->filter_size = filter_size;
    this->stride = stride;
    this->groups = groups;
    this->padding = padding;
    this->act_name = act_name;
    if (act_name)
    {
        if (act_name == nullptr)
        {
            this->act = nullptr;
        }