#ifndef __MAXPOOL2D_H__
#define __MAXPOOL2D_H__

#include "../macros.h"
#include "../framework/layer.h"

NS_MM_BEGIN

class MaxPool2d : public Layer
{
public:
    MaxPool2d(int kernel_