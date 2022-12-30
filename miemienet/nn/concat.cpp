
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include "concat.h"
#include "../framework/config.h"

#if BACKEND_X86
#include "common/concat_common.h"
#endif // BACKEND_X86

#if BACKEND_ARM
#include "common/concat_common.h"
#endif // BACKEND_ARM

NS_MM_BEGIN

Concat::Concat(int dim)
{
    this->dim = dim;
    this->son = nullptr;
}

Concat::~Concat()
{
    if (son != nullptr)
    {
        delete son;
    }
}

Tensor* Concat::create_tensors(Tensor* input1, Tensor* input2)
{
    if (input_tensors->size() == 0)
    {
        input1->referenceCount++;
        input_tensors->push_back(input1);
        input2->referenceCount++;
        input_tensors->push_back(input2);
