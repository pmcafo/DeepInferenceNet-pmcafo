
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "layer.h"

NS_MM_BEGIN

Layer::Layer()
{
    this->inplace = false;
    this->training = true;
    this->input_tensors = new std::vector<Tensor*>;
    this->temp_tensors = new std::vector<Tensor*>;
    this->temp2_tensors = new std::vector<Tensor*>;
    this->output_tensors = new std::vector<Tensor*>;
}

Layer::~Layer()
{
    // 倒序遍历，可以边遍历边删除
    for (int i = input_tensors->size() - 1; i >= 0; i--)
    {
        Tensor* tensor = input_tensors->at(i);
        if (tensor != nullptr)
        {
            tensor->referenceCount--;
            if (tensor->referenceCount <= 0){
                delete tensor;
            }
        }
        input_tensors->erase(input_tensors->begin() + i);
    }
    delete input_tensors;
    // 倒序遍历，可以边遍历边删除
    for (int i = temp_tensors->size() - 1; i >= 0; i--)
    {
        Tensor* tensor = temp_tensors->at(i);
        if (tensor != nullptr)
        {
            tensor->referenceCount--;
            if (tensor->referenceCount <= 0){
                delete tensor;
            }
        }
        temp_tensors->erase(temp_tensors->begin() + i);
    }
    delete temp_tensors;
    // 倒序遍历，可以边遍历边删除
    for (int i = temp2_tensors->size() - 1; i >= 0; i--)
    {
        Tensor* tensor = temp2_tensors->at(i);
        if (tensor != nullptr)
        {
            tensor->referenceCount--;
            if (tensor->referenceCount <= 0){
                delete tensor;
            }
        }
        temp2_tensors->erase(temp2_tensors->begin() + i);
    }
    delete temp2_tensors;
    // 倒序遍历，可以边遍历边删除
    for (int i = output_tensors->size() - 1; i >= 0; i--)
    {
        Tensor* tensor = output_tensors->at(i);
        if (tensor != nullptr)
        {
            tensor->referenceCount--;
            if (tensor->referenceCount <= 0){
                delete tensor;
            }
        }
        output_tensors->erase(output_tensors->begin() + i);
    }
    delete output_tensors;
}

void Layer::register_sublayer(char* name, Layer* layer)
{
    this->sublayer_names.push_back(name);
    this->sublayers.push_back(layer);
}

void Layer::train()
{
    training = true;
    for (int i = 0; i < this->sublayers.size(); i++)
    {
        this->sublayers[i]->train();
    }
}

void Layer::eval()
{
    training = false;
    for (int i = 0; i < this->sublayers.size(); i++)
    {
        this->sublayers[i]->eval();
    }
}

Tensor* Layer::register_buffer(char* name, std::vector<int>* shape, int dtype, bool init, float init_value)
{
    Tensor* buffer = new SNT Tensor(shape, dtype, true, init, init_value);
    buffer->is_buffer = true;       // layer->requires_grad_(bool)时，对buffer张量无效。
    this->params.push_back(buffer);
    this->param_names.push_back(name);