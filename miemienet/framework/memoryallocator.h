#ifndef __MEMORYALLOCATOR_H__
#define __MEMORYALLOCATOR_H__

#include <vector>
#include "../macros.h"
#include "layer.h"

NS_MM_BEGIN

class MemoryAllocator
{
public:
    static MemoryAllocator* getInstance();
    static void destroyInstance();
    void reset();
//    float* assign_fp32_memory(const int id, const int bytes);
//    int* assign_int32_memory(const int id, const int bytes);
    float* assign_fp32_memory(const int bytes);
    int* assign_int32_memory(const int bytes);
private:
    MemoryAllocator();
    ~MemoryAllocator();
    static MemoryAl