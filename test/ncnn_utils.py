#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description : 参考了部分onnx2ncnn.cpp的代码
#
# ================================================================
import struct
import numpy as np
import math


convert_to_fp16 = False


def bits_to_unsignedshort(bits):
    n = len(bits)
    sum = 0
    for i in range(n):
        if bits[i] == 1:
            sum += 2 ** (n - 1 - i)
    return sum


def int_to_bits(zhengshu, len=8):
    bits = np.zeros((len,), dtype=np.int32)
    if zhengshu > 0:
        zhengshu_bits_n = int(np.log2(zhengshu)) + 1
        zhengshu_bits = np.zeros((zhengshu_bits_n,), dtype=np.int32)
        temp = 