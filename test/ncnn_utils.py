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
        temp = 0
        for i in range(zhengshu_bits_n):
            temp2 = 2 ** (zhengshu_bits_n - 1 - i)
            temp3 = temp2 + temp
            if temp3 == zhengshu:
                zhengshu_bits[i] = 1
                break
            elif temp3 > zhengshu:
                pass
            elif temp3 < zhengshu:
                zhengshu_bits[i] = 1
                temp = temp3
    else:
        zhengshu_bits_n = 1
        zhengshu_bits = np.zeros((zhengshu_bits_n,), dtype=np.int32)
    k = 0
    for i in range(len):
        if i < (len - zhengshu_bits_n):
            bits[i] = 0
        else:
            bits[i] = zhengshu_bits[k]
            k += 1
    return bits


def float_to_hex(v, fp16=True):
    bits = np.zeros((16,), dtype=np.int32)
    if not fp16:
        bits = np.zeros((32,), dtype=np.int32)
    if v == 0.0:
        return bits

    # fp16
    jiema = 15
    jiema_n = 5
    weishu_n = 10
    if not fp16:
        jiema = 127
        jiema_n = 8
        weishu_n = 23


    if v >= 0.0:
        bits[0] = 0
    else:
        bits[0] = 1
        v = 0.0 - v
    b = math.modf(v)
    xiaoshu = b[0]
    zhengshu = int(b[1])

    xiaoshu_bits_n = 32

    if zhengshu > 0:
        xiaoshu_bits_n = weishu_n
        zhengshu_bits_n = int(np.log2(zhengshu)) + 1
        zhengshu_bits = np.zeros((zhengshu_bits_n,), dtype=np.int32)
        temp = 0
        for i in range(zhengshu_bits_n):
            temp2 = 2 ** (zhengshu_bits_n - 1 - i)
            temp3 = temp2 + temp
            if temp3 == zhengshu:
                zhengshu_bits[i] = 1
                break
            elif temp3 > zhengshu:
                pass
            elif temp3 < zhengshu:
                zhengshu_bits[i] = 1
                temp = temp3
    else:
        zhengshu_bits_n = 1
        zhengshu_bits = np.zeros((zhengshu_bits_n,), dtype=np.int32)

    xiaoshu_bits = np.zeros((xiaoshu_bits_n,), dtype=np.int32)
    sum = 0.0
    half = 0.5
    for i in range(xiaoshu_bits_n):
        temp3 = sum + half
        if temp3 == xiaoshu:
            xiaoshu_bits[i] = 1
            break
        elif temp3 > xiaoshu:
            pass
        elif temp3 < xiaoshu:
            xiaoshu_bits[i] = 1
            sum = temp3
        half *= 0.5
    if len(zhengshu_bits) > 1:
        E = len(zhengshu_bits) - 1
        xiaoshu_bits = np.concatenate([zhengshu_bits[1:], xiaoshu_bits])
    elif len(zhengshu_bits) == 1:
        if zhengshu_bits[0] == 1:
            E = 0
        el