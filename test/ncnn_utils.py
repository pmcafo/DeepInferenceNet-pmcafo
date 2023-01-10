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
        elif zhengshu_bits[0] == 0:
            k = 0
            for i in range(xiaoshu_bits_n):
                if xiaoshu_bits[i] > 0:
                    k = i
                    break
            E = -1 - k
            xiaoshu_bits = xiaoshu_bits[k+1:]
    E += jiema
    jiema_bits = int_to_bits(E, len=jiema_n)
    bits[1:1+jiema_n] = jiema_bits
    k = 1 + jiema_n
    for i in range(weishu_n):
        if i >= len(xiaoshu_bits):
            bits[k] = 0
        else:
            bits[k] = xiaoshu_bits[i]
        k += 1
    return bits


def set_convert_to_fp16(_convert_to_fp16=False):
    global convert_to_fp16
    convert_to_fp16 = _convert_to_fp16


def bp_write_tag(bp):
    global convert_to_fp16
    if convert_to_fp16:
        s = struct.pack('i', 19950407)
    else:
        s = struct.pack('i', 0)
    bp.write(s)
    return bp


def bp_write_value(bp, value, force_fp32=False):
    global convert_to_fp16
    if force_fp32:
        s = struct.pack('f', value)
    else:
        if convert_to_fp16:
            bits_fp16 = float_to_hex(value)
            v_unsignedshort = bits_to_unsignedshort(bits_fp16)
            s = struct.pack('H', v_unsignedshort)
        else:
            s = struct.pack('f', value)
    bp.write(s)
    return bp


def create_new_param_bin(save_name, input_num):
    bp = open('%s.bin' % save_name, 'wb')
    pp = ''
    layer_id = 0
    tensor_id = 0
    bottom_names = []
    pp += 'Input\tlayer_%.8d\t0 %d' % (layer_id, input_num)
    for i in range(input_num):
        pp += ' tensor_%.8d' % tensor_id
        bottom_names.append('tensor_%.8d' % tensor_id)
        tensor_id += 1
    pp += '\n'
    layer_id += 1

    ncnn_data = {}
    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return ncnn_data, bottom_names


def save_param(save_name, ncnn_data, bottom_names, replace_input_names=[], replace_output_names=[]):
    assert isinstance(bottom_names, list)
    assert isinstance(replace_input_names, list)
    assert isinstance(replace_output_names, list)
    assert len(bottom_names) == len(replace_output_names)
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']
    for i in range(len(replace_input_names)):
        pp = pp.replace('tensor_%.8d' % (i,), replace_input_names[i])
    for i in range(len(replace_output_names)):
        pp = pp.replace(bottom_names[i], replace_output_names[i])
    pp = '7767517\n%d %d\n' % (layer_id, tensor_id) + pp
    with open('%s.param' % save_name, 'w', encoding='utf-8') as f:
        f.write(pp)
        f.close()
    return ncnn_data, bottom_names


def newest_bottom_names(ncnn_data):
    tensor_id = ncnn_data['tensor_id']
    bottom_names = ['tensor_%.8d' % (tensor_id - 1,), ]
    return bottom_names


def check_bottom_names(bottom_names):
    if isinstance(bottom_names, str):
        bottom_names = [bottom_names, ]
    elif isinstance(bottom_names, list):
        all_is_str = True
        num_input = len(bottom_names)
        for i in range(num_input):
            if not isinstance(bottom_names[i], str):
                all_is_str = False
                break
        if not all_is_str:
            raise NotImplementedError("bottom_names elements type not implemented.")
    else:
        raise NotImplementedError("bottom_names type not implemented.")
    return bottom_names


def create_top_names(ncnn_data, num):
    assert num >= 1
    tensor_id = ncnn_data['tensor_id']
    # if tensor_id == 242:
    #     print()
    top_names = []
    for i in range(num):
        top_names.append('tensor_%.8d' % (tensor_id + i,))
    return top_names


def pretty_format(ncnn_data, bottom_names):
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    lines = pp.split('\n')
    lines = lines[:-1]
    content = ''
    for i, line in enumerate(lines):
        ss = line.split()
        line2 = ''
        for kkk, s in enumerate(ss):
            if kkk == 0:
                line2 += "%-24s"%s
            elif kkk == 1:
                line2 += ' %-24s'%s
            elif kkk == 2:
                line2 += ' ' + s
            else:
                line2 += ' ' + s
        content += line2 + '\n'
    pp = content

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return bottom_names


def rename_tensor(ncnn_data, bottom_names):
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    tensor_id = 0
    lines = pp.split('\n')
    lines = lines[:-1]

    tensors_dic = {}
    tensor_id = 0
    for i, line in enumerate(lines):
        ss = line.split()
        in_num = int(ss[2])
        out_num = int(ss[3])
        p = 4
        for i1 in range(in_num):
            tensor_name = ss[p]
            if tensor_name not in tensors_dic.keys():
                aaaaaaaaaa = 'tensor_%.8d' % (tensor_id, )
                tensor_id += 1
                tensors_dic[tensor_name] = aaaaaaaaaa
            p += 1
        for i2 in range(out_num):
            tensor_name = ss[p]
            if tensor_name not in tensors_dic.keys():
                aaaaaaaaaa = 'tensor_%.8d' % (tensor_id, )
                tensor_id += 1
                tensors_dic[tensor_name] = aaaaaaaaaa
            p += 1
    content = ''
    for i, line in enumerate(lines):
        ss = line.split()
        in_num = int(ss[2])
        out_num = int(ss[3])
        p = 4 + in_num + out_num - 1
        for i1 in range(in_num):
            tensor_name = ss[p]
            ss[p] = tensors_dic[tensor_name]
            p -= 1
        for i2 in range(out_num):
            tensor_name = ss[p]
            ss[p] = tensors_dic[tensor_name]
            p -= 1
        line2 = ''
        for kkk, s in enumerate(ss):
            if kkk == 0:
                line2 += s
            elif kkk == 1:
                line2 += '\t' + s
            elif kkk == 2:
                line2 += '\t' + s
            else:
                line2 += ' ' + s
        content += line2 + '\n'
    pp = content

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    new_bottom_names = []
    for bname in bottom_names:
        new_bottom_names.append(tensors_dic[bname])
    return new_bottom_names


def split_input_tensor(ncnn_data, bottom_names):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    lines = pp.split('\n')
    lines = lines[:-1]

    # 统计张量被作为输入的次数
    tensor_as_input_count = {}
    for i, line in enumerate(lines):
        ss = line.split()
        in_num = int(ss[2])
        out_num = int(ss[3])
        p = 4
        for i1 in range(in_num):
            tensor_name = ss[p]
            if tensor_name not in tensor_as_input_count.keys():
                tensor_as_input_count[tensor_name] = 1
            else:
                tensor_as_input_count[tensor_name] += 1
            p += 1


    keys = tensor_as_input_count.keys()
    for split_tensor_name in keys:
        count = tensor_as_input_count[split_tensor_name]
        if count > 1:
            # 给网络插入1个Split层
            new_lines = []
            # 找到输出首次是split_tensor_name的层，在这个层的后面插入Split层
            find = False
            copy_i = 0
            for i, line in enumerate(lines):
                if not find:
                    ss = line.split()
                    in_num = int(ss[2])
           