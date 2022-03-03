import os
import itertools
import copy


def gen_transpose_common():
    # 打印全排列，transpose op用到
    trans2d = list(itertools.permutations([0, 1]))
    trans3d = list(itertools.permutations([0, 1, 2]))
    trans4d = list(itertools.permutations([0, 1, 2, 3]))
    for trans2d_e in trans2d:
        print("%d%d" % (trans2d_e[0], trans2d_e[1]))
    print()
    for trans3d_e in trans3d:
        print("%d%d%d" % (trans3d_e[0], trans3d_e[1], trans3d_e[2]))
    print()
    for trans4d_e in trans4d:
        print("%d%d%d%d" % (trans4d_e[0], trans4d_e[1], trans4d_e[2], trans4d_e[3]))
    print()
    content_cpp = ''
    content_cpp_invoke = ''
    content_forward = ''

    # 2d
    D = ['H', 'W']
    d = ['h', 'w']
    kkk = 0
    for t in trans2d:
        template = 'template<typename data_t>\n' + \
                   'void transpose2d_%d%d_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int %s, int %s) {\n' % (t[0], t[1], D[0], D[1]) + \
                   '    // y[%s][%s] = x[%s][%s]\n' % (d[t[0]], d[t[1]], d[0], d[1]) + \
                   '    #pragma omp parallel for num_threads(num_threads_)\n' + \
                   '    for (int %s = 0; %s < %s; %s++) {\n' % (d[0], d[0], D[0], d[0]) + \
                   '        for (int %s = 0; %s < %s; %s++) {\n' % (d[1], d[1], D[1], d[1]) + \
                   '            y[(%s * %s) + %s] = x[(%s * %s) + %s];\n' % (d[t[0]], D[t[1]], d[t[1]],      d[0], D[1], d[1]) + \
                   '        }\n' + \
                   '    }\n' + \
                   '}\n'
        content_cpp += template + '\n'
        templat2 = '        else if (transpose_type == TRANS2D_%d%d) {\n' % (t[0], t[1]) + \
                   '            if (input->dims != %d) {\n' % (2, ) + \
                   '                printf("Error from transpose op, transpose_type == TRANS2D_%d%d, input->dims != %d\\n");\n' % (t[0], t[1], 2) + \
                   '                exit(1);\n' + \
                   '            }\n' + \
                   '            const int %s = input->shape->at(%d);\n' % (D[0], 0) + \
                   '            const int %s = input->shape->at(%d);\n' % (D[1], 1) + \
                   '            transpose2d_%d%d_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, %s, %s);\n' % (t[0], t[1], D[0], D[1]) + \
                   '        }\n'
        templat3 = '    else if (transpose_type == TRANS2D_%d%d) {\n' % (t[0], t[1]) + \
                   '        if (input->dims != %d) {\n' % (2, ) + \
                   '            printf("Error from transpose op, transpose_type == TRANS2D_%d%d, input->dims != %d\\n");\n' % (t[0], t[1], 2) + \
                   '            exit(1);\n' + \
                   '        }\n' + \
                   '        const int %s = input->shape->at(%d);\n' % (D[0], 0) + \
                   '        const int %s = input->shape->at(%d);\n' % (D[1], 1) + \
                   '        output = new SNT Tensor(MMSHAPE2D(%s, %s), FP32, false, false);\n' % (D[t[0]], D[t[1]]) + \
                   '    }\n'
        if kkk == 0:
            templat2 = templat2.replace("else if (", "if (")
            templat3 = templat3.replace("else if (", "if (")
        kkk += 1
        content_cpp_invoke += templat2
        content_forward += templat3

    # 3d
    D = ['N', 'H', 'W']
    d = ['n', 'h', 'w']
    for t in trans3d:
        template = 'template<typename data_t>\n' + \
                   'void transpose3d_%d%d%d_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int %s, int %s, int %s) {\n' % (t[0], t[1], t[2], D[0], D[1], D[2]) + \
                   '    // y[%s][%s][%s] = x[%s][%s][%s]\n' % (d[t[0]], d[t[1]], d[t[2]], d[0], d[1], d[2]) + \
                   '    #pragma omp parallel for num_threads(num_threads_)\n' + \
                   '    for (int %s = 0; %s < %s; %s++) {\n' % (d[0], d[0], D[0], d[0]) + \
                   '        for (int %s = 0; %s < %s; %s++) {\n' % (d[1], d[1], D[1], d[1]) + \
                   '            for (int %s = 0; %s < %s; %s++) {\n' % (d[2], d[2], D[2], d[2]) + \
                   '                y[((%s * %s) + %s) * %s + %s] = x[((%s * %s) + %s) * %s + %s];\n' % (d[t[0]], D[t[1]], d[t[1]], D[t[2]], d[t[2]],      d[0], D[1], d[1], D[2], d[2]) + \
                   '            }\n' + \
                   '        }\n' + \
                   '    }\n' + \
                   '}\n'
        content_cpp += template + '\n'
        templat2 = '        else if (transpose_type == TRANS3D_%d%d%d) {\n' % (t[0], t[1], t[2]) + \
                   '            if (input->dims != %d) {\n' % (3, ) + \
                   '                printf("Error from transpose op, transpose_type == TRANS3D_%d%d%d, input->dims != %d\\n");\n' % (t[0], t[1], t[2], 3) + \
                   '                exit(1);\n' + \
                   '            }\n' + \
                   '            const int %s = input->shape->at(%d);\n' % (D[0], 0) + \
                   '            const int %s = input->shape->at(%d);\n' % (D[1], 1) + \
                   '            const int %s = input->shape->at(%d);\n' % (D[2], 2) + \
                   '            transpose3d_%d%d%d_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, %s, %s, %s);\n' % (t[0], t[1], t[2], D[0], D[1], D[2]) + \
                   '        }\n'
        templat3 = '    else if (transpose_type == TRANS3D_%d%d%d) {\n' % (t[0], t[1], t[2]) + \
                   '        if (input->dims != %d) {\n' % (3, ) + \
                   '            printf("Error from transpose op, transpose_type == TRANS3D_%d%d%d, input->dims != %d\\n");\n' % (t[0], t[1], t[2], 3) + \
                   '            exit(1);\n' + \
                   '        }\n' + \
                   '        const int %s = input->shape->at(%d);\n' % (D[0], 0) + \
                   '        const int %s = input->shape->at(%d);\n' % (D[1], 1) + \
                   '        const int %s = input->shape->at(%d);\n' % (D[2], 2) + \
                   '        output = new SNT Tensor(MMSHAPE3D(%s, %s, %s), FP32, false, false);\n' % (D[t[0]], D[t[1]], D[t[2]]) + \
                   '    }\n'
        content_cpp_invoke += templat2
        content_forward += templat3

    # 4d
    D = ['N', 'C', 'H', 'W']
    d = ['n', 'c', 'h', 'w']
    for t in trans4d:
        template = 'template<typename data_t>\n' + \
                   'void transpose4d_%d%d%d%d_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int %s, int %s, int %s, int %s) {\n' % (t[0], t[1], t[2], t[3], D[0], D[1], D[2], D[3]) + \
                   '    // y[%s][%s][%s][%s] = x[%s][%s][%s][%s]\n' % (d[t[0]], d[t[1]], d[t[2]], d[t[3]], d[0], d[1], d[2], d[3]) + \
                   '    #pragma omp parallel for num_threads(num_threads_)\n' + \
                   '    for (int %s = 0; %s < %s; %s++) {\n' % (d[0], d[0], D[0], d[0]) + \
                   '        for (int %s = 0; %s < %s; %s++) {\n' % (d[1], d[1], D[1], d[1]) + \
                   '            for (int %s = 0; %s < %s; %s++) {\n' % (d[2], d[2], D[2], d[2]) + \
                   '                for (int %s = 0; %s < %s; %s++) {\n' % (d[3], d[3], D[3], d[3]) + \
                   '                    y[(((%s * %s) + %s) * %s + %s) * %s + %s] = x[(((%s * %s) + %s) * %s + %s) * %s + %s];\n' % (d[t[0]], D[t[1]], d[t[1]], D[t[2]], d[t[2]], D[t[3]], d[t[3]],      d[0], D[1], d[1], D[2], d[2], D[3], d[3]) + \
                   '                }\n' + \
                   '            }\n' + \
                   '        }\n' + \
                   '    }\n' + \
                   '}\n'
        content_cpp += template + '\n'
        templat2 = '        else if (transpose_type == TRANS4D_%d%d%d%d) {\n' % (t[0], t[1], t[2], t[3]) + \
                   '            if (input->dims != %d) {\n' % (4, ) + \
                   '                printf("Error from transpose op, transpose_type == TRANS4D_%d%d%d%d, input->dims != %d\\n");\n' % (t[0], t[1], t[2], t[3], 4) + \
                   '                exit(1);\n' + \
                   '            }\n' + \
                   '            const int %s = input->shape->at(%d);\n' % (D[0], 0) + \
                   '            const int %s = input->shape->at(%d);\n' % (D[1], 1) + \
                   '            const int %s = input->shape->at(%d);\n' % (D[2], 2) + \
                   '            const int %s = input->shape->at(%d);\n' % (D[3], 3) + \
                   '            transpose4d_%d%d%d%d_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, %s, %s, %s, %s);\n' % (t[0], t[1], t[2], t[3], D[0], D[1], D[2], D[3]) + \
                   '        }\n'
        templat3 = '    else if (transpose_type == TRANS4D_%d%d%d%d) {\n' % (t[0], t[1], t[2], t[3]) + \
                   '        if (input->dims != %d) {\n' % (4, ) + \
                   '            printf("Error from transpose op, transpose_type == TRANS4D_%d%d%d%d, input->dims != %d\\n");\n' % (t[0], t[1], t[2], t[3], 4) + \
                   '            exit(1);\n' + \
                   '        }\n' + \
                   '        const int %s = input->shape->at(%d);\n' % (D[0], 0) + \
                   '        const int %s = input->shape->at(%d);\n' % (D[1], 1) + \
                   '        const int %s = input->shape->at(%d);\n' % (D[2], 2) + \
                   '        const int %s = input->shape->at(%d);\n' % (D[3], 3) + \
                   '        output = new SNT Tensor(MMSHAPE4D(%s, %s, %s, %s), FP32, false, false);\n' % (D[t[0]], D[t[1]], D[t[2]], D[t[3]]) + \
                   '    }\n'
        content_cpp_invoke += templat2
        content_forward += templat3

    with open('gen_code_cpp.txt', 'w', encoding='utf-8') as f:
        f.write(content_cpp)
        f.close()
    with open('gen_code_cpp_invoke.txt', 'w', encoding='utf-8') as f:
        f.write(content_cpp_invoke)
        f.close()


    content_x86 = content_cpp.replace("_cpp_kernel(", "_x86_kernel(") + '\n'
    content_x86_invoke = content_cpp_invoke.replace("_cpp_kernel<", "_x86_kernel<")
    content_x86 = '#if BACKEND_X86\n' + content_x86 + '#endif // BACKEND_X86\n'
    content_x86_invoke = '#if BACKEND_X86\n' + content_x86_invoke + '#endif // BACKEND_X86\n'
    with open('gen_code_x86.txt', 'w', encoding='utf-8') as f:
        f.write(content_x86)
        f.close()
    with open('gen_code_x86_invoke.txt', 'w', encoding='utf-8') as f:
        f.write(content_x86_invoke)
        f.close()

    # 直接替换代码
    '''
注解对配对使用：
// gen cpp code start
// gen cpp code end

// gen x86 code start
// gen x86 code end

// gen cpp invoke code start
// gen cpp invoke code end

// gen x86 invoke code start
// gen x86 invoke code end

// gen forward code start
// gen forward code end

    '''
    src_path = 'miemienet/nn/common/transpose_common.cpp'
    new_code = ''
    paste_zone = False
    with open(src_path, 'r', encoding='utf-8') as f:
        for line in f:
            if paste_zone:
                if 'gen cpp code end' in line:
                    paste_zone = False
                    new_code += line
                if 'gen x86 code end' in line:
                    paste_zone = False
                    new_code += line
                if 'gen cpp invoke code end' in line:
                    paste_zone = False
                    new_code += line
                if 'gen x86 invoke code end' in line:
                    paste_zone = False
                    new_code += line
            else:
                new_code += line
                if 'gen cpp code start' in line:
                    paste_zone = True
                    new_code += content_cpp
                if 'gen x86 code start' in line:
                    paste_zone = True
                    new_code += content_x86
                if 'gen cpp invoke code start' in line:
                    paste_zone = True
                    new_code += content_cpp_invoke
                if 'gen x86 invoke code start' in line:
                    paste_zone = True
                    new_code += content_x86_invoke
        f.close()
    with open(src_path, 'w', encoding='utf-8') as f:
        f.write(new_code)
        f.close()

    src_path = 'miemienet/nn/transpose.cpp'
    new_code = ''
    paste_zone = False
    with open(src_path, 'r', encoding='utf-8') as f:
        for line in f:
            if paste_zone:
                if 'gen forward code end' in line:
                    paste_zone = False
                    new_code += line
            else:
                new_code += line
                if 'gen forward code start' in line:
                    paste_zone = True
                    new_code += content_forward
        f.close()
    with open(src_path, 'w', encoding='utf-8') as f:
        f.write(new_code)
        f.close()
    print()

def elem_get_index(d, D, shape):
    real_shape = []
    real_i = []
    p = 0
    for s in shape:
        if s != '1':
            real_shape.append(D[p])
            real_i.append(d[p])
        p += 1
    if len(real_shape) == 4:
        _index = '((%s * %s + %s) * %s + %s) * %s + %s' % (real_i[0], real_shape[1], real_i[1], real_shape[2], real_i[2], real_shape[3], real_i[3])
    elif len(real_shape) == 3:
        _index = '(%s * %s + %s) * %s + %s' % (real_i[0], real_shape[1], real_i[1], real_shape[2], real_i[2])
    elif len(real_shape) == 2:
        _index = '%s * %s + %s' % (real_i[0], real_shape[1], real_i[1])
    elif len(real_shape) == 1:
        _index = '%s' % (real_i[0], )
    elif len(real_shape) == 0:
        _index = '0'
    return _index

def elem_get_out_index(d, D, shape1, shape2):
    real_shape = []
    real_i = []
    p = 0
    for s1 in shape1:
        s2 = shape2[p]
        if s1 != '1' or s2 != '1':
            real_shape.append(D[p])
            real_i.append(d[p])
        p += 1

    if len(real_shape) == 4:
        _index = '((%s * %s + %s) * %s + %s) * %s + %s' % (real_i[0], real_shape[1], real_i[1], real_shape[2], real_i[2], real_shape[3], real_i[3])
    elif len(real_shape) == 3:
        _index = '(%s * %s + %s) * %s + %s' % (real_i[0], real_shape[1], real_i[1], real_shape[2], real_i[2])
    elif len(real_shape) == 2:
        _index = '%s * %s + %s' % (real_i[0], real_shape[1], real_i[1])
    elif len(real_shape) == 1:
        _index = '%s' % (real_i[0], )
    elif len(real_shape) == 0:
        _index = '0'
    return _index

def gen_elementwise_common():
    ndim = 4
    D = ['N', 'C', 'H', 'W']
    d = ['n', 'c', 'h', 'w']
    tensor1_shapes = []
    tensor2_shapes = []
    for i in range(ndim + 1):
        tensor1_shape = list(itertools.combinations(range(ndim), i))
        for shape1 in tensor1_shape:
            cp1 = copy.deepcopy(D)
            for i1 in shape1:
                cp1[i1] = '1'
            tensor1_shapes.append(cp1)
            tensor2_shapes.append(copy.deepcopy(cp1))

    content_cpp_xop = ''
    content_cpp = ''
    content_cpp_invoke  = '        const int N0 = a->shape->at(0);\n'
    content_cpp_invoke += '        const int C0 = a->shape->at(1);\n'
    content_cpp_invoke += '        const int H0 = a->shape->at(2);\n'
    content_cpp_invoke += '        const int W0 = a->shape->at(3);\n'
    content_cpp_invoke += '        const int N1 = b->shape->at(0);\n'
    content_cpp_invoke += '        const int C1 = b->shape->at(1);\n'
    content_cpp_invoke += '        const int H1 = b->shape->at(2);\n'
    content_cpp_invoke += '        const int W1 = b->shape->at(3);\n'
    content_cpp_invoke += '        const int N = std::max(N0, N1);\n'
    content_cpp_invoke += '        const int C = std::max(C0, C1);\n'
    content_cpp_invoke += '        const int H = std::max(H0, H1);\n'
    content_cpp_invoke += '        const int W = std::max(W0, W1);\n'
    content_cpp_xop_invoke = ''
    kkk = 0
    for s1 in tensor1_shapes:
        for s2 in tensor2_shapes:
            # 是否过滤掉一些不常见情况。过滤掉的话可以减少代码量，提高编译速度。
            filt_ = False
            filt_ = True
            continue_ = True
            if filt_:
                # 添加白名单。白名单不会跳过
                if s1[0] == 'N' and s1[1] == 'C' and s1[2] == 'H' and s1[3] == 'W' and s2[0] == 'N' and s2[1] == 'C' and s2[2] == 'H' and s2[3] == 'W':
                    continue_ = False
                if s1[0] == '1' and s1[1] == '1' and s1[2] == '1' and s2[0] == '1' and s2[1] == '1' and s2[2] == '1':
                    continue_ = False
                if s1[0] == '1' and s1[1] == '1' and s2[0] == '1' and s2[1] == '1':
                    continue_ = False
                if s1[0] == '1' and s2[0] == '1':
                    continue_ = False
                if s1[0] == 'N' and s1[1] == 'C' and s1[2] == 'H' and s1[3] == 'W' and s2[0] == '1' and s2[1] == '1' and s2[2] == '1' and s2[3] == 'W':
                    continue_ = False
                if s1[0] == 'N' and s1[1] == 'C' and s1[2] == 'H' and s1[3] == 'W' and s2[0] == '1'