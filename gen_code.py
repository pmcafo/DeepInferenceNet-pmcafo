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
      