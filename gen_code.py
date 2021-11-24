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
                   '            if (input->dims != %d) {\n' % (2, ) 