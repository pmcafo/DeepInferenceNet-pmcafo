
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include "../../framework/config.h"
#include "../../framework/tensor.h"
#include "elementwise_common.h"

#if BACKEND_X86
#include <immintrin.h>
#endif // BACKEND_X86

#if BACKEND_ARM
//#include <arm_neon.h>
#endif // BACKEND_ARM

NS_MM_F_BEGIN


// gen cpp code start
template<typename data_t>
void elem4d_NCHW_add_NCHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] + y[((n * C + c) * H + h) * W + w];
                }
            }
        }
    }

    // x86
//    const int elempack = 8;
//    const int num_packs = num / elempack;
//    #pragma omp parallel for num_threads(num_threads_)
//    for (int pid = 0; pid < num_packs; pid++) {
//        const float* x_ptr = x + pid * elempack;
//        const float* y_ptr = y + pid * elempack;
//        float* z_ptr = z + pid * elempack;
//        __m256 _a = _mm256_loadu_ps(x_ptr);
//        __m256 _b = _mm256_loadu_ps(y_ptr);
//        __m256 _out = _mm256_add_ps(_a, _b);
//        _mm256_storeu_ps(z_ptr, _out);
//    }
//    int offset_ = num_packs * elempack;
//    if (num - offset_ >= 4)
//    {
//        const float* x_ptr = x + offset_;
//        const float* y_ptr = y + offset_;
//        float* z_ptr = z + offset_;
//        __m128 _a = _mm_load_ps(x_ptr);
//        __m128 _b = _mm_load_ps(y_ptr);
//        __m128 _out = _mm_add_ps(_a, _b);
//        _mm_store_ps(z_ptr, _out);
//        offset_ += 4;
//    }
//    #pragma omp parallel for num_threads(num_threads_)
//    for (int i = offset_; i < num; i++) {
//        z[i] = x[i] + y[i];
//    }
}

template<typename data_t>
void elem4d_NCHW_add_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_add_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] + y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_add_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] + y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_add_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] + y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_add_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] + y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_add_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] + y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_add_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_add_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] + y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_add_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] + y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_add_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] + y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_add_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] + y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_add_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] + y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_add_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] + y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_add_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] + y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_add_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_add_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] + y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_add_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] + y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_add_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] + y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_add_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] + y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_add_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] + y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_add_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] + y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_add_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] + y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_add_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_add_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] + y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_add_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] + y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_add_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] + y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_add_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] + y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_add_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] + y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_add_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] + y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_add_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] + y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_add_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_add_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] + y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_add_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] + y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_add_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] + y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_add_N11W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = x[n * W + w] + y[n * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_add_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = x[n * W + w] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_add_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[w] + y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_add_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[w] + y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_add_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[w] + y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_add_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[w] + y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_add_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[w] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_add_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[w] + y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_add_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[w] + y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_add_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[w] + y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_add_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h] + y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_add_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h] + y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_add_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h] + y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_add_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[h] + y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_add_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_add_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[h] + y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_add_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[h] + y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_add_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[h] + y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_add_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c] + y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_add_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c] + y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_add_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c] + y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_add_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c] + y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_add_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_add_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c] + y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_add_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[c] + y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_add_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[c] + y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_add_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[0] + y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_add_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[0] + y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_add_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[0] + y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_add_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[0] + y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_add_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[0] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_add_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[0] + y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_add_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[0] + y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_add_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[0] = x[0] + y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_sub_NCHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] - y[((n * C + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_sub_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_sub_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] - y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_sub_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] - y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_sub_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] - y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_sub_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] - y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_sub_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] - y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_sub_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_sub_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] - y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_sub_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] - y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_sub_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] - y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_sub_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] - y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_sub_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] - y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_sub_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] - y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_sub_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] - y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_sub_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_sub_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] - y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_sub_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] - y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_sub_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] - y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_sub_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] - y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_sub_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] - y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_sub_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] - y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_sub_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] - y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_sub_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_sub_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] - y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_sub_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] - y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_sub_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] - y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_sub_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] - y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_sub_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] - y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_sub_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] - y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_sub_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] - y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_sub_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_sub_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] - y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_sub_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] - y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_sub_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] - y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_sub_N11W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = x[n * W + w] - y[n * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_sub_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = x[n * W + w] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_sub_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[w] - y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_sub_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[w] - y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_sub_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[w] - y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_sub_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[w] - y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_sub_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[w] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_sub_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[w] - y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_sub_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[w] - y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_sub_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[w] - y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_sub_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h] - y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_sub_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h] - y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_sub_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h] - y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_sub_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[h] - y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_sub_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_sub_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[h] - y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_sub_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[h] - y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_sub_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[h] - y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_sub_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c] - y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_sub_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c] - y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_sub_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c] - y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_sub_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c] - y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_sub_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_sub_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c] - y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_sub_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[c] - y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_sub_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[c] - y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_sub_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[0] - y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_sub_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[0] - y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_sub_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[0] - y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_sub_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[0] - y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_sub_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[0] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_sub_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[0] - y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_sub_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[0] - y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_sub_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[0] = x[0] - y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_mul_NCHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] * y[((n * C + c) * H + h) * W + w];
                }
            }
        }
    }

    // x86
//    const int elempack = 8;
//    const int num_packs = num / elempack;
//    #pragma omp parallel for num_threads(num_threads_)
//    for (int pid = 0; pid < num_packs; pid++) {
//        const float* x_ptr = x + pid * elempack;
//        const float* y_ptr = y + pid * elempack;
//        float* z_ptr = z + pid * elempack;
//        __m256 _a = _mm256_loadu_ps(x_ptr);
//        __m256 _b = _mm256_loadu_ps(y_ptr);
//        __m256 _out = _mm256_mul_ps(_a, _b);
//        _mm256_storeu_ps(z_ptr, _out);
//    }
//    int offset_ = num_packs * elempack;
//    if (num - offset_ >= 4)
//    {
//        const float* x_ptr = x + offset_;
//        const float* y_ptr = y + offset_;
//        float* z_ptr = z + offset_;
//        __m128 _a = _mm_load_ps(x_ptr);
//        __m128 _b = _mm_load_ps(y_ptr);
//        __m128 _out = _mm_mul_ps(_a, _b);
//        _mm_store_ps(z_ptr, _out);
//        offset_ += 4;
//    }
//    #pragma omp parallel for num_threads(num_threads_)
//    for (int i = offset_; i < num; i++) {
//        z[i] = x[i] * y[i];
//    }
}

template<typename data_t>
void elem4d_NCHW_mul_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_mul_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] * y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_mul_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] * y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_mul_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] * y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_mul_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] * y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_mul_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] * y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_mul_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_mul_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] * y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_mul_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] * y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_mul_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] * y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_mul_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] * y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_mul_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] * y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_mul_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] * y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_mul_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] * y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_mul_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_mul_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] * y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_mul_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] * y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_mul_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] * y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_mul_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] * y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_mul_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] * y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_mul_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] * y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_mul_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] * y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_mul_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_mul_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] * y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_mul_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] * y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_mul_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] * y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_mul_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] * y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_mul_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] * y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_mul_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] * y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_mul_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] * y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_mul_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_mul_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] * y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_mul_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] * y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_mul_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] * y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_mul_N11W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = x[n * W + w] * y[n * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_mul_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = x[n * W + w] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_mul_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[w] * y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_mul_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[w] * y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_mul_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[w] * y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_mul_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[w] * y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_mul_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[w] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_mul_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[w] * y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_mul_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[w] * y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_mul_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[w] * y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_mul_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h] * y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_mul_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h] * y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_mul_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h] * y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_mul_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[h] * y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_mul_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_mul_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[h] * y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_mul_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[h] * y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_mul_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[h] * y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_mul_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c] * y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_mul_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c] * y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_mul_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c] * y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_mul_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c] * y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_mul_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_mul_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c] * y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_mul_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[c] * y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_mul_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[c] * y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_mul_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[0] * y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_mul_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[0] * y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_mul_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[0] * y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_mul_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[0] * y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_mul_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[0] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_mul_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[0] * y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_mul_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[0] * y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_mul_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[0] = x[0] * y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_div_NCHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] / y[((n * C + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_div_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_div_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] / y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_div_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] / y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_div_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] / y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_div_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] / y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_div_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] / y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_div_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_div_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] / y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_div_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] / y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_div_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] / y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_div_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] / y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_div_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] / y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_div_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] / y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_div_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] / y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_div_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_div_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] / y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_div_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] / y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_div_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] / y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_div_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] / y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_div_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] / y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_div_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] / y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_div_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] / y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_div_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_div_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] / y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_div_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] / y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_div_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] / y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_div_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] / y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_div_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] / y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_div_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] / y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_div_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] / y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_div_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_div_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] / y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_div_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] / y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_div_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] / y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_div_N11W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = x[n * W + w] / y[n * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_div_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = x[n * W + w] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_div_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[w] / y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_div_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[w] / y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_div_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[w] / y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_div_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[w] / y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_div_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[w] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_div_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[w] / y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_div_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[w] / y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_div_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[w] / y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_div_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h] / y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_div_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h] / y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_div_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h] / y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_div_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[h] / y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_div_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_div_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[h] / y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_div_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[h] / y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_div_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[h] / y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_div_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c] / y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_div_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c] / y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_div_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c] / y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_div_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c] / y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_div_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_div_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c] / y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_div_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[c] / y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_div_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[c] / y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_div_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[0] / y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_div_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[0] / y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_div_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[0] / y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_div_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[0] / y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_div_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[0] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_div_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[0] / y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_div_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[0] / y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_div_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[0] = x[0] / y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_min_NCHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = std::min(x[((n * C + c) * H + h) * W + w], y[((n * C + c) * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_min_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = std::min(x[((n * C + c) * H + h) * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_min_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = std::min(x[((n * C + c) * H + h) * W + w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_min_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[(c * H + h) * W + w], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_min_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[(c * H + h) * W + w], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_min_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[(c * H + h) * W + w], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_min_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[(c * H + h) * W + w], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_min_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[(c * H + h) * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_min_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[(c * H + h) * W + w], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_min_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[(c * H + h) * W + w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_min_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[(c * H + h) * W + w], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_min_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[h * W + w], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_min_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::min(x[h * W + w], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_min_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[h * W + w], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_min_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[h * W + w], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_min_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::min(x[h * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_min_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::min(x[h * W + w], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_min_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[h * W + w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_min_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::min(x[h * W + w], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_min_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c * W + w], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_min_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c * W + w], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_min_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::min(x[c * W + w], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_min_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c * W + w], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_min_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::min(x[c * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_min_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c * W + w], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_min_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::min(x[c * W + w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_min_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::min(x[c * W + w], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_min_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c * H + h], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_min_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c * H + h], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_min_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c * H + h], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_min_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::min(x[c * H + h], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_min_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c * H + h], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_min_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::min(x[c * H + h], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_min_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::min(x[c * H + h], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_min_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::min(x[c * H + h], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_min_N11W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = std::min(x[n * W + w], y[n * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_min_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = std::min(x[n * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_min_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[w], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_min_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::min(x[w], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_min_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::min(x[w], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_min_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[w], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_min_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = std::min(x[w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_min_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::min(x[w], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_min_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::min(x[w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_min_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = std::min(x[w], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_min_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[h], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_min_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::min(x[h], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_min_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[h], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_min_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::min(x[h], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_min_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::min(x[h], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_min_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = std::min(x[h], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_min_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::min(x[h], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_min_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = std::min(x[h], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_min_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_min_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_min_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::min(x[c], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_min_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::min(x[c], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_min_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::min(x[c], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_min_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::min(x[c], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_min_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = std::min(x[c], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_min_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = std::min(x[c], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_min_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[0], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_min_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::min(x[0], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_min_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::min(x[0], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_min_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::min(x[0], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_min_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = std::min(x[0], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_min_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = std::min(x[0], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_min_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = std::min(x[0], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_min_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[0] = std::min(x[0], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_max_NCHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = std::max(x[((n * C + c) * H + h) * W + w], y[((n * C + c) * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_max_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = std::max(x[((n * C + c) * H + h) * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_max_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = std::max(x[((n * C + c) * H + h) * W + w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_max_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[(c * H + h) * W + w], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_max_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[(c * H + h) * W + w], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_max_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[(c * H + h) * W + w], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_max_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[(c * H + h) * W + w], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_max_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[(c * H + h) * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_max_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[(c * H + h) * W + w], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_max_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[(c * H + h) * W + w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_max_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[(c * H + h) * W + w], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_max_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[h * W + w], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_max_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::max(x[h * W + w], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_max_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[h * W + w], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_max_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[h * W + w], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_max_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::max(x[h * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_max_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::max(x[h * W + w], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_max_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[h * W + w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_max_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::max(x[h * W + w], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_max_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[c * W + w], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_max_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[c * W + w], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_max_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::max(x[c * W + w], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_max_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[c * W + w], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_max_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::max(x[c * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_max_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[c * W + w], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_max_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::max(x[c * W + w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_max_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::max(x[c * W + w], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_max_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[c * H + h], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_max_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[c * H + h], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_max_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[c * H + h], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_max_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::max(x[c * H + h], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_max_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[c * H + h], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_max_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::max(x[c * H + h], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_max_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::max(x[c * H + h], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_max_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::max(x[c * H + h], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_max_N11W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = std::max(x[n * W + w], y[n * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_max_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = std::max(x[n * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_max_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[w], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_max_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::max(x[w], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_max_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::max(x[w], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_max_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[w], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_max_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = std::max(x[w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_max_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::max(x[w], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_max_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::max(x[w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_max_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = std::max(x[w], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_max_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[h], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_max_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::max(x[h], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_max_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[h], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_max_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::max(x[h], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_max_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)