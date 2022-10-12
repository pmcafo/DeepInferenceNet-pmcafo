
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