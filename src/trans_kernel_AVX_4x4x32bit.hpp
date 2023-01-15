
#pragma once

#include "transpose_defs.hpp"

#if defined(__AVX__) || defined(__AVX2__)
#  include <immintrin.h>
#  define HAVE_AVX_4X4X32_KERNEL 1
#endif

#ifdef HAVE_AVX_4X4X32_KERNEL

#include <cstdint>

namespace transpose_kernels
{

template <class T, bool CONJUGATE_TPL = false>
struct AVX_4x4x32Kernel
{
  // requires AVX
  static constexpr unsigned KERNEL_SZ = 4;
  static constexpr bool HAS_AA = true;
  static constexpr bool CONJUGATE = CONJUGATE_TPL;
  static_assert( !CONJUGATE_TPL, "CONJUGATE is not supported by AVX_4x4x32Kernel" );
  using BaseType = float;

  struct MATRIX {
    union {
      __m128 m[4];
      __m256 n[2];
    };
  };

  // => looks 2 times slower than AVX_8x8x32Kernel or even SSE_4x4x32Kernel !

  // unaligned matrix pointers - or unaligned rowSizes
  ALWAYS_INLINE static void op_uu(const T * RESTRICT A_, T * RESTRICT B_, const unsigned rowSizeA, const unsigned rowSizeB) {
    const BaseType * RESTRICT A = reinterpret_cast<const BaseType * RESTRICT>(A_);
    BaseType * RESTRICT B = reinterpret_cast<BaseType * RESTRICT>(B_);
    static_assert( sizeof(T) == sizeof(BaseType), "" );
    static_assert( sizeof(T) == sizeof(int32_t), "" );

    MATRIX inp, result;

    inp.m[0] = _mm_loadu_ps(&A[0*rowSizeA]);
    inp.m[1] = _mm_loadu_ps(&A[1*rowSizeA]);
    inp.m[2] = _mm_loadu_ps(&A[2*rowSizeA]);
    inp.m[3] = _mm_loadu_ps(&A[3*rowSizeA]);

    // == myTranspose(MATRIX in)
    //   from https://stackoverflow.com/questions/16941098/fast-memory-transpose-with-sse-avx-and-openmp
    //   but made loads and stores unaligned
    // {
        // This takes 15 assembler instructions (compile not inline), 
        // and is faster than XMTranspose
        // Comes in like this  1  2  3  4  5  6  7  8
        //                     9 10 11 12 13 14 15 16
        //
        // Want the result     1  5  9 13  2  6 10 14
        //                     3  7 11 15  4  8 12 16

        __m256 t0 = _mm256_unpacklo_ps(inp.n[0], inp.n[1]);  // t0 =  1,  9,  2, 10,  5, 13,  6, 14
        __m256 t1 = _mm256_unpackhi_ps(inp.n[0], inp.n[1]);  // t1 =  3, 11,  4, 12,  7, 15,  8, 16

        __m256 t2 = _mm256_permute2f128_ps(t0, t1, 0x20);    // t2 =  1,  9,  2, 10,  3, 11,  4, 12
        __m256 t3 = _mm256_permute2f128_ps(t0, t1, 0x31);    // t3 =  5, 13,  6, 14,  7, 15,  8, 16

        __m256 t4 = _mm256_unpacklo_ps(t2, t3);              // t2 =  1,  5,  9, 13,  3,  7, 11, 15
        __m256 t5 = _mm256_unpackhi_ps(t2, t3);              // t3 =  2,  6, 10, 14,  4,  8, 12, 16

        result.n[0] = _mm256_permute2f128_ps(t4, t5, 0x20);  // t6 =  1,  5,  9, 13,  2,  6, 10, 14

        _mm_storeu_ps(&B[0*rowSizeB], result.m[0]);
        _mm_storeu_ps(&B[1*rowSizeB], result.m[1]);

        result.n[1] = _mm256_permute2f128_ps(t4, t5, 0x31);  // t7 =  3,  7, 11, 15,  4,  8, 12, 16

        _mm_storeu_ps(&B[2*rowSizeB], result.m[2]);
        _mm_storeu_ps(&B[3*rowSizeB], result.m[3]);
  }

  // aa: aligned to aligned
  // aligned matrix pointers - and aligned rowSizes (aligned to multiples of KERNEL_SZ x int32_t = 4*4 = 16 bytes)
  ALWAYS_INLINE static void op_aa(const T * RESTRICT A_, T * RESTRICT B_, const unsigned rowSizeA, const unsigned rowSizeB) {
    const BaseType * RESTRICT A = reinterpret_cast<const BaseType * RESTRICT>(A_);
    BaseType * RESTRICT B = reinterpret_cast<BaseType * RESTRICT>(B_);
    static_assert( sizeof(T) == sizeof(BaseType), "" );
    static_assert( sizeof(T) == sizeof(int32_t), "" );

    MATRIX inp, result;
    inp.m[0] = _mm_load_ps(&A[0*rowSizeA]);
    inp.m[1] = _mm_load_ps(&A[1*rowSizeA]);
    inp.m[2] = _mm_load_ps(&A[2*rowSizeA]);
    inp.m[3] = _mm_load_ps(&A[3*rowSizeA]);
        __m256 t0 = _mm256_unpacklo_ps(inp.n[0], inp.n[1]);  // t0 =  1,  9,  2, 10,  5, 13,  6, 14
        __m256 t1 = _mm256_unpackhi_ps(inp.n[0], inp.n[1]);  // t1 =  3, 11,  4, 12,  7, 15,  8, 16
        __m256 t2 = _mm256_permute2f128_ps(t0, t1, 0x20);    // t2 =  1,  9,  2, 10,  3, 11,  4, 12
        __m256 t3 = _mm256_permute2f128_ps(t0, t1, 0x31);    // t3 =  5, 13,  6, 14,  7, 15,  8, 16
        __m256 t4 = _mm256_unpacklo_ps(t2, t3);              // t2 =  1,  5,  9, 13,  3,  7, 11, 15
        __m256 t5 = _mm256_unpackhi_ps(t2, t3);              // t3 =  2,  6, 10, 14,  4,  8, 12, 16
        result.n[0] = _mm256_permute2f128_ps(t4, t5, 0x20);  // t6 =  1,  5,  9, 13,  2,  6, 10, 14
    _mm_store_ps(&B[0*rowSizeB], result.m[0]);
    _mm_store_ps(&B[1*rowSizeB], result.m[1]);
        result.n[1] = _mm256_permute2f128_ps(t4, t5, 0x31);  // t7 =  3,  7, 11, 15,  4,  8, 12, 16
    _mm_store_ps(&B[2*rowSizeB], result.m[2]);
    _mm_store_ps(&B[3*rowSizeB], result.m[3]);
  }

};

} // namespace

#endif
