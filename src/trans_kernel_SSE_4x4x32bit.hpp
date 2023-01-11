
#pragma once

#include "transpose_defs.hpp"

#if defined(__aarch64__) || defined(__arm__)
#  include "sse2neon/sse2neon.h"
#  define HAVE_SSE_4X4X32_KERNEL 1
#elif defined(__SSE__) || ( defined(__x86_64__) || defined(_M_X64) || (defined(_M_IX86) && _M_IX86 >= 1) )
#  include <immintrin.h>
#  define HAVE_SSE_4X4X32_KERNEL 1
#endif

#ifdef HAVE_SSE_4X4X32_KERNEL

#include <cstdint>

namespace transpose_kernels
{

template <class T>
struct SSE_4x4x32Kernel
{
  // requires SSE
  static constexpr unsigned KERNEL_SZ = 4;
  static constexpr bool HAS_AA = true;
  using BaseType = float;


  // => looks to give best performance for 32 bit :-)
  //   performance is similar to Intel OneAPI IPP's ippiTranspose_*()
  //   but doesn't allow transformation (see FuncId{}) in bench.cpp  :-(

  ALWAYS_INLINE static void op_uu(const T * RESTRICT A_, T * RESTRICT B_, const unsigned rowSizeA, const unsigned rowSizeB) {
    const BaseType * RESTRICT A = reinterpret_cast<const BaseType * RESTRICT>(A_);
    BaseType * RESTRICT B = reinterpret_cast<BaseType * RESTRICT>(B_);
    static_assert( sizeof(T) == sizeof(BaseType), "" );
    static_assert( sizeof(T) == sizeof(int32_t), "" );

    // see https://stackoverflow.com/questions/16941098/fast-memory-transpose-with-sse-avx-and-openmp
    //   but made loads and stores unaligned
    __m128 row1 = _mm_loadu_ps(&A[0*rowSizeA]);
    __m128 row2 = _mm_loadu_ps(&A[1*rowSizeA]);
    __m128 row3 = _mm_loadu_ps(&A[2*rowSizeA]);
    __m128 row4 = _mm_loadu_ps(&A[3*rowSizeA]);
    // https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-sse/macro-functions-1/macro-function-for-matrix-transposition.html
    _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
    _mm_storeu_ps(&B[0*rowSizeB], row1);
    _mm_storeu_ps(&B[1*rowSizeB], row2);
    _mm_storeu_ps(&B[2*rowSizeB], row3);
    _mm_storeu_ps(&B[3*rowSizeB], row4);
  }

  ALWAYS_INLINE static void op_aa(const T * RESTRICT A_, T * RESTRICT B_, const unsigned rowSizeA, const unsigned rowSizeB) {
    const BaseType * RESTRICT A = reinterpret_cast<const BaseType * RESTRICT>(A_);
    BaseType * RESTRICT B = reinterpret_cast<BaseType * RESTRICT>(B_);
    static_assert( sizeof(T) == sizeof(BaseType), "" );
    static_assert( sizeof(T) == sizeof(int32_t), "" );
    __m128 row1 = _mm_load_ps(&A[0*rowSizeA]);
    __m128 row2 = _mm_load_ps(&A[1*rowSizeA]);
    __m128 row3 = _mm_load_ps(&A[2*rowSizeA]);
    __m128 row4 = _mm_load_ps(&A[3*rowSizeA]);
    _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
    _mm_store_ps(&B[0*rowSizeB], row1);
    _mm_store_ps(&B[1*rowSizeB], row2);
    _mm_store_ps(&B[2*rowSizeB], row3);
    _mm_store_ps(&B[3*rowSizeB], row4);
  }

};

} // namespace

#endif
