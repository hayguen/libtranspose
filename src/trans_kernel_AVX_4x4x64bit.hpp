
#pragma once

#include "transpose_defs.hpp"

#if defined(__AVX__) || defined(__AVX2__)
#  include <immintrin.h>
#  define HAVE_AVX_4X4X64_KERNEL 1
#endif

#ifdef HAVE_AVX_4X4X64_KERNEL

#include <cstdint>

namespace transpose_kernels
{

template <class T>
struct AVX_4x4x64Kernel
{
  // requires AVX
  static constexpr unsigned KERNEL_SZ = 4;
  static constexpr bool HAS_AA = true;
  using BaseType = double;

  // => ?

  ALWAYS_INLINE static void op_uu(const T * RESTRICT A_, T * RESTRICT B_, const unsigned rowSizeA, const unsigned rowSizeB) {
    const BaseType * RESTRICT A = reinterpret_cast<const BaseType * RESTRICT>(A_);
    BaseType * RESTRICT B = reinterpret_cast<BaseType * RESTRICT>(B_);
    static_assert( sizeof(T) == sizeof(BaseType), "" );
    static_assert( sizeof(T) == sizeof(int64_t), "" );

    // _MM_TRANSPOSE4_PD()
    // from https://github.com/romeric/Fastor/blob/master/Fastor/backend/transpose/transpose_kernels.h
    // added unaligned loads and stores; interleaved instructions

    __m256d row0 = _mm256_loadu_pd(&A[0*rowSizeA]);
    __m256d row1 = _mm256_loadu_pd(&A[1*rowSizeA]);

    __m256d tmp0 = _mm256_shuffle_pd((row0),(row1), 0x0);
    __m256d tmp2 = _mm256_shuffle_pd((row0),(row1), 0xF);

    __m256d row2 = _mm256_loadu_pd(&A[2*rowSizeA]);
    __m256d row3 = _mm256_loadu_pd(&A[3*rowSizeA]);

    __m256d tmp1 = _mm256_shuffle_pd((row2),(row3), 0x0);
    __m256d tmp3 = _mm256_shuffle_pd((row2),(row3), 0xF);

    row0 = _mm256_permute2f128_pd(tmp0, tmp1, 0x20);
    row2 = _mm256_permute2f128_pd(tmp0, tmp1, 0x31);

    _mm256_storeu_pd(&B[0*rowSizeB], row0);
    _mm256_storeu_pd(&B[2*rowSizeB], row2);

    row1 = _mm256_permute2f128_pd(tmp2, tmp3, 0x20);
    row3 = _mm256_permute2f128_pd(tmp2, tmp3, 0x31);

    _mm256_storeu_pd(&B[1*rowSizeB], row1);
    _mm256_storeu_pd(&B[3*rowSizeB], row3);
  }

  ALWAYS_INLINE static void op_aa(const T * RESTRICT A_, T * RESTRICT B_, const unsigned rowSizeA, const unsigned rowSizeB) {
    const BaseType * RESTRICT A = reinterpret_cast<const BaseType * RESTRICT>(A_);
    BaseType * RESTRICT B = reinterpret_cast<BaseType * RESTRICT>(B_);
    static_assert( sizeof(T) == sizeof(BaseType), "" );
    static_assert( sizeof(T) == sizeof(int64_t), "" );

    __m256d row0 = _mm256_load_pd(&A[0*rowSizeA]);
    __m256d row1 = _mm256_load_pd(&A[1*rowSizeA]);

    __m256d tmp0 = _mm256_shuffle_pd((row0),(row1), 0x0);
    __m256d tmp2 = _mm256_shuffle_pd((row0),(row1), 0xF);

    __m256d row2 = _mm256_load_pd(&A[2*rowSizeA]);
    __m256d row3 = _mm256_load_pd(&A[3*rowSizeA]);

    __m256d tmp1 = _mm256_shuffle_pd((row2),(row3), 0x0);
    __m256d tmp3 = _mm256_shuffle_pd((row2),(row3), 0xF);

    row0 = _mm256_permute2f128_pd(tmp0, tmp1, 0x20);
    row2 = _mm256_permute2f128_pd(tmp0, tmp1, 0x31);

    _mm256_store_pd(&B[0*rowSizeB], row0);
    _mm256_store_pd(&B[2*rowSizeB], row2);

    row1 = _mm256_permute2f128_pd(tmp2, tmp3, 0x20);
    row3 = _mm256_permute2f128_pd(tmp2, tmp3, 0x31);

    _mm256_store_pd(&B[1*rowSizeB], row1);
    _mm256_store_pd(&B[3*rowSizeB], row3);
  }

};

} // namespace

#endif
