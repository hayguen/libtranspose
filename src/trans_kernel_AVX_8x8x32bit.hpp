
#pragma once

#include "transpose_defs.hpp"

#if defined(__AVX__) || defined(__AVX2__)
#  include <immintrin.h>
#  define HAVE_AVX_8X8X32_KERNEL 1
#endif

#ifdef HAVE_AVX_8X8X32_KERNEL

#include <cstdint>

namespace transpose_kernels
{

template <class T, bool CONJUGATE_TPL = false>
struct AVX_8x8x32Kernel
{
  // requires AVX
  static constexpr unsigned KERNEL_SZ = 8;
  static constexpr bool HAS_AA = true;
  static constexpr bool CONJUGATE = CONJUGATE_TPL;
  static_assert( !CONJUGATE_TPL, "CONJUGATE is not supported by AVX_8x8x32Kernel" );

  // => looks to be similar or slightly slower than SSE_4x4x32Kernel !

  ALWAYS_INLINE static void op_uu(const T * RESTRICT A_, T * RESTRICT B_, const unsigned rowSizeA, const unsigned rowSizeB) {
    using BaseType = float;
    const BaseType * RESTRICT A = reinterpret_cast<const BaseType * RESTRICT>(A_);
    BaseType * RESTRICT B = reinterpret_cast<BaseType * RESTRICT>(B_);
    static_assert( sizeof(T) == sizeof(BaseType), "" );
    static_assert( sizeof(T) == sizeof(int32_t), "" );

    // transpose8_ps(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3, __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7)
    //   but made loads and stores unaligned, interleaved instructions
    // from https://stackoverflow.com/questions/25622745/transpose-an-8x8-float-using-avx-avx2
    // == __MM_TRANSPOSE8_PS
    // from  https://github.com/zeux/phyx/blob/master/src/base/SIMD_AVX2_Transpose.h
    // {
        __m256 row0 = _mm256_loadu_ps(&A[0*rowSizeA]);
        __m256 row1 = _mm256_loadu_ps(&A[1*rowSizeA]);

        __m256 __t0 = _mm256_unpacklo_ps(row0, row1);
        __m256 __t1 = _mm256_unpackhi_ps(row0, row1);

        __m256 row2 = _mm256_loadu_ps(&A[2*rowSizeA]);
        __m256 row3 = _mm256_loadu_ps(&A[3*rowSizeA]);

        __m256 __t2 = _mm256_unpacklo_ps(row2, row3);
        __m256 __t3 = _mm256_unpackhi_ps(row2, row3);

        __m256 __tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
        __m256 __tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
        __m256 __tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
        __m256 __tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));

        __m256 row4 = _mm256_loadu_ps(&A[4*rowSizeA]);
        __m256 row5 = _mm256_loadu_ps(&A[5*rowSizeA]);

        __m256 __t4 = _mm256_unpacklo_ps(row4, row5);
        __m256 __t5 = _mm256_unpackhi_ps(row4, row5);

        __m256 row6 = _mm256_loadu_ps(&A[6*rowSizeA]);
        __m256 row7 = _mm256_loadu_ps(&A[7*rowSizeA]);

        __m256 __t6 = _mm256_unpacklo_ps(row6, row7);
        __m256 __t7 = _mm256_unpackhi_ps(row6, row7);

        __m256 __tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
        row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
        _mm256_storeu_ps(&B[0*rowSizeB], row0);
        row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
        _mm256_storeu_ps(&B[4*rowSizeB], row4);

        __m256 __tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
        row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
        _mm256_storeu_ps(&B[1*rowSizeB], row1);
        row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
        _mm256_storeu_ps(&B[5*rowSizeB], row5);

        __m256 __tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
        row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
        _mm256_storeu_ps(&B[2*rowSizeB], row2);
        row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
        _mm256_storeu_ps(&B[6*rowSizeB], row6);

        __m256 __tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
        row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
        _mm256_storeu_ps(&B[3*rowSizeB], row3);
        row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
        _mm256_storeu_ps(&B[7*rowSizeB], row7);
  }

  ALWAYS_INLINE static void op_aa(const T * RESTRICT A_, T * RESTRICT B_, const unsigned rowSizeA, const unsigned rowSizeB) {
    using BaseType = float;
    const BaseType * RESTRICT A = reinterpret_cast<const BaseType * RESTRICT>(A_);
    BaseType * RESTRICT B = reinterpret_cast<BaseType * RESTRICT>(B_);
    static_assert( sizeof(T) == sizeof(BaseType), "" );
    static_assert( sizeof(T) == sizeof(int32_t), "" );

        __m256 row0 = _mm256_load_ps(&A[0*rowSizeA]);
        __m256 row1 = _mm256_load_ps(&A[1*rowSizeA]);

        __m256 __t0 = _mm256_unpacklo_ps(row0, row1);
        __m256 __t1 = _mm256_unpackhi_ps(row0, row1);

        __m256 row2 = _mm256_load_ps(&A[2*rowSizeA]);
        __m256 row3 = _mm256_load_ps(&A[3*rowSizeA]);

        __m256 __t2 = _mm256_unpacklo_ps(row2, row3);
        __m256 __t3 = _mm256_unpackhi_ps(row2, row3);

        __m256 __tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
        __m256 __tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
        __m256 __tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
        __m256 __tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));

        __m256 row4 = _mm256_load_ps(&A[4*rowSizeA]);
        __m256 row5 = _mm256_load_ps(&A[5*rowSizeA]);

        __m256 __t4 = _mm256_unpacklo_ps(row4, row5);
        __m256 __t5 = _mm256_unpackhi_ps(row4, row5);

        __m256 row6 = _mm256_load_ps(&A[6*rowSizeA]);
        __m256 row7 = _mm256_load_ps(&A[7*rowSizeA]);

        __m256 __t6 = _mm256_unpacklo_ps(row6, row7);
        __m256 __t7 = _mm256_unpackhi_ps(row6, row7);

        __m256 __tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
        row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
        _mm256_store_ps(&B[0*rowSizeB], row0);
        row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
        _mm256_store_ps(&B[4*rowSizeB], row4);

        __m256 __tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
        row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
        _mm256_store_ps(&B[1*rowSizeB], row1);
        row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
        _mm256_store_ps(&B[5*rowSizeB], row5);

        __m256 __tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
        row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
        _mm256_store_ps(&B[2*rowSizeB], row2);
        row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
        _mm256_store_ps(&B[6*rowSizeB], row6);

        __m256 __tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
        row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
        _mm256_store_ps(&B[3*rowSizeB], row3);
        row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
        _mm256_store_ps(&B[7*rowSizeB], row7);
  }

};

template <class T, bool CONJUGATE_TPL = false>
struct AVX_8x8x32VINS_Kernel
{
  // requires AVX
  static constexpr unsigned KERNEL_SZ = 8;
  using BaseType = float;

  static constexpr bool HAS_AA = false;
  static constexpr bool CONJUGATE = CONJUGATE_TPL;
  static_assert( !CONJUGATE_TPL, "CONJUGATE is not supported by AVX_8x8x32VINS_Kernel" );

  ALWAYS_INLINE static void op_aa(const T * RESTRICT, T * RESTRICT, const unsigned, const unsigned) { }

  // => looks to be similar or slightly slower than AVX_8x8x32Kernel !

  ALWAYS_INLINE static void op_uu(const T * RESTRICT A_, T * RESTRICT B_, const unsigned rowSizeA, const unsigned rowSizeB) {
    const BaseType * RESTRICT A = reinterpret_cast<const BaseType * RESTRICT>(A_);
    BaseType * RESTRICT B = reinterpret_cast<BaseType * RESTRICT>(B_);
    static_assert( sizeof(T) == sizeof(BaseType), "" );
    static_assert( sizeof(T) == sizeof(int32_t), "" );

    //Example 11-20. 8x8 Matrix Transpose Using VINSERTF128 loads
    // void tran(float* mat, float* matT)
    // from https://stackoverflow.com/questions/25622745/transpose-an-8x8-float-using-avx-avx2
    //   but made loads and stores unaligned, interleaved instructions
    // {

        __m256 r0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(&A[0*rowSizeA +0])), _mm_loadu_ps(&A[4*rowSizeA +0]), 1);
        __m256 r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(&A[1*rowSizeA +0])), _mm_loadu_ps(&A[5*rowSizeA +0]), 1);
        __m256 t0 = _mm256_unpacklo_ps(r0,r1);
        __m256 t1 = _mm256_unpackhi_ps(r0,r1);

        __m256 r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(&A[2*rowSizeA +0])), _mm_loadu_ps(&A[6*rowSizeA +0]), 1);
        __m256 r3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(&A[3*rowSizeA +0])), _mm_loadu_ps(&A[7*rowSizeA +0]), 1);
        __m256 t2 = _mm256_unpacklo_ps(r2,r3);
        __m256 t3 = _mm256_unpackhi_ps(r2,r3);

        r0 = _mm256_shuffle_ps(t0,t2, 0x44);
        _mm256_storeu_ps(&B[0*rowSizeB], r0);

        r1 = _mm256_shuffle_ps(t0,t2, 0xEE);
        _mm256_storeu_ps(&B[1*rowSizeB], r1);

        r2 = _mm256_shuffle_ps(t1,t3, 0x44);
        _mm256_storeu_ps(&B[2*rowSizeB], r2);

        r3 = _mm256_shuffle_ps(t1,t3, 0xEE);
        _mm256_storeu_ps(&B[3*rowSizeB], r3);

        __m256 r4 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(&A[0*rowSizeA +4])), _mm_loadu_ps(&A[4*rowSizeA +4]), 1);
        __m256 r5 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(&A[1*rowSizeA +4])), _mm_loadu_ps(&A[5*rowSizeA +4]), 1);
        __m256 t4 = _mm256_unpacklo_ps(r4,r5);
        __m256 t5 = _mm256_unpackhi_ps(r4,r5);

        __m256 r6 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(&A[2*rowSizeA +4])), _mm_loadu_ps(&A[6*rowSizeA +4]), 1);
        __m256 r7 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(&A[3*rowSizeA +4])), _mm_loadu_ps(&A[7*rowSizeA +4]), 1);
        __m256 t6 = _mm256_unpacklo_ps(r6,r7);
        __m256 t7 = _mm256_unpackhi_ps(r6,r7);

        r4 = _mm256_shuffle_ps(t4,t6, 0x44);
        _mm256_storeu_ps(&B[4*rowSizeB], r4);

        r5 = _mm256_shuffle_ps(t4,t6, 0xEE);
        _mm256_storeu_ps(&B[5*rowSizeB], r5);

        r6 = _mm256_shuffle_ps(t5,t7, 0x44);
        _mm256_storeu_ps(&B[6*rowSizeB], r6);

        r7 = _mm256_shuffle_ps(t5,t7, 0xEE);
        _mm256_storeu_ps(&B[7*rowSizeB], r7);
  }
};

} // namespace

#endif
