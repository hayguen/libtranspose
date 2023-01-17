
#pragma once

#include "transpose_defs.hpp"
#include <complex>
#include <type_traits>

#if defined(__AVX__) || defined(__AVX2__)
#  include <immintrin.h>
#  define HAVE_AVX_4X4X128_KERNEL 1
#endif

#ifdef HAVE_AVX_4X4X128_KERNEL

#include <cstdint>

namespace transpose_kernels
{

template <class T, bool CONJUGATE_TPL = false>
struct AVX_4x4x128Kernel
{
  // requires AVX
  static constexpr unsigned KERNEL_SZ = 4;
  static constexpr bool HAS_AA = true;
  static constexpr bool CONJUGATE = CONJUGATE_TPL;
  using BaseType = std::complex<double>;

  static_assert( !CONJUGATE
    || std::is_same<T, std::complex<float> >::value
    || std::is_same<T, std::complex<double> >::value
    , "CONJUGATE is only supported by AVX_4x4x128Kernel for std::complex<double>" );
};

} // namespace


#define KERNEL_NAME AVX_4x4x128Kernel

// could not find another way to have single initialization of variables - other than using C macros
#define KERNEL_INIT() \
  static constexpr int64_t neg_bm = uint64_t(0x80000000) << 32; \
  const __m256i cj = _mm256_set_epi64x(neg_bm, 0, neg_bm, 0); \
  constexpr int imm_lo128 = 0x20; \
  constexpr int imm_hi128 = 0x31; \
  constexpr int C = sizeof(__m256d) / sizeof(T);


// ALWAYS_INLINE void operator()(const T * RESTRICT A_, T * RESTRICT B_, const unsigned rowSizeA, const unsigned rowSizeB) const {
#define KERNEL_OP_UU() do { \
    const BaseType * RESTRICT A = reinterpret_cast<const BaseType * RESTRICT>(A_);                  \
    BaseType * RESTRICT B = reinterpret_cast<BaseType * RESTRICT>(B_);                              \
    static_assert( sizeof(T) == sizeof(BaseType), "" );                                             \
\
    __m256d a11a12 = _mm256_loadu_pd(reinterpret_cast<const double*>(&A[0*rowSizeA+0]));            \
    __m256d a21a22 = _mm256_loadu_pd(reinterpret_cast<const double*>(&A[1*rowSizeA+0]));            \
    __m256d a31a32 = _mm256_loadu_pd(reinterpret_cast<const double*>(&A[2*rowSizeA+0]));            \
    __m256d a41a42 = _mm256_loadu_pd(reinterpret_cast<const double*>(&A[3*rowSizeA+0]));            \
\
    __m256d a11a21 = _mm256_permute2f128_pd( a11a12, a21a22, imm_lo128 );                           \
    __m256d a31a41 = _mm256_permute2f128_pd( a31a32, a41a42, imm_lo128 );                           \
    __m256d a12a22 = _mm256_permute2f128_pd( a11a12, a21a22, imm_hi128 );                           \
    __m256d a32a42 = _mm256_permute2f128_pd( a31a32, a41a42, imm_hi128 );                           \
    if constexpr ( CONJUGATE ) {                                                                    \
        a11a21 = _mm256_castsi256_pd( _mm256_xor_si256( _mm256_castpd_si256( a11a21 ), cj ) );      \
        a31a41 = _mm256_castsi256_pd( _mm256_xor_si256( _mm256_castpd_si256( a31a41 ), cj ) );      \
        a12a22 = _mm256_castsi256_pd( _mm256_xor_si256( _mm256_castpd_si256( a12a22 ), cj ) );      \
        a32a42 = _mm256_castsi256_pd( _mm256_xor_si256( _mm256_castpd_si256( a32a42 ), cj ) );      \
    }                                                                                               \
\
    __m256d a13a14 = _mm256_loadu_pd(reinterpret_cast<const double*>(&A[0*rowSizeA+C]));            \
    __m256d a23a24 = _mm256_loadu_pd(reinterpret_cast<const double*>(&A[1*rowSizeA+C]));            \
    __m256d a33a34 = _mm256_loadu_pd(reinterpret_cast<const double*>(&A[2*rowSizeA+C]));            \
    __m256d a43a44 = _mm256_loadu_pd(reinterpret_cast<const double*>(&A[3*rowSizeA+C]));            \
\
    _mm256_storeu_pd(reinterpret_cast<double*>(&B[0*rowSizeB+0]), a11a21);                          \
    _mm256_storeu_pd(reinterpret_cast<double*>(&B[0*rowSizeB+C]), a31a41);                          \
    _mm256_storeu_pd(reinterpret_cast<double*>(&B[1*rowSizeB+0]), a12a22);                          \
    _mm256_storeu_pd(reinterpret_cast<double*>(&B[1*rowSizeB+C]), a32a42);                          \
\
    __m256d a13a23 = _mm256_permute2f128_pd( a13a14, a23a24, imm_lo128 );                           \
    __m256d a33a43 = _mm256_permute2f128_pd( a33a34, a43a44, imm_lo128 );                           \
    __m256d a14a24 = _mm256_permute2f128_pd( a13a14, a23a24, imm_hi128 );                           \
    __m256d a34a44 = _mm256_permute2f128_pd( a33a34, a43a44, imm_hi128 );                           \
    if constexpr ( CONJUGATE ) {                                                                    \
        a13a23 = _mm256_castsi256_pd( _mm256_xor_si256( _mm256_castpd_si256( a13a23 ), cj ) );      \
        a33a43 = _mm256_castsi256_pd( _mm256_xor_si256( _mm256_castpd_si256( a33a43 ), cj ) );      \
        a14a24 = _mm256_castsi256_pd( _mm256_xor_si256( _mm256_castpd_si256( a14a24 ), cj ) );      \
        a34a44 = _mm256_castsi256_pd( _mm256_xor_si256( _mm256_castpd_si256( a34a44 ), cj ) );      \
    }                                                                                               \
\
    _mm256_storeu_pd(reinterpret_cast<double*>(&B[2*rowSizeB+0]), a13a23);                          \
    _mm256_storeu_pd(reinterpret_cast<double*>(&B[2*rowSizeB+C]), a33a43);                          \
    _mm256_storeu_pd(reinterpret_cast<double*>(&B[3*rowSizeB+0]), a14a24);                          \
    _mm256_storeu_pd(reinterpret_cast<double*>(&B[3*rowSizeB+C]), a34a44);                          \
  } while (0)

// ALWAYS_INLINE static void op_aa(const T * RESTRICT A_, T * RESTRICT B_, const unsigned rowSizeA, const unsigned rowSizeB) {
#define KERNEL_OP_AA() do { \
    const BaseType * RESTRICT A = reinterpret_cast<const BaseType * RESTRICT>(A_);                  \
    BaseType * RESTRICT B = reinterpret_cast<BaseType * RESTRICT>(B_);                              \
    static_assert( sizeof(T) == sizeof(BaseType), "" );                                             \
\
    __m256d a11a12 = _mm256_load_pd(reinterpret_cast<const double*>(&A[0*rowSizeA+0]));             \
    __m256d a21a22 = _mm256_load_pd(reinterpret_cast<const double*>(&A[1*rowSizeA+0]));             \
    __m256d a31a32 = _mm256_load_pd(reinterpret_cast<const double*>(&A[2*rowSizeA+0]));             \
    __m256d a41a42 = _mm256_load_pd(reinterpret_cast<const double*>(&A[3*rowSizeA+0]));             \
\
    __m256d a11a21 = _mm256_permute2f128_pd( a11a12, a21a22, imm_lo128 );                           \
    __m256d a31a41 = _mm256_permute2f128_pd( a31a32, a41a42, imm_lo128 );                           \
    __m256d a12a22 = _mm256_permute2f128_pd( a11a12, a21a22, imm_hi128 );                           \
    __m256d a32a42 = _mm256_permute2f128_pd( a31a32, a41a42, imm_hi128 );                           \
    if constexpr ( CONJUGATE ) {                                                                    \
        a11a21 = _mm256_castsi256_pd( _mm256_xor_si256( _mm256_castpd_si256( a11a21 ), cj ) );      \
        a31a41 = _mm256_castsi256_pd( _mm256_xor_si256( _mm256_castpd_si256( a31a41 ), cj ) );      \
        a12a22 = _mm256_castsi256_pd( _mm256_xor_si256( _mm256_castpd_si256( a12a22 ), cj ) );      \
        a32a42 = _mm256_castsi256_pd( _mm256_xor_si256( _mm256_castpd_si256( a32a42 ), cj ) );      \
    }                                                                                               \
\
    __m256d a13a14 = _mm256_load_pd(reinterpret_cast<const double*>(&A[0*rowSizeA+C]));             \
    __m256d a23a24 = _mm256_load_pd(reinterpret_cast<const double*>(&A[1*rowSizeA+C]));             \
    __m256d a33a34 = _mm256_load_pd(reinterpret_cast<const double*>(&A[2*rowSizeA+C]));             \
    __m256d a43a44 = _mm256_load_pd(reinterpret_cast<const double*>(&A[3*rowSizeA+C]));             \
\
    _mm256_store_pd(reinterpret_cast<double*>(&B[0*rowSizeB+0]), a11a21);                           \
    _mm256_store_pd(reinterpret_cast<double*>(&B[0*rowSizeB+C]), a31a41);                           \
    _mm256_store_pd(reinterpret_cast<double*>(&B[1*rowSizeB+0]), a12a22);                           \
    _mm256_store_pd(reinterpret_cast<double*>(&B[1*rowSizeB+C]), a32a42);                           \
\
    __m256d a13a23 = _mm256_permute2f128_pd( a13a14, a23a24, imm_lo128 );                           \
    __m256d a33a43 = _mm256_permute2f128_pd( a33a34, a43a44, imm_lo128 );                           \
    __m256d a14a24 = _mm256_permute2f128_pd( a13a14, a23a24, imm_hi128 );                           \
    __m256d a34a44 = _mm256_permute2f128_pd( a33a34, a43a44, imm_hi128 );                           \
    if constexpr ( CONJUGATE ) {                                                                    \
        a13a23 = _mm256_castsi256_pd( _mm256_xor_si256( _mm256_castpd_si256( a13a23 ), cj ) );      \
        a33a43 = _mm256_castsi256_pd( _mm256_xor_si256( _mm256_castpd_si256( a33a43 ), cj ) );      \
        a14a24 = _mm256_castsi256_pd( _mm256_xor_si256( _mm256_castpd_si256( a14a24 ), cj ) );      \
        a34a44 = _mm256_castsi256_pd( _mm256_xor_si256( _mm256_castpd_si256( a34a44 ), cj ) );      \
    }                                                                                               \
\
    _mm256_store_pd(reinterpret_cast<double*>(&B[2*rowSizeB+0]), a13a23);                           \
    _mm256_store_pd(reinterpret_cast<double*>(&B[2*rowSizeB+C]), a33a43);                           \
    _mm256_store_pd(reinterpret_cast<double*>(&B[3*rowSizeB+0]), a14a24);                           \
    _mm256_store_pd(reinterpret_cast<double*>(&B[3*rowSizeB+C]), a34a44);                           \
  } while (0)

#endif  // HAVE_AVX_4X4X128_KERNEL
