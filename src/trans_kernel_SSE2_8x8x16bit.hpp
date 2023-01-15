
#pragma once

#include "transpose_defs.hpp"

#if defined(__aarch64__) || defined(__arm__)
#  include "sse2neon/sse2neon.h"
#  define HAVE_SSE2_8x8x16_KERNEL 1
#elif (defined(__SSE__) && defined(__SSE2__) ) || ( defined(__x86_64__) || defined(_M_X64) || (defined(_M_IX86) && _M_IX86 >= 2) )
#  include <immintrin.h>
#  define HAVE_SSE2_8x8x16_KERNEL 1
#endif

#ifdef HAVE_SSE2_8x8x16_KERNEL

#include <cstdint>

namespace transpose_kernels
{

template <class T, bool CONJUGATE_TPL = false>
struct SSE2_8x8x16Kernel
{
  // requires SSE2
  static constexpr unsigned KERNEL_SZ = 8;
  static constexpr bool HAS_AA = true;
  static constexpr bool CONJUGATE = CONJUGATE_TPL;
  static_assert( !CONJUGATE_TPL, "CONJUGATE is not supported by SSE2_8x8x16Kernel" );
  using BaseType = uint16_t;


  // => looks to give best performance for 16 bit :-)
  //   performance is mostly better than Intel OneAPI IPP's ippiTranspose_*()

  ALWAYS_INLINE static void op_uu(const T * RESTRICT A_, T * RESTRICT B_, const unsigned rowSizeA, const unsigned rowSizeB) {
    const BaseType * RESTRICT A = reinterpret_cast<const BaseType * RESTRICT>(A_);
    BaseType * RESTRICT B = reinterpret_cast<BaseType * RESTRICT>(B_);
    static_assert( sizeof(T) == sizeof(BaseType), "" );

    // https://stackoverflow.com/questions/2517584/transpose-for-8-registers-of-16-bit-elements-on-sse2-ssse3
    //   but made loads and stores unaligned, interleaved instructions

    // read 128 bits == 8 int16 per row
    __m128i a = _mm_loadu_si128( reinterpret_cast<const __m128i*>(&A[0*rowSizeA]) );
    __m128i b = _mm_loadu_si128( reinterpret_cast<const __m128i*>(&A[1*rowSizeA]) );
    __m128i a03b03 = _mm_unpacklo_epi16(a, b);
    __m128i a47b47 = _mm_unpackhi_epi16(a, b);

    __m128i c = _mm_loadu_si128( reinterpret_cast<const __m128i*>(&A[2*rowSizeA]) );
    __m128i d = _mm_loadu_si128( reinterpret_cast<const __m128i*>(&A[3*rowSizeA]) );
    __m128i c03d03 = _mm_unpacklo_epi16(c, d);
    __m128i c47d47 = _mm_unpackhi_epi16(c, d);

    __m128i e = _mm_loadu_si128( reinterpret_cast<const __m128i*>(&A[4*rowSizeA]) );
    __m128i f = _mm_loadu_si128( reinterpret_cast<const __m128i*>(&A[5*rowSizeA]) );
    __m128i e03f03 = _mm_unpacklo_epi16(e, f);
    __m128i e47f47 = _mm_unpackhi_epi16(e, f);

    __m128i g = _mm_loadu_si128( reinterpret_cast<const __m128i*>(&A[6*rowSizeA]) );
    __m128i h = _mm_loadu_si128( reinterpret_cast<const __m128i*>(&A[7*rowSizeA]) );
    __m128i g03h03 = _mm_unpacklo_epi16(g, h);
    __m128i g47h47 = _mm_unpackhi_epi16(g, h);


    __m128i a01b01c01d01 = _mm_unpacklo_epi32(a03b03, c03d03);
    __m128i e01f01g01h01 = _mm_unpacklo_epi32(e03f03, g03h03);
    __m128i a1b1c1d1e1f1g1h1 = _mm_unpackhi_epi64(a01b01c01d01, e01f01g01h01);
    _mm_storeu_si128( reinterpret_cast<__m128i*>(&B[1*rowSizeB]), a1b1c1d1e1f1g1h1 );

    __m128i a0b0c0d0e0f0g0h0 = _mm_unpacklo_epi64(a01b01c01d01, e01f01g01h01);
    _mm_storeu_si128( reinterpret_cast<__m128i*>(&B[0*rowSizeB]), a0b0c0d0e0f0g0h0 );

    __m128i a23b23c23d23 = _mm_unpackhi_epi32(a03b03, c03d03);
    __m128i e23f23g23h23 = _mm_unpackhi_epi32(e03f03, g03h03);
    __m128i a2b2c2d2e2f2g2h2 = _mm_unpacklo_epi64(a23b23c23d23, e23f23g23h23);
    _mm_storeu_si128( reinterpret_cast<__m128i*>(&B[2*rowSizeB]), a2b2c2d2e2f2g2h2 );
    __m128i a3b3c3d3e3f3g3h3 = _mm_unpackhi_epi64(a23b23c23d23, e23f23g23h23);
    _mm_storeu_si128( reinterpret_cast<__m128i*>(&B[3*rowSizeB]), a3b3c3d3e3f3g3h3 );

    __m128i a45b45c45d45 = _mm_unpacklo_epi32(a47b47, c47d47);
    __m128i e45f45g45h45 = _mm_unpacklo_epi32(e47f47, g47h47);
    __m128i a4b4c4d4e4f4g4h4 = _mm_unpacklo_epi64(a45b45c45d45, e45f45g45h45);
    _mm_storeu_si128( reinterpret_cast<__m128i*>(&B[4*rowSizeB]), a4b4c4d4e4f4g4h4 );
    __m128i a5b5c5d5e5f5g5h5 = _mm_unpackhi_epi64(a45b45c45d45, e45f45g45h45);
    _mm_storeu_si128( reinterpret_cast<__m128i*>(&B[5*rowSizeB]), a5b5c5d5e5f5g5h5 );

    __m128i a67b67c67d67 = _mm_unpackhi_epi32(a47b47, c47d47);
    __m128i e67f67g67h67 = _mm_unpackhi_epi32(e47f47, g47h47);
    __m128i a6b6c6d6e6f6g6h6 = _mm_unpacklo_epi64(a67b67c67d67, e67f67g67h67);
    _mm_storeu_si128( reinterpret_cast<__m128i*>(&B[6*rowSizeB]), a6b6c6d6e6f6g6h6 );
    __m128i a7b7c7d7e7f7g7h7 = _mm_unpackhi_epi64(a67b67c67d67, e67f67g67h67);
    _mm_storeu_si128( reinterpret_cast<__m128i*>(&B[7*rowSizeB]), a7b7c7d7e7f7g7h7 );
  }

  ALWAYS_INLINE static void op_aa(const T * RESTRICT A_, T * RESTRICT B_, const unsigned rowSizeA, const unsigned rowSizeB) {
    const BaseType * RESTRICT A = reinterpret_cast<const BaseType * RESTRICT>(A_);
    BaseType * RESTRICT B = reinterpret_cast<BaseType * RESTRICT>(B_);
    static_assert( sizeof(T) == sizeof(BaseType), "" );

    __m128i a = _mm_load_si128( reinterpret_cast<const __m128i*>(&A[0*rowSizeA]) );
    __m128i b = _mm_load_si128( reinterpret_cast<const __m128i*>(&A[1*rowSizeA]) );
    __m128i a03b03 = _mm_unpacklo_epi16(a, b);
    __m128i a47b47 = _mm_unpackhi_epi16(a, b);

    __m128i c = _mm_load_si128( reinterpret_cast<const __m128i*>(&A[2*rowSizeA]) );
    __m128i d = _mm_load_si128( reinterpret_cast<const __m128i*>(&A[3*rowSizeA]) );
    __m128i c03d03 = _mm_unpacklo_epi16(c, d);
    __m128i c47d47 = _mm_unpackhi_epi16(c, d);

    __m128i e = _mm_load_si128( reinterpret_cast<const __m128i*>(&A[4*rowSizeA]) );
    __m128i f = _mm_load_si128( reinterpret_cast<const __m128i*>(&A[5*rowSizeA]) );
    __m128i e03f03 = _mm_unpacklo_epi16(e, f);
    __m128i e47f47 = _mm_unpackhi_epi16(e, f);

    __m128i g = _mm_load_si128( reinterpret_cast<const __m128i*>(&A[6*rowSizeA]) );
    __m128i h = _mm_load_si128( reinterpret_cast<const __m128i*>(&A[7*rowSizeA]) );
    __m128i g03h03 = _mm_unpacklo_epi16(g, h);
    __m128i g47h47 = _mm_unpackhi_epi16(g, h);

    __m128i a01b01c01d01 = _mm_unpacklo_epi32(a03b03, c03d03);
    __m128i e01f01g01h01 = _mm_unpacklo_epi32(e03f03, g03h03);
    __m128i a1b1c1d1e1f1g1h1 = _mm_unpackhi_epi64(a01b01c01d01, e01f01g01h01);
    _mm_store_si128( reinterpret_cast<__m128i*>(&B[1*rowSizeB]), a1b1c1d1e1f1g1h1 );

    __m128i a0b0c0d0e0f0g0h0 = _mm_unpacklo_epi64(a01b01c01d01, e01f01g01h01);
    _mm_store_si128( reinterpret_cast<__m128i*>(&B[0*rowSizeB]), a0b0c0d0e0f0g0h0 );

    __m128i a23b23c23d23 = _mm_unpackhi_epi32(a03b03, c03d03);
    __m128i e23f23g23h23 = _mm_unpackhi_epi32(e03f03, g03h03);
    __m128i a2b2c2d2e2f2g2h2 = _mm_unpacklo_epi64(a23b23c23d23, e23f23g23h23);
    _mm_store_si128( reinterpret_cast<__m128i*>(&B[2*rowSizeB]), a2b2c2d2e2f2g2h2 );
    __m128i a3b3c3d3e3f3g3h3 = _mm_unpackhi_epi64(a23b23c23d23, e23f23g23h23);
    _mm_store_si128( reinterpret_cast<__m128i*>(&B[3*rowSizeB]), a3b3c3d3e3f3g3h3 );

    __m128i a45b45c45d45 = _mm_unpacklo_epi32(a47b47, c47d47);
    __m128i e45f45g45h45 = _mm_unpacklo_epi32(e47f47, g47h47);
    __m128i a4b4c4d4e4f4g4h4 = _mm_unpacklo_epi64(a45b45c45d45, e45f45g45h45);
    _mm_store_si128( reinterpret_cast<__m128i*>(&B[4*rowSizeB]), a4b4c4d4e4f4g4h4 );
    __m128i a5b5c5d5e5f5g5h5 = _mm_unpackhi_epi64(a45b45c45d45, e45f45g45h45);
    _mm_store_si128( reinterpret_cast<__m128i*>(&B[5*rowSizeB]), a5b5c5d5e5f5g5h5 );

    __m128i a67b67c67d67 = _mm_unpackhi_epi32(a47b47, c47d47);
    __m128i e67f67g67h67 = _mm_unpackhi_epi32(e47f47, g47h47);
    __m128i a6b6c6d6e6f6g6h6 = _mm_unpacklo_epi64(a67b67c67d67, e67f67g67h67);
    _mm_store_si128( reinterpret_cast<__m128i*>(&B[6*rowSizeB]), a6b6c6d6e6f6g6h6 );
    __m128i a7b7c7d7e7f7g7h7 = _mm_unpackhi_epi64(a67b67c67d67, e67f67g67h67);
    _mm_store_si128( reinterpret_cast<__m128i*>(&B[7*rowSizeB]), a7b7c7d7e7f7g7h7 );
  }

};

} // namespace

#endif
