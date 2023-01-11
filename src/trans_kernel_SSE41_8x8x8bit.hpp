
#pragma once

#include "transpose_defs.hpp"

#if defined(__aarch64__) || defined(__arm__)
#  include "sse2neon/sse2neon.h"
#  define HAVE_SSE41_8x8x8_KERNEL 1
#elif (defined(__SSE__) && defined(__SSE4_1__))
#  include <immintrin.h>
#  define HAVE_SSE41_8x8x8_KERNEL 1
#endif

#ifdef HAVE_SSE41_8x8x8_KERNEL

#include <cstdint>

namespace transpose_kernels
{

template <class T>
struct SSE41_8x8x8Kernel
{
  // requires SSE4.1
  static constexpr unsigned KERNEL_SZ = 8;
  using BaseType = uint8_t;

  static constexpr bool HAS_AA = false;
};


#define KERNEL_NAME SSE41_8x8x8Kernel

// could not find another way to have single initialization of variables - other than using C macros
#define KERNEL_INIT() \
  const __m128i transpose4x4mask = _mm_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13,  9, 5, 1, 12,  8, 4, 0); \
  const __m128i shuffle8x8Mask   = _mm_setr_epi8(0, 1, 2, 3, 8, 9, 10, 11, 4,  5, 6, 7, 12,  13, 14, 15)


#define KERNEL_OP_AA() do { } while (0)

// https://stackoverflow.com/questions/42162270/a-better-8x8-bytes-matrix-transpose-with-sse

// void TransposeBlock8x8(uint8_t *src, uint8_t *dst, int srcStride, int dstStride) {
// from https://stackoverflow.com/questions/42162270/a-better-8x8-bytes-matrix-transpose-with-sse
//   interleaved loads and stores

// ALWAYS_INLINE void operator()(const T * RESTRICT A_, T * RESTRICT B_, const unsigned rowSizeA, const unsigned rowSizeB) const {
#define KERNEL_OP_UU() do { \
    const BaseType * RESTRICT A = reinterpret_cast<const BaseType * RESTRICT>(A_);                  \
    BaseType * RESTRICT B = reinterpret_cast<BaseType * RESTRICT>(B_);                              \
    static_assert( sizeof(T) == sizeof(BaseType), "" );                                             \
    __m128i load0 = _mm_set_epi64x(*(uint64_t*)(A + 1 * rowSizeA), *(uint64_t*)(A + 0 * rowSizeA)); \
    __m128i shuffle0 = _mm_shuffle_epi8(load0, shuffle8x8Mask);                                     \
    __m128i load1 = _mm_set_epi64x(*(uint64_t*)(A + 3 * rowSizeA), *(uint64_t*)(A + 2 * rowSizeA)); \
    __m128i shuffle1 = _mm_shuffle_epi8(load1, shuffle8x8Mask);                                     \
    __m128i load2 = _mm_set_epi64x(*(uint64_t*)(A + 5 * rowSizeA), *(uint64_t*)(A + 4 * rowSizeA)); \
    __m128i shuffle2 = _mm_shuffle_epi8(load2, shuffle8x8Mask);                                     \
    __m128i load3 = _mm_set_epi64x(*(uint64_t*)(A + 7 * rowSizeA), *(uint64_t*)(A + 6 * rowSizeA)); \
    __m128i shuffle3 = _mm_shuffle_epi8(load3, shuffle8x8Mask);                                     \
    __m128i block0 = _mm_unpacklo_epi64(shuffle0, shuffle1);                                        \
    __m128i block1 = _mm_unpackhi_epi64(shuffle0, shuffle1);                                        \
    __m128i block2 = _mm_unpacklo_epi64(shuffle2, shuffle3);                                        \
    __m128i block3 = _mm_unpackhi_epi64(shuffle2, shuffle3);                                        \
    __m128i transposed0 = _mm_shuffle_epi8(block0, transpose4x4mask);                               \
    __m128i transposed1 = _mm_shuffle_epi8(block1, transpose4x4mask);                               \
    __m128i transposed2 = _mm_shuffle_epi8(block2, transpose4x4mask);                               \
    __m128i transposed3 = _mm_shuffle_epi8(block3, transpose4x4mask);                               \
    __m128i store0 = _mm_unpacklo_epi32(transposed0, transposed2);                                  \
    *((uint64_t*)(B + 0 * rowSizeB)) = _mm_extract_epi64(store0, 0);                                \
    *((uint64_t*)(B + 1 * rowSizeB)) = _mm_extract_epi64(store0, 1);                                \
    __m128i store1 = _mm_unpackhi_epi32(transposed0, transposed2);                                  \
    *((uint64_t*)(B + 2 * rowSizeB)) = _mm_extract_epi64(store1, 0);                                \
    *((uint64_t*)(B + 3 * rowSizeB)) = _mm_extract_epi64(store1, 1);                                \
    __m128i store2 = _mm_unpacklo_epi32(transposed1, transposed3);                                  \
    *((uint64_t*)(B + 4 * rowSizeB)) = _mm_extract_epi64(store2, 0);                                \
    *((uint64_t*)(B + 5 * rowSizeB)) = _mm_extract_epi64(store2, 1);                                \
    __m128i store3 = _mm_unpackhi_epi32(transposed1, transposed3);                                  \
    *((uint64_t*)(B + 6 * rowSizeB)) = _mm_extract_epi64(store3, 0);                                \
    *((uint64_t*)(B + 7 * rowSizeB)) = _mm_extract_epi64(store3, 1);                                \
  } while (0)

} // namespace

#endif

#if 0

void TransposeBlock8x8(uint8_t *src, uint8_t *dst, int srcStride, int dstStride) {
    __m128i load0 = _mm_set_epi64x(*(uint64_t*)(src + 1 * srcStride), *(uint64_t*)(src + 0 * srcStride));
    __m128i load1 = _mm_set_epi64x(*(uint64_t*)(src + 3 * srcStride), *(uint64_t*)(src + 2 * srcStride));
    __m128i load2 = _mm_set_epi64x(*(uint64_t*)(src + 5 * srcStride), *(uint64_t*)(src + 4 * srcStride));
    __m128i load3 = _mm_set_epi64x(*(uint64_t*)(src + 7 * srcStride), *(uint64_t*)(src + 6 * srcStride));

    __m128i shuffle0 = _mm_shuffle_epi8(load0, shuffle8x8Mask);
    __m128i shuffle1 = _mm_shuffle_epi8(load1, shuffle8x8Mask);
    __m128i shuffle2 = _mm_shuffle_epi8(load2, shuffle8x8Mask);
    __m128i shuffle3 = _mm_shuffle_epi8(load3, shuffle8x8Mask);

    __m128i block0 = _mm_unpacklo_epi64(shuffle0, shuffle1);
    __m128i block1 = _mm_unpackhi_epi64(shuffle0, shuffle1);
    __m128i block2 = _mm_unpacklo_epi64(shuffle2, shuffle3);
    __m128i block3 = _mm_unpackhi_epi64(shuffle2, shuffle3);

    __m128i transposed0 = _mm_shuffle_epi8(block0, transpose4x4mask);   
    __m128i transposed1 = _mm_shuffle_epi8(block1, transpose4x4mask);   
    __m128i transposed2 = _mm_shuffle_epi8(block2, transpose4x4mask);   
    __m128i transposed3 = _mm_shuffle_epi8(block3, transpose4x4mask);   

    __m128i store0 = _mm_unpacklo_epi32(transposed0, transposed2);
    __m128i store1 = _mm_unpackhi_epi32(transposed0, transposed2);
    __m128i store2 = _mm_unpacklo_epi32(transposed1, transposed3);
    __m128i store3 = _mm_unpackhi_epi32(transposed1, transposed3);

    *((uint64_t*)(dst + 0 * dstStride)) = _mm_extract_epi64(store0, 0);
    *((uint64_t*)(dst + 1 * dstStride)) = _mm_extract_epi64(store0, 1);
    *((uint64_t*)(dst + 2 * dstStride)) = _mm_extract_epi64(store1, 0);
    *((uint64_t*)(dst + 3 * dstStride)) = _mm_extract_epi64(store1, 1);
    *((uint64_t*)(dst + 4 * dstStride)) = _mm_extract_epi64(store2, 0);
    *((uint64_t*)(dst + 5 * dstStride)) = _mm_extract_epi64(store2, 1);
    *((uint64_t*)(dst + 6 * dstStride)) = _mm_extract_epi64(store3, 0);
    *((uint64_t*)(dst + 7 * dstStride)) = _mm_extract_epi64(store3, 1);
}

#endif
