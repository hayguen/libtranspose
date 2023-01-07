
#pragma once

#if defined(__GNUC__)
#  define ALWAYS_INLINE(return_type) inline return_type __attribute__ ((always_inline))
#  define NEVER_INLINE(return_type) return_type __attribute__ ((noinline))
#  define RESTRICT __restrict

#elif defined(_MSC_VER)
#  define ALWAYS_INLINE(return_type) __forceinline return_type
#  define NEVER_INLINE(return_type) __declspec(noinline) return_type
#  define RESTRICT __restrict

#else
#  define ALWAYS_INLINE(return_type) inline return_type
#  define NEVER_INLINE(return_type) return_type
#  define RESTRICT __restrict__
#endif

namespace transpose
{

#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
static constexpr unsigned CACHE_LINE_SZ = 128;
#else
static constexpr unsigned CACHE_LINE_SZ =  64;
#endif


template <class T>
struct mat_info
{
  unsigned nRows;  // #rows
  unsigned nCols;  // #cols
  unsigned rowSize; // row size >= nCols

  T * RESTRICT vector;
};

template <class  T>
constexpr unsigned numElemsInCacheLine()
{
  return ( sizeof(T) < CACHE_LINE_SZ ) ? ( CACHE_LINE_SZ / sizeof(T) ) : 1U;
}

}
