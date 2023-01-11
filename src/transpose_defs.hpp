
#pragma once

#include <hedley.h>

// shorten some used macros
#  define ALWAYS_INLINE  HEDLEY_ALWAYS_INLINE
#  define RESTRICT       HEDLEY_RESTRICT


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
