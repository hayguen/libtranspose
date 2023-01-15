
#pragma once

#include "transpose_defs.hpp"

#include <memory>
#include <complex>
#include <type_traits>


namespace transpose
{


template <class T, class U, bool CONJUGATE = false>
ALWAYS_INLINE HEDLEY_NO_THROW
static void tail_transpose_out(
  NO_ESCAPE const T * RESTRICT pin, NO_ESCAPE U * RESTRICT pout,
  const unsigned nRows, const unsigned nCols, const unsigned rowSizeA, const unsigned rowSizeB )
{
  static_assert( !CONJUGATE
    || std::is_same<T, std::complex<float> >::value
    || std::is_same<T, std::complex<double> >::value
    , "CONJUGATE is only supported by tail_transpose_out for std::complex<float or double>" );
  unsigned out_off, in_off;
  if constexpr ( CONJUGATE ) {
    for( unsigned r = out_off = 0; r < nRows; ++r, out_off += rowSizeB ) {
      for( unsigned c = in_off = 0; c < nCols; ++c, in_off += rowSizeA )
        pout[out_off+c] = std::conj( pin[in_off+r] );
    }
  }
  else
  {
    for( unsigned r = out_off = 0; r < nRows; ++r, out_off += rowSizeB ) {
      for( unsigned c = in_off = 0; c < nCols; ++c, in_off += rowSizeA )
        pout[out_off+c] = pin[in_off+r];
    }
  }
}

//////////////////////////////////////////////////////

template <class T, class U, bool CONJUGATE = false>
ALWAYS_INLINE HEDLEY_NO_THROW
static void tail_transpose_in(
  NO_ESCAPE const T * RESTRICT pin, NO_ESCAPE U * RESTRICT pout,
  const unsigned nRows, const unsigned nCols, const unsigned rowSizeA, const unsigned rowSizeB )
{
  static_assert( !CONJUGATE
    || std::is_same<T, std::complex<float> >::value
    || std::is_same<T, std::complex<double> >::value
    , "CONJUGATE is only supported by tail_transpose_in for std::complex<float or double>" );
  unsigned out_off, in_off;
  if constexpr ( CONJUGATE ) {
    for( unsigned r = in_off = 0; r < nRows; ++r, in_off += rowSizeA ) {
      for( unsigned c = out_off = 0; c < nCols; ++c, out_off += rowSizeB )
        pout[out_off+r] = std::conj( pin[in_off+c] );
    }
  }
  else
  {
    for( unsigned r = in_off = 0; r < nRows; ++r, in_off += rowSizeA ) {
      for( unsigned c = out_off = 0; c < nCols; ++c, out_off += rowSizeB )
        pout[out_off+r] = pin[in_off+c];
    }
  }
}


}
