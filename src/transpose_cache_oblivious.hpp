
#pragma once

#include "transpose_defs.hpp"
#include "transpose_cache_aware_tails.hpp"
#include <complex>
#include <type_traits>


namespace transpose
{

// see https://en.wikipedia.org/wiki/Cache-oblivious_algorithm
// see Matrix::cachetranspose()
//   from https://stackoverflow.com/questions/5200338/a-cache-efficient-matrix-transpose-program
//   but with important changes
//     - check for recursion first - assuming bigger matrices
//     - fix 2nd recursion for uneven nRows or nCols


// reduce stack size
template <class T, class U>
struct mats_info
{
  const T * const RESTRICT pin;
  U * const RESTRICT pout;

  const unsigned nRows_in;    // #rows
  const unsigned nCols_in;    // #cols
  const unsigned rowSize_in;  // row size >= nCols

  const unsigned nRows_out;   // #rows
  const unsigned nCols_out;   // #cols
  const unsigned rowSize_out; // row size >= nCols
};


template <class T, class U, bool CONJUGATE = false>
HEDLEY_NO_THROW
static void cache_oblivious_in(
  NO_ESCAPE const mats_info<T,U> & RESTRICT io,
  const unsigned row_off, const unsigned col_off,
  const unsigned nRows, const unsigned nCols
) {
  // RECUR_MIN: minimum number for recursion
  // constexpr unsigned RECUR_MIN = numElemsInCacheLine<T>() / 2U;  // cheat ?
  constexpr unsigned RECUR_MIN = 4;
  static_assert( !CONJUGATE
    || std::is_same<T, std::complex<float> >::value
    || std::is_same<T, std::complex<double> >::value
    , "CONJUGATE is only supported by cache_oblivious_in for std::complex<float or double>" );

  if ( nCols > RECUR_MIN || nRows > RECUR_MIN ) {
    if( nRows >= nCols ) {
      const unsigned halfRows = nRows / 2U;
      cache_oblivious_in<T, U, CONJUGATE>( io, row_off, col_off, halfRows, nCols );
      cache_oblivious_in<T, U, CONJUGATE>( io, row_off +halfRows, col_off, nRows - halfRows, nCols );
    } else {
      const unsigned halfCols = nCols / 2U;
      cache_oblivious_in<T, U, CONJUGATE>( io, row_off, col_off, nRows, halfCols );
      cache_oblivious_in<T, U, CONJUGATE>( io, row_off, col_off +halfCols, nRows, nCols - halfCols );
    }
  } else {
    const unsigned rowSize_in = io.rowSize_in;
    const unsigned rowSize_out = io.rowSize_out;
    U * RESTRICT pout = io.pout + col_off * rowSize_out + row_off;
    const T * RESTRICT pin = io.pin + row_off * rowSize_in + col_off;
    tail_transpose_in<T, U, CONJUGATE>( pin, pout, nRows, nCols, rowSize_in, rowSize_out );
  }
}


template <class T, class U, bool CONJUGATE = false>
HEDLEY_NO_THROW
static void cache_oblivious_out(
  NO_ESCAPE const mats_info<T,U> & RESTRICT io,
  const unsigned row_off, const unsigned col_off,
  const unsigned nRows, const unsigned nCols
) {
  // RECUR_MIN: minimum number for recursion
  // constexpr unsigned RECUR_MIN = numElemsInCacheLine<T>() / 2U;  // cheat ?
  constexpr unsigned RECUR_MIN = 4;
  static_assert( !CONJUGATE
    || std::is_same<T, std::complex<float> >::value
    || std::is_same<T, std::complex<double> >::value
    , "CONJUGATE is only supported by cache_oblivious_out for std::complex<float or double>" );

  if ( nCols > RECUR_MIN || nRows > RECUR_MIN ) {
    if( nRows >= nCols ) {
      const unsigned halfRows = nRows / 2U;
      cache_oblivious_out<T, U, CONJUGATE>( io, row_off, col_off, halfRows, nCols );
      cache_oblivious_out<T, U, CONJUGATE>( io, row_off +halfRows, col_off, nRows - halfRows, nCols );
    } else {
      const unsigned halfCols = nCols / 2U;
      cache_oblivious_out<T, U, CONJUGATE>( io, row_off, col_off, nRows, halfCols );
      cache_oblivious_out<T, U, CONJUGATE>( io, row_off, col_off +halfCols, nRows, nCols - halfCols );
    }
  } else {
    const unsigned rowSize_in = io.rowSize_in;
    const unsigned rowSize_out = io.rowSize_out;
    U * RESTRICT pout = io.pout + row_off * rowSize_out + col_off;
    const T * RESTRICT pin = io.pin + col_off * rowSize_in + row_off;
    tail_transpose_out<T, U, CONJUGATE>( pin, pout, nRows, nCols, rowSize_in, rowSize_out );
  }
}


template <class T, class U, bool CONJUGATE = false>
HEDLEY_NO_THROW
static void cache_oblivious_in(
  const mat_info &in, NO_ESCAPE const T * RESTRICT pin,
  const mat_info &out, NO_ESCAPE U * RESTRICT pout )
{
  const mats_info<T,U> io {
    pin, pout,
    in.nRows, in.nCols, in.rowSize,
    out.nRows, out.nCols, out.rowSize
  };
  return cache_oblivious_in<T, U, CONJUGATE>( io, 0, 0, in.nRows, in.nCols );
}


template <class T, class U, bool CONJUGATE = false>
HEDLEY_NO_THROW
static void cache_oblivious_out(
  const mat_info &in, NO_ESCAPE const T * RESTRICT pin,
  const mat_info &out, NO_ESCAPE U * RESTRICT pout )
{
  const mats_info<T,U> io {
    pin, pout,
    in.nRows, in.nCols, in.rowSize,
    out.nRows, out.nCols, out.rowSize
  };
  return cache_oblivious_out<T, U, CONJUGATE>( io, 0, 0, out.nRows, out.nCols );
}


template <class T, class U, bool CONJUGATE = false>
HEDLEY_NO_THROW
static void cache_oblivious_meta(
  const mat_info &in, NO_ESCAPE const T * RESTRICT pin,
  const mat_info &out, NO_ESCAPE U * RESTRICT pout )
{
  const mats_info<T,U> io {
    pin, pout,
    in.nRows, in.nCols, in.rowSize,
    out.nRows, out.nCols, out.rowSize
  };
  if ( in.nRows < in.nCols )
    cache_oblivious_in<T, U, CONJUGATE>( io, 0, 0, in.nRows, in.nCols );
  else
    cache_oblivious_out<T, U, CONJUGATE>( io, 0, 0, out.nRows, out.nCols );
}


}
