
#pragma once

#include "transpose_defs.hpp"


namespace transpose
{

// see https://en.wikipedia.org/wiki/Cache-oblivious_algorithm
// see Matrix::cachetranspose()
//   from https://stackoverflow.com/questions/5200338/a-cache-efficient-matrix-transpose-program
//   but with important changes
//     - check for recursion first - assuming bigger matrices
//     - fix 2nd recursion for uneven nRows or nCols

template <class T, class U, class FUNC>
static void cache_oblivious_in( const mat_info<T> &in, mat_info<U> &out,
  const unsigned row_off, const unsigned col_off,
  const unsigned nRows, const unsigned nCols
) {
  // RECUR_MIN: minimum number for recursion
  // constexpr unsigned RECUR_MIN = numElemsInCacheLine<T>() / 2U;  // cheat ?
  constexpr unsigned RECUR_MIN = 4;

  if ( nCols > RECUR_MIN || nRows > RECUR_MIN ) {
    if( nRows >= nCols ) {
      const unsigned halfRows = nRows / 2U;
      cache_oblivious_in<T, U, FUNC>( in, out, row_off, col_off, halfRows, nCols );
      cache_oblivious_in<T, U, FUNC>( in, out, row_off +halfRows, col_off, nRows - halfRows, nCols );
    } else {
      const unsigned halfCols = nCols / 2U;
      cache_oblivious_in<T, U, FUNC>( in, out, row_off, col_off, nRows, halfCols );
      cache_oblivious_in<T, U, FUNC>( in, out, row_off, col_off +halfCols, nRows, nCols - halfCols );
    }
  } else {
    U * RESTRICT pout = out.vector + col_off * out.rowSize + row_off;
    const T * RESTRICT pin = in.vector + row_off * in.rowSize + col_off;
    unsigned in_row_off, out_row_off;
    FUNC f;

    for( unsigned row = in_row_off = 0; row < nRows; ++row, in_row_off += in.rowSize ) {
      for( unsigned col = out_row_off = 0; col < nCols; ++col, out_row_off += out.rowSize ) {
        // out[col][row] = in[row][col];
        pout[out_row_off+row] = f( pin[in_row_off+col] );
      }
    }
  }
}


template <class T, class U, class FUNC>
static void cache_oblivious_out( const mat_info<T> &in, mat_info<U> &out,
  const unsigned row_off, const unsigned col_off,
  const unsigned nRows, const unsigned nCols
) {
  // RECUR_MIN: minimum number for recursion
  // constexpr unsigned RECUR_MIN = numElemsInCacheLine<T>() / 2U;  // cheat ?
  constexpr unsigned RECUR_MIN = 4;

  if ( nCols > RECUR_MIN || nRows > RECUR_MIN ) {
    if( nRows >= nCols ) {
      const unsigned halfRows = nRows / 2U;
      cache_oblivious_out<T, U, FUNC>( in, out, row_off, col_off, halfRows, nCols );
      cache_oblivious_out<T, U, FUNC>( in, out, row_off +halfRows, col_off, nRows - halfRows, nCols );
    } else {
      const unsigned halfCols = nCols / 2U;
      cache_oblivious_out<T, U, FUNC>( in, out, row_off, col_off, nRows, halfCols );
      cache_oblivious_out<T, U, FUNC>( in, out, row_off, col_off +halfCols, nRows, nCols - halfCols );
    }
  } else {
    // U * RESTRICT pout = out.vector + col_off * out.rowSize + row_off;
    // const T * RESTRICT pin = in.vector + row_off * in.rowSize + col_off;
    U * RESTRICT pout = out.vector + row_off * out.rowSize + col_off;
    const T * RESTRICT pin = in.vector + col_off * in.rowSize + row_off;
    unsigned in_row_off, out_row_off;
    FUNC f;

    for( unsigned row = out_row_off = 0; row < nRows; ++row, out_row_off += out.rowSize ) {
      for( unsigned col = in_row_off = 0; col < nCols; ++col, in_row_off += in.rowSize ) {
        // out[row][col] = in[col][row];
        pout[out_row_off+col] = f( pin[in_row_off+row] );
      }
    }
  }
}


template <class T, class U, class FUNC>
static void cache_oblivious_in( const mat_info<T> &in, mat_info<U> &out ) {
  return cache_oblivious_in<T, U, FUNC>( in, out, 0, 0, in.nRows, in.nCols );
}


template <class T, class U, class FUNC>
static void cache_oblivious_out( const mat_info<T> &in, mat_info<U> &out ) {
  return cache_oblivious_out<T, U, FUNC>( in, out, 0, 0, out.nRows, out.nCols );
}


template <class T, class U, class FUNC>
static void cache_oblivious_meta( const mat_info<T> &in, mat_info<U> &out ) {
  if ( in.nRows < in.nCols )
    cache_oblivious_in<T, U, FUNC>( in, out, 0, 0, in.nRows, in.nCols );
  else
    cache_oblivious_out<T, U, FUNC>( in, out, 0, 0, out.nRows, out.nCols );
}


}
