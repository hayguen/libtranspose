
#pragma once

#include "transpose_defs.hpp"


namespace transpose
{

//////////////////////////////////////////////////////

template <class T, class U, class FUNC>
HEDLEY_NO_THROW
static void caware_out(
  const mat_info &in, NO_ESCAPE const T * RESTRICT pin,
  const mat_info &out, NO_ESCAPE U * RESTRICT pout )
{
  constexpr unsigned L = ( numElemsInCacheLine<T>() < numElemsInCacheLine<U>() ? numElemsInCacheLine<T>() : numElemsInCacheLine<U>() );
  unsigned row, col;
  // iterate linearly through output matrix indices
  const unsigned N = out.nRows;
  const unsigned M = out.nCols;
  const unsigned out_RS = out.rowSize;
  const unsigned in_RS = in.rowSize;
  const unsigned out_inc = L * out.rowSize;
  const unsigned in_inc = L * in.rowSize;
  unsigned out_row_off, in_row_off, out_off, in_off;
  FUNC f;

  for( row = out_row_off = 0; row + L <= N; row += L, out_row_off += out_inc ) {
    for( col = in_row_off = 0; col + L <= M; col += L, in_row_off += in_inc ) {
      // do transpose on this submatrix
      out_off = out_row_off;
      for( unsigned r = row; r < row + L; ++r, out_off += out_RS ) {
        in_off = in_row_off;
        for( unsigned c = col; c < col + L; ++c, in_off += in_RS )
          pout[out_off+c] = f( pin[in_off+r] );
      }
    }
    // tail columns
    if ( col < M ) {
      out_off = out_row_off;
      for( unsigned r = row; r < N; ++r, out_off += out_RS ) {
        in_off = in_row_off;
        for( unsigned c = col; c < M; ++c, in_off += in_RS )
          pout[out_off+c] = f( pin[in_off+r] );
      }
    }
  }

  // tail rows
  if ( row < N ) {
    for( col = in_row_off = 0; col + L <= M; col += L, in_row_off += in_inc ) {
      // do transpose on this submatrix
      out_off = out_row_off;
      for( unsigned r = row; r < N; ++r, out_off += out_RS ) {
        in_off = in_row_off;
        for( unsigned c = col; c < col + L; ++c, in_off += in_RS )
          pout[out_off+c] = f( pin[in_off+r] );
      }
    }
    // tail columns
    if ( col < M ) {
      // do transpose on this submatrix
      out_off = out_row_off;
      for( unsigned r = row; r < N; ++r, out_off += out_RS ) {
        in_off = in_row_off;
        for( unsigned c = col; c < M; ++c, in_off += in_RS )
          pout[out_off+c] = f( pin[in_off+r] );
      }
    }
  }
}

//////////////////////////////////////////////////////

// cache aware: 5th try
template <class T, class U, class FUNC>
HEDLEY_NO_THROW
static void caware_in(
  const mat_info &in, NO_ESCAPE const T * RESTRICT pin,
  const mat_info &out, NO_ESCAPE U * RESTRICT pout )
{
  constexpr unsigned L = ( numElemsInCacheLine<T>() < numElemsInCacheLine<U>() ? numElemsInCacheLine<T>() : numElemsInCacheLine<U>() );
  unsigned row, col;
  // iterate linearly through input matrix indices
  const unsigned N = in.nRows;
  const unsigned M = in.nCols;
  const unsigned out_RS = out.rowSize;
  const unsigned in_RS = in.rowSize;
  const unsigned out_inc = L * out.rowSize;
  const unsigned in_inc = L * in.rowSize;
  unsigned out_row_off, in_row_off, out_off, in_off;
  FUNC f;

  for( row = in_row_off = 0; row + L <= N; row += L, in_row_off += in_inc ) {
    for( col = out_row_off = 0; col + L <= M; col += L, out_row_off += out_inc ) {
      // do transpose on this submatrix
      in_off = in_row_off;
      for( unsigned r = row; r < row + L; ++r, in_off += in_RS ) {
        out_off = out_row_off;
        for( unsigned c = col; c < col + L; ++c, out_off += out_RS )
          pout[out_off+r] = f( pin[in_off+c] );
      }
    }
    // tail columns
    if ( col < M ) {
      in_off = in_row_off;
      for( unsigned r = row; r < N; ++r, in_off += in_RS ) {
        out_off = out_row_off;
        for( unsigned c = col; c < M; ++c, out_off += out_RS )
          pout[out_off+r] = f( pin[in_off+c] );
      }
    }
  }

  // tail rows
  if ( row < N ) {
    for( col = out_row_off = 0; col + L <= M; col += L, out_row_off += out_inc ) {
      // do transpose on this submatrix
      in_off = in_row_off;
      for( unsigned r = row; r < N; ++r, in_off += in_RS ) {
        out_off = out_row_off;
        for( unsigned c = col; c < col + L; ++c, out_off += out_RS )
          pout[out_off+r] = f( pin[in_off+c] );
      }
    }
    // tail columns
    if ( col < M ) {
      // do transpose on this submatrix
      in_off = in_row_off;
      for( unsigned r = row; r < N; ++r, in_off += in_RS ) {
        out_off = out_row_off;
        for( unsigned c = col; c < M; ++c, out_off += out_RS )
          pout[out_off+r] = f( pin[in_off+c] );
      }
    }
  }
}

//////////////////////////////////////////////////////

template <class T, class U, class FUNC>
HEDLEY_NO_THROW
static void caware_meta(
  const mat_info &in, NO_ESCAPE const T * RESTRICT pin,
  const mat_info &out, NO_ESCAPE U * RESTRICT pout )
{
  //if ( in.nRows * sizeof(T) < in.nCols * sizeof(U) )
  if ( in.nRows < in.nCols )
    caware_in<T, U, FUNC>( in, pin, out, pout );
  else
    caware_out<T, U, FUNC>( in, pin, out, pout );
}

}
