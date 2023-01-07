
#pragma once

#include "transpose_defs.hpp"


namespace transpose
{

//////////////////////////////////////////////////////

template <class T, class U, class FUNC >
static void naive_in( const mat_info<T> &in, mat_info<U> &out ) {
  // iterate linearly through input matrix indices
  const unsigned N = in.nRows;
  const unsigned M = in.nCols;
  U * RESTRICT pout = out.vector;
  const T * RESTRICT pin = in.vector;
  unsigned out_off, in_off;
  FUNC f;

  for( unsigned r = in_off = 0; r < N; ++r, in_off += in.rowSize ) {
    for( unsigned c = out_off = 0; c < M; ++c, out_off += out.rowSize ) {
      // out(c,r) = in(r,c);
      pout[out_off+r] = f( pin[in_off+c] );
    }
  }
}

//////////////////////////////////////////////////////

template <class T, class U, class FUNC >
static void naive_out( const mat_info<T> &in, mat_info<U> &out ) {
  // iterate linearly through output matrix indices
  const unsigned N = out.nRows;
  const unsigned M = out.nCols;
  U * RESTRICT pout = out.vector;
  const T * RESTRICT pin = in.vector;
  unsigned out_off, in_off;
  FUNC f;

  for( unsigned r = out_off = 0; r < N; ++r, out_off += out.rowSize ) {
    for( unsigned c = in_off = 0; c < M; ++c, in_off += in.rowSize ) {
      // out(r,c) = in(c,r);
      pout[out_off+c] = f( pin[in_off+r] );
    }
  }
}

//////////////////////////////////////////////////////

// template <class T, class U, class FUNC = FuncId<T,U> >
template <class T, class U, class FUNC >
static void naive_meta( const mat_info<T> &in, mat_info<U> &out ) {
  if ( in.nRows * sizeof(T) < in.nCols * sizeof(U) )
    naive_in<T, U, FUNC>( in, out );
  else
    naive_out<T, U, FUNC>( in, out );
}

}
