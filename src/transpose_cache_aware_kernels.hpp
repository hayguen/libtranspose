
#pragma once

#include "transpose_defs.hpp"

#include <memory>


namespace transpose
{


template <class T>
ALWAYS_INLINE HEDLEY_NO_THROW
static void caware_kernel_out_tail(
  NO_ESCAPE const T * RESTRICT pin, NO_ESCAPE T * RESTRICT pout,
  const unsigned nRows, const unsigned nCols, const unsigned rowSizeA, const unsigned rowSizeB )
{
  unsigned out_off = 0; // out_row_off;
  // for( unsigned r = row; r < N; ++r, out_off += rowSizeB ) {
  for( unsigned r = 0; r < nRows; ++r, out_off += rowSizeB ) {
    unsigned in_off = 0; // in_row_off;
    // for( unsigned c = col; c < M; ++c, in_off += rowSizeA )
    for( unsigned c = 0; c < nCols; ++c, in_off += rowSizeA )
      pout[out_off+c] = pin[in_off+r];
  }
}

//////////////////////////////////////////////////////

template <class T>
ALWAYS_INLINE HEDLEY_NO_THROW
static void caware_kernel_in_tail(
  NO_ESCAPE const T * RESTRICT pin, NO_ESCAPE T * RESTRICT pout,
  const unsigned nRows, const unsigned nCols, const unsigned rowSizeA, const unsigned rowSizeB )
{
  unsigned in_off = 0; // in_row_off;
  // for( unsigned r = row; r < N; ++r, in_off += rowSizeA ) {
  for( unsigned r = 0; r < nRows; ++r, in_off += rowSizeA ) {
    unsigned out_off = 0; // out_row_off;
    // for( unsigned c = col; c < M; ++c, out_off += rowSizeB )
    for( unsigned c = 0; c < nCols; ++c, out_off += rowSizeB )
      pout[out_off+r] = pin[in_off+c];
  }
}


//////////////////////////////////////////////////////

template <class T, class KERNEL>
struct caware_kernel
{
  static constexpr bool HAS_AA = KERNEL::HAS_AA;
  static constexpr unsigned KERNEL_SZ = KERNEL::KERNEL_SZ;

  HEDLEY_NO_THROW
  static void uu_out(
    const mat_info &in, NO_ESCAPE const T * RESTRICT pin,
    const mat_info &out, NO_ESCAPE T * RESTRICT pout )
  {
    // iterate linearly through output matrix indices
    const unsigned N = out.nRows, M = out.nCols;
    const unsigned rowSizeB = out.rowSize, rowSizeA = in.rowSize;
    const unsigned out_inc = KERNEL_SZ * out.rowSize, in_inc = KERNEL_SZ * in.rowSize;
    unsigned out_row_off, in_row_off, row, col;

    for( row = out_row_off = 0; row + KERNEL_SZ <= N; row += KERNEL_SZ, out_row_off += out_inc ) {
      for( col = in_row_off = 0; col + KERNEL_SZ <= M; col += KERNEL_SZ, in_row_off += in_inc ) {
        KERNEL::op_uu( &pin[in_row_off+row], &pout[out_row_off+col], rowSizeA, rowSizeB );
      }
      if ( col < M )  // tail columns with KERNEL_SZ rows
        caware_kernel_out_tail( &pin[in_row_off+row], &pout[out_row_off+col], KERNEL_SZ, M - col, rowSizeA, rowSizeB );
    }
    if ( row < N ) {  // tail rows: #rows < KERNEL_SZ, #cols == KERNEL_SZ
      for( col = in_row_off = 0; col + KERNEL_SZ <= M; col += KERNEL_SZ, in_row_off += in_inc )
        caware_kernel_out_tail( &pin[in_row_off+row], &pout[out_row_off+col], N - row, KERNEL_SZ, rowSizeA, rowSizeB );
      if ( col < M )  // tail columns - #rows < KERNEL_SZ, #cols < KERNEL_SZ
        caware_kernel_out_tail( &pin[in_row_off+row], &pout[out_row_off+col], N - row, M - col, rowSizeA, rowSizeB );
    }
  }

  HEDLEY_NO_THROW
  static void uu_in(
    const mat_info &in, NO_ESCAPE const T * RESTRICT pin,
    const mat_info &out, NO_ESCAPE T * RESTRICT pout )
  {
    // iterate linearly through input matrix indices
    const unsigned N = in.nRows, M = in.nCols;
    const unsigned rowSizeB = out.rowSize, rowSizeA = in.rowSize;
    const unsigned out_inc = KERNEL_SZ * rowSizeB, in_inc = KERNEL_SZ * rowSizeA;
    unsigned out_row_off, in_row_off, row, col;

    for( row = in_row_off = 0; row + KERNEL_SZ <= N; row += KERNEL_SZ, in_row_off += in_inc ) {
      for( col = out_row_off = 0; col + KERNEL_SZ <= M; col += KERNEL_SZ, out_row_off += out_inc ) {
        KERNEL::op_uu( &pin[in_row_off+col], &pout[out_row_off+row], rowSizeA, rowSizeB );
      }
      if ( col < M )  // tail columns with KERNEL_SZ rows
        caware_kernel_in_tail( &pin[in_row_off+col], &pout[out_row_off+row], KERNEL_SZ, M - col, rowSizeA, rowSizeB );
    }
    if ( row < N ) {  // tail rows: #rows < KERNEL_SZ, #cols == KERNEL_SZ
      for( col = out_row_off = 0; col + KERNEL_SZ <= M; col += KERNEL_SZ, out_row_off += out_inc )
        caware_kernel_in_tail( &pin[in_row_off+col], &pout[out_row_off+row], N - row, KERNEL_SZ, rowSizeA, rowSizeB );
      if ( col < M )  // tail columns - #rows < KERNEL_SZ, #cols < KERNEL_SZ
        caware_kernel_in_tail( &pin[in_row_off+col], &pout[out_row_off+row], N - row, M - col, rowSizeA, rowSizeB );
    }
  }

  HEDLEY_NO_THROW
  static void uu_meta(
    const mat_info &in, NO_ESCAPE const T * RESTRICT pin,
    const mat_info &out, NO_ESCAPE T * RESTRICT pout )
  {
    if ( in.nRows < in.nCols )
      uu_in( in, pin, out, pout );
    else
      uu_out( in, pin, out, pout );
  }


  HEDLEY_NO_THROW
  static void aa_out(
    const mat_info &in, NO_ESCAPE const T * RESTRICT pin,
    const mat_info &out, NO_ESCAPE T * RESTRICT pout )
  {
    // iterate linearly through output matrix indices
    const unsigned Nup = KERNEL_SZ * ( (out.nRows + KERNEL_SZ - 1) / KERNEL_SZ);
    const unsigned Mup = KERNEL_SZ * ( (out.nCols + KERNEL_SZ - 1) / KERNEL_SZ);
    const unsigned rowSizeB = out.rowSize, rowSizeA = in.rowSize;
    const unsigned out_inc = KERNEL_SZ * out.rowSize, in_inc = KERNEL_SZ * in.rowSize;
    unsigned out_row_off, in_row_off, row, col;

    for( row = out_row_off = 0; row + KERNEL_SZ <= Nup; row += KERNEL_SZ, out_row_off += out_inc ) {
      for( col = in_row_off = 0; col + KERNEL_SZ <= Mup; col += KERNEL_SZ, in_row_off += in_inc ) {
        KERNEL::op_aa( &pin[in_row_off+row], &pout[out_row_off+col], rowSizeA, rowSizeB );
      }
    }
  }

  HEDLEY_NO_THROW
  static void aa_in(
    const mat_info &in, NO_ESCAPE const T * RESTRICT pin,
    const mat_info &out, NO_ESCAPE T * RESTRICT pout )
  {
    // iterate linearly through input matrix indices
    const unsigned Nup = KERNEL_SZ * ( (in.nRows + KERNEL_SZ - 1) / KERNEL_SZ);
    const unsigned Mup = KERNEL_SZ * ( (in.nCols + KERNEL_SZ - 1) / KERNEL_SZ);
    const unsigned rowSizeB = out.rowSize, rowSizeA = in.rowSize;
    const unsigned out_inc = KERNEL_SZ * rowSizeB, in_inc = KERNEL_SZ * rowSizeA;
    unsigned out_row_off, in_row_off, row, col;

    for( row = in_row_off = 0; row + KERNEL_SZ <= Nup; row += KERNEL_SZ, in_row_off += in_inc ) {
      for( col = out_row_off = 0; col + KERNEL_SZ <= Mup; col += KERNEL_SZ, out_row_off += out_inc ) {
        KERNEL::op_aa( &pin[in_row_off+col], &pout[out_row_off+row], rowSizeA, rowSizeB );
      }
    }
  }

  HEDLEY_NO_THROW
  static void aa_meta(
    const mat_info &in, NO_ESCAPE const T * RESTRICT pin,
    const mat_info &out, NO_ESCAPE T * RESTRICT pout )
  {
    if ( in.nRows < in.nCols )
      aa_in( in, pin, out, pout );
    else
      aa_out( in, pin, out, pout );
  }

  HEDLEY_NO_THROW   HEDLEY_CONST
  static bool aa_possible(
    const mat_info &in, NO_ESCAPE const T * RESTRICT pin,
    const mat_info &out, NO_ESCAPE const T * RESTRICT pout )
  {
    if (!HAS_AA)
      return false;

    unsigned out_mod = (out.rowSize % KERNEL_SZ);
    unsigned in_mod = (in.rowSize % KERNEL_SZ);

    std::size_t space_inp = in.nRows * in.rowSize * sizeof(T);
    void * raw_inp = const_cast<T*>(pin);
    void * res_inp = std::align( KERNEL_SZ * sizeof(T), space_inp, raw_inp, space_inp );

    std::size_t space_out = out.nRows * out.rowSize * sizeof(T);
    void * raw_out = const_cast<T*>(pout);
    void * res_out = std::align( KERNEL_SZ * sizeof(T), space_out, raw_out, space_out );

    return ( (!out_mod && !in_mod) && (raw_inp == res_inp) && (raw_out == res_out) );
  }

};


}
