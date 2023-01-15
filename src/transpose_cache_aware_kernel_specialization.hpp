
#include "transpose_defs.hpp"

#if !defined(KERNEL_NAME) || !defined(KERNEL_INIT) || !defined(KERNEL_OP_UU) || !defined(KERNEL_OP_AA)
# error specialization required the preprocessor macros KERNEL_NAME, KERNEL_INIT, KERNEL_OP_UU and KERNEL_OP_AA
#endif

namespace transpose
{


template <class T, bool CONJUGATE_TPL>
struct caware_kernel<T, CONJUGATE_TPL, transpose_kernels::KERNEL_NAME<T, CONJUGATE_TPL> >
{
  using KERNEL = transpose_kernels::KERNEL_NAME<T, CONJUGATE_TPL>;
  using BaseType = typename KERNEL::BaseType;
  static constexpr bool HAS_AA = KERNEL::HAS_AA;
  static constexpr unsigned KERNEL_SZ = KERNEL::KERNEL_SZ;
  static constexpr bool CONJUGATE = KERNEL::CONJUGATE;
  static_assert( CONJUGATE_TPL == CONJUGATE, "mismatching template parameters of caware_kernel and it's kernel" );

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

    KERNEL_INIT();
    for( row = out_row_off = 0; row + KERNEL_SZ <= N; row += KERNEL_SZ, out_row_off += out_inc ) {
      for( col = in_row_off = 0; col + KERNEL_SZ <= M; col += KERNEL_SZ, in_row_off += in_inc ) {
        const T * RESTRICT A_ = &pin[in_row_off+row];
        T * RESTRICT B_ = &pout[out_row_off+col];
        KERNEL_OP_UU();
      }
      if ( col < M )  // tail columns with KERNEL_SZ rows
        tail_transpose_out<T, T, CONJUGATE>( &pin[in_row_off+row], &pout[out_row_off+col], KERNEL_SZ, M - col, rowSizeA, rowSizeB );
    }
    if ( row < N ) {  // tail rows: #rows < KERNEL_SZ, #cols == KERNEL_SZ
      for( col = in_row_off = 0; col + KERNEL_SZ <= M; col += KERNEL_SZ, in_row_off += in_inc )
        tail_transpose_out<T, T, CONJUGATE>( &pin[in_row_off+row], &pout[out_row_off+col], N - row, KERNEL_SZ, rowSizeA, rowSizeB );
      if ( col < M )  // tail columns - #rows < KERNEL_SZ, #cols < KERNEL_SZ
        tail_transpose_out<T, T, CONJUGATE>( &pin[in_row_off+row], &pout[out_row_off+col], N - row, M - col, rowSizeA, rowSizeB );
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

    KERNEL_INIT();
    for( row = in_row_off = 0; row + KERNEL_SZ <= N; row += KERNEL_SZ, in_row_off += in_inc ) {
      for( col = out_row_off = 0; col + KERNEL_SZ <= M; col += KERNEL_SZ, out_row_off += out_inc ) {
        const T * RESTRICT A_ = &pin[in_row_off+col];
        T * RESTRICT B_ = &pout[out_row_off+row];
        KERNEL_OP_UU();
      }
      if ( col < M )  // tail columns with KERNEL_SZ rows
        tail_transpose_in<T, T, CONJUGATE>( &pin[in_row_off+col], &pout[out_row_off+row], KERNEL_SZ, M - col, rowSizeA, rowSizeB );
    }
    if ( row < N ) {  // tail rows: #rows < KERNEL_SZ, #cols == KERNEL_SZ
      for( col = out_row_off = 0; col + KERNEL_SZ <= M; col += KERNEL_SZ, out_row_off += out_inc )
        tail_transpose_in<T, T, CONJUGATE>( &pin[in_row_off+col], &pout[out_row_off+row], N - row, KERNEL_SZ, rowSizeA, rowSizeB );
      if ( col < M )  // tail columns - #rows < KERNEL_SZ, #cols < KERNEL_SZ
        tail_transpose_in<T, T, CONJUGATE>( &pin[in_row_off+col], &pout[out_row_off+row], N - row, M - col, rowSizeA, rowSizeB );
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

    KERNEL_INIT();
    for( row = out_row_off = 0; row + KERNEL_SZ <= Nup; row += KERNEL_SZ, out_row_off += out_inc ) {
      for( col = in_row_off = 0; col + KERNEL_SZ <= Mup; col += KERNEL_SZ, in_row_off += in_inc ) {
        const T * RESTRICT A_ = &pin[in_row_off+row];
        T * RESTRICT B_ = &pout[out_row_off+col];
        KERNEL_OP_AA();
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

    KERNEL_INIT();
    for( row = in_row_off = 0; row + KERNEL_SZ <= Nup; row += KERNEL_SZ, in_row_off += in_inc ) {
      for( col = out_row_off = 0; col + KERNEL_SZ <= Mup; col += KERNEL_SZ, out_row_off += out_inc ) {
        const T * RESTRICT A_ = &pin[in_row_off+col];
        T * RESTRICT B_ = &pout[out_row_off+row];
        KERNEL_OP_AA();
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


} // namespace
