
#include "transpose_defs.hpp"

#if !defined(KERNEL_NAME) || !defined(KERNEL_INIT) || !defined(KERNEL_OP_UU) || !defined(KERNEL_OP_AA)
# error specialization required the preprocessor macros KERNEL_NAME, KERNEL_INIT, KERNEL_OP_UU and KERNEL_OP_AA
#endif

namespace transpose
{


template <class T>
struct caware_kernel<T, transpose_kernels::KERNEL_NAME<T> >
{
  using KERNEL = transpose_kernels::KERNEL_NAME<T>;
  using BaseType = typename KERNEL::BaseType;
  static constexpr bool HAS_AA = KERNEL::HAS_AA;
  static constexpr unsigned KERNEL_SZ = KERNEL::KERNEL_SZ;

  static void uu_out( const mat_info<T> &in, mat_info<T> &out ) {
    // iterate linearly through output matrix indices
    const unsigned N = out.nRows, M = out.nCols;
    const unsigned rowSizeB = out.rowSize, rowSizeA = in.rowSize;
    const unsigned out_inc = KERNEL_SZ * out.rowSize, in_inc = KERNEL_SZ * in.rowSize;
    unsigned out_row_off, in_row_off, row, col;
    T * RESTRICT pout = out.vector;
    const T * RESTRICT pin = in.vector;

    KERNEL_INIT();
    for( row = out_row_off = 0; row + KERNEL_SZ <= N; row += KERNEL_SZ, out_row_off += out_inc ) {
      for( col = in_row_off = 0; col + KERNEL_SZ <= M; col += KERNEL_SZ, in_row_off += in_inc ) {
        const T * RESTRICT A_ = &pin[in_row_off+row];
        T * RESTRICT B_ = &pout[out_row_off+col];
        KERNEL_OP_UU();
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

  static void uu_in( const mat_info<T> &in, mat_info<T> &out ) {
    // iterate linearly through input matrix indices
    const unsigned N = in.nRows, M = in.nCols;
    const unsigned rowSizeB = out.rowSize, rowSizeA = in.rowSize;
    const unsigned out_inc = KERNEL_SZ * rowSizeB, in_inc = KERNEL_SZ * rowSizeA;
    unsigned out_row_off, in_row_off, row, col;
    T * RESTRICT pout = out.vector;
    const T * RESTRICT pin = in.vector;

    KERNEL_INIT();
    for( row = in_row_off = 0; row + KERNEL_SZ <= N; row += KERNEL_SZ, in_row_off += in_inc ) {
      for( col = out_row_off = 0; col + KERNEL_SZ <= M; col += KERNEL_SZ, out_row_off += out_inc ) {
        const T * RESTRICT A_ = &pin[in_row_off+col];
        T * RESTRICT B_ = &pout[out_row_off+row];
        KERNEL_OP_UU();
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

  static void uu_meta( const mat_info<T> &in, mat_info<T> &out ) {
    if ( in.nRows < in.nCols )
      uu_in( in, out );
    else
      uu_out( in, out );
  }


  static void aa_out( const mat_info<T> &in, mat_info<T> &out ) {
    // iterate linearly through output matrix indices
    const unsigned Nup = KERNEL_SZ * ( (out.nRows + KERNEL_SZ - 1) / KERNEL_SZ);
    const unsigned Mup = KERNEL_SZ * ( (out.nCols + KERNEL_SZ - 1) / KERNEL_SZ);
    const unsigned rowSizeB = out.rowSize, rowSizeA = in.rowSize;
    const unsigned out_inc = KERNEL_SZ * out.rowSize, in_inc = KERNEL_SZ * in.rowSize;
    unsigned out_row_off, in_row_off, row, col;
    T * RESTRICT pout = out.vector;
    const T * RESTRICT pin = in.vector;

    KERNEL_INIT();
    for( row = out_row_off = 0; row + KERNEL_SZ <= Nup; row += KERNEL_SZ, out_row_off += out_inc ) {
      for( col = in_row_off = 0; col + KERNEL_SZ <= Mup; col += KERNEL_SZ, in_row_off += in_inc ) {
        const T * RESTRICT A_ = &pin[in_row_off+row];
        T * RESTRICT B_ = &pout[out_row_off+col];
        KERNEL_OP_AA();
      }
    }
  }

  static void aa_in( const mat_info<T> &in, mat_info<T> &out ) {
    // iterate linearly through input matrix indices
    const unsigned Nup = KERNEL_SZ * ( (in.nRows + KERNEL_SZ - 1) / KERNEL_SZ);
    const unsigned Mup = KERNEL_SZ * ( (in.nCols + KERNEL_SZ - 1) / KERNEL_SZ);
    const unsigned rowSizeB = out.rowSize, rowSizeA = in.rowSize;
    const unsigned out_inc = KERNEL_SZ * rowSizeB, in_inc = KERNEL_SZ * rowSizeA;
    unsigned out_row_off, in_row_off, row, col;
    T * RESTRICT pout = out.vector;
    const T * RESTRICT pin = in.vector;

    KERNEL_INIT();
    for( row = in_row_off = 0; row + KERNEL_SZ <= Nup; row += KERNEL_SZ, in_row_off += in_inc ) {
      for( col = out_row_off = 0; col + KERNEL_SZ <= Mup; col += KERNEL_SZ, out_row_off += out_inc ) {
        const T * RESTRICT A_ = &pin[in_row_off+col];
        T * RESTRICT B_ = &pout[out_row_off+row];
        KERNEL_OP_AA();
      }
    }
  }

  static void aa_meta( const mat_info<T> &in, mat_info<T> &out ) {
    if ( in.nRows < in.nCols )
      aa_in( in, out );
    else
      aa_out( in, out );
  }

  static bool aa_possible( const mat_info<T> &in, const mat_info<T> &out ) {
    if (!HAS_AA)
      return false;

    unsigned out_mod = (out.rowSize % KERNEL_SZ);
    unsigned in_mod = (in.rowSize % KERNEL_SZ);

    std::size_t space_inp = in.nRows * in.rowSize * sizeof(T);
    void * raw_inp = const_cast<T*>(in.vector);
    void * res_inp = std::align( KERNEL_SZ * sizeof(T), space_inp, raw_inp, space_inp );

    std::size_t space_out = out.nRows * out.rowSize * sizeof(T);
    void * raw_out = const_cast<T*>(out.vector);
    void * res_out = std::align( KERNEL_SZ * sizeof(T), space_out, raw_out, space_out );

    return ( (!out_mod && !in_mod) && (raw_inp == res_inp) && (raw_out == res_out) );
  }

};


} // namespace

#undef KERNEL_NAME
#undef KERNEL_INIT
#undef KERNEL_OP_UU
#undef KERNEL_OP_AA