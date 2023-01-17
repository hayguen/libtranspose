
#pragma once

#if defined(HAVE_SYSTEM_MKL)
#  include <mkl/mkl.h>
#  define HAVE_MKL_KERNEL 1
#endif

#if defined(HAVE_ONEAPI_MKL)
#  include <mkl.h>
#  define HAVE_MKL_KERNEL 1
#endif

#ifdef HAVE_MKL_KERNEL

#include <transpose_defs.hpp>
#include <complex>
#include <type_traits>

//////////////////////////////////////////////////////

namespace transpose
{

// https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-like-extensions/mkl-omatcopy.html
// - no idea what happens with values corresponding to float NAN
//   needs to be tested - but for a 'simple' performance comparison, we can skip
// - bigger datatypes MKL_Complex8 and MKL_Complex16 for std::complex<float> and std::complex<double> possible,
//   see https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/overview/c-datatypes-specific-to-intel-mkl.html

// => unfortunately, performance is not that good :-(
//   Intel OneAPI IPP's ippiTranspose_*() is better

template <class T>
static void trans_mkl32(
  const mat_info &in, NO_ESCAPE const T * RESTRICT pin,
  const mat_info &out, NO_ESCAPE T * RESTRICT pout )
{
  using MklType = float;
  static_assert( sizeof(T) == sizeof(MklType), "" );

  mkl_somatcopy('R', 'T',
    in.nRows, in.nCols, 1.0F, reinterpret_cast<const MklType*>( pin ), in.rowSize,
    reinterpret_cast<MklType*>( pout ), out.rowSize
  );
}

template <class T>
static void trans_mkl64(
  const mat_info &in, NO_ESCAPE const T * RESTRICT pin,
  const mat_info &out, NO_ESCAPE T * RESTRICT pout )
{
  using MklType = double;
  static_assert( sizeof(T) == sizeof(MklType), "" );

  mkl_domatcopy('R', 'T',
    in.nRows, in.nCols, 1.0F, reinterpret_cast<const MklType*>( pin ), in.rowSize,
    reinterpret_cast<MklType*>( pout ), out.rowSize
  );
}

template <class T, bool CONJUGATE = false>
static void trans_mkl32c(
  const mat_info &in, NO_ESCAPE const T * RESTRICT pin,
  const mat_info &out, NO_ESCAPE T * RESTRICT pout )
{
  using MklType = MKL_Complex8;
  static_assert( sizeof(T) == sizeof(MklType), "" );
  static_assert( !CONJUGATE || std::is_same<T, std::complex<float> >::value
    , "CONJUGATE is only supported by trans_mkl32c for std::complex<float>" );
  MklType alpha;
  alpha.real = 1.0F;
  alpha.imag = 0.0F;

  mkl_comatcopy('R', CONJUGATE ? 'C' : 'T',
    in.nRows, in.nCols, alpha, reinterpret_cast<const MklType*>( pin ), in.rowSize,
    reinterpret_cast<MklType*>( pout ), out.rowSize
  );
}

template <class T, bool CONJUGATE = false>
static void trans_mkl64c(
  const mat_info &in, NO_ESCAPE const T * RESTRICT pin,
  const mat_info &out, NO_ESCAPE T * RESTRICT pout )
{
  using MklType = MKL_Complex16;
  static_assert( sizeof(T) == sizeof(MklType), "" );
  static_assert( !CONJUGATE || std::is_same<T, std::complex<double> >::value
    , "CONJUGATE is only supported by trans_mkl64c for std::complex<double>" );
  MklType alpha;
  alpha.real = 1.0;
  alpha.imag = 0.0;

  mkl_zomatcopy('R', CONJUGATE ? 'C' : 'T',
    in.nRows, in.nCols, alpha, reinterpret_cast<const MklType*>( pin ), in.rowSize,
    reinterpret_cast<MklType*>( pout ), out.rowSize
  );
}

} // namespace

#endif
