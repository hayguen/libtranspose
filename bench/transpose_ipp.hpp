
#pragma once

#if defined(HAVE_ONEAPI_IPP)
#  include <ippi.h>
#  define HAVE_IPP_KERNEL 1
#endif

#ifdef HAVE_IPP_KERNEL

#include <transpose_defs.hpp>

//////////////////////////////////////////////////////

namespace transpose
{

// https://www.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/top/volume-2-image-processing/image-data-exchange-and-initialization-functions/transpose.html
// unfortunately no 64bit variant for double or std::complex<float>

// => looks to give best performance :-)
//   performance is similar to SSE_4x4x32Kernel

template <class T>
static void trans_ipp32( const mat_info<T> &in, mat_info<T> &out ) {
  using IppType = Ipp32s;
  static_assert( sizeof(T) == sizeof(IppType), "" );

  IppiSize roi = { int(in.nCols), int(in.nRows) };  // width, then height
  ippiTranspose_32s_C1R(
    reinterpret_cast<const IppType*>( in.vector ), int(in.rowSize * sizeof(T)),
    reinterpret_cast<IppType*>( out.vector ), int(out.rowSize * sizeof(T)),
    roi
  );
}

template <class T>
static void trans_ipp16( const mat_info<T> &in, mat_info<T> &out ) {
  using IppType = Ipp16s;
  static_assert( sizeof(T) == sizeof(IppType), "" );

  IppiSize roi = { int(in.nCols), int(in.nRows) };  // width, then height
  ippiTranspose_16s_C1R(
    reinterpret_cast<const IppType*>( in.vector ), int(in.rowSize * sizeof(T)),
    reinterpret_cast<IppType*>( out.vector ), int(out.rowSize * sizeof(T)),
    roi
  );
}

template <class T>
static void trans_ipp8( const mat_info<T> &in, mat_info<T> &out ) {
  using IppType = Ipp8u;
  static_assert( sizeof(T) == sizeof(IppType), "" );

  IppiSize roi = { int(in.nCols), int(in.nRows) };  // width, then height
  ippiTranspose_8u_C1R(
    reinterpret_cast<const IppType*>( in.vector ), int(in.rowSize * sizeof(T)),
    reinterpret_cast<IppType*>( out.vector ), int(out.rowSize * sizeof(T)),
    roi
  );
}

}

#endif
