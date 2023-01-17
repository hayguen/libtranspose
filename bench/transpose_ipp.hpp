
#pragma once

#if defined(HAVE_ONEAPI_IPP)
#  include <ippi.h>
#  include <ippcore.h>
#  define HAVE_IPP_KERNEL 1
#endif

#ifdef HAVE_IPP_KERNEL

#include <transpose_defs.hpp>
#include <cstdlib>
#include <cstdio>

//////////////////////////////////////////////////////

namespace transpose
{

// https://www.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/top/volume-2-image-processing/image-data-exchange-and-initialization-functions/transpose.html
// unfortunately no 64bit variant for double or std::complex<float>

// => looks to give best performance :-)
//   performance is similar to SSE_4x4x32Kernel

void ipp_single_thread(int verbose) {
  // https://www.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/top/volume-1-signal-and-data-processing/support-functions/common-functions/setnumthreads.html
  if ( !getenv("OMP_NUM_THREADS") ) {
    if (verbose)
      puts("limiting IPP threads, without environment var OMP_NUM_THREADS\n");
    ippSetNumThreads(1);
  }
  else if (verbose)
    puts("detected environment var OMP_NUM_THREADS - not limiting IPP threads\n");
}


template <class T>
static void trans_ipp32(
  const mat_info &in, NO_ESCAPE const T * RESTRICT pin,
  const mat_info &out, NO_ESCAPE T * RESTRICT pout )
{
  using IppType = Ipp32s;
  static_assert( sizeof(T) == sizeof(IppType), "" );

  IppiSize roi = { int(in.nCols), int(in.nRows) };  // width, then height
  ippiTranspose_32s_C1R(
    reinterpret_cast<const IppType*>( pin ), int(in.rowSize * sizeof(T)),
    reinterpret_cast<IppType*>( pout ), int(out.rowSize * sizeof(T)),
    roi
  );
}

template <class T>
static void trans_ipp16(
  const mat_info &in, NO_ESCAPE const T * RESTRICT pin,
  const mat_info &out, NO_ESCAPE T * RESTRICT pout )
{
  using IppType = Ipp16s;
  static_assert( sizeof(T) == sizeof(IppType), "" );

  IppiSize roi = { int(in.nCols), int(in.nRows) };  // width, then height
  ippiTranspose_16s_C1R(
    reinterpret_cast<const IppType*>( pin ), int(in.rowSize * sizeof(T)),
    reinterpret_cast<IppType*>( pout ), int(out.rowSize * sizeof(T)),
    roi
  );
}

template <class T>
static void trans_ipp8(
  const mat_info &in, NO_ESCAPE const T * RESTRICT pin,
  const mat_info &out, NO_ESCAPE T * RESTRICT pout )
{
  using IppType = Ipp8u;
  static_assert( sizeof(T) == sizeof(IppType), "" );

  IppiSize roi = { int(in.nCols), int(in.nRows) };  // width, then height
  ippiTranspose_8u_C1R(
    reinterpret_cast<const IppType*>( pin ), int(in.rowSize * sizeof(T)),
    reinterpret_cast<IppType*>( pout ), int(out.rowSize * sizeof(T)),
    roi
  );
}

}

#endif
