
#include <transpose_tpl.hpp>

#include "transpose_ipp.hpp"
#include "transpose_mkl.hpp"

#include "matrix.hpp"

#include <sys/time.h>
#include <cstdint>
#include <cassert>
#include <cstdint>
#include <cstring>

#include <iostream>
#include <iomanip>  // setprecision
#include <algorithm>
#include <cmath>

#include <cpu_features_macros.h>

#if defined(CPU_FEATURES_ARCH_AARCH64)
#  include <cpuinfo_aarch64.h>
#  include "sse2neon/sse2neon.h"
#elif defined(CPU_FEATURES_ARCH_ARM)
#  include <cpuinfo_arm.h>
#  include "sse2neon/sse2neon.h"
#elif defined(CPU_FEATURES_ARCH_X86_64)
#  include <cpuinfo_x86.h>
#  include <x86intrin.h>
#endif


///////////////////////////////////////////

// int8_t / uint16_t / uint32_t / uint64_t
// #define/#if for SAME_DTYPE_SIZES / DTYPEX_SZ to avoid c++17 which brings if constexpr()

#ifndef BENCH_VARIANT
#define BENCH_VARIANT 44
#endif

#if BENCH_VARIANT == 11
  // 8 bit -> 8 bit ; 1 byte -> 1 byte
  using DTYPEX = int8_t;
  using DTYPEY = int8_t;
  using DTYPEP = int64_t;
  #define SAME_DTYPE_SIZES  1
  #define DTYPEX_SZ 1
#elif BENCH_VARIANT == 22
  // 16 bit -> 16 bit ; 2 byte -> 2 byte
  using DTYPEX = int16_t;
  using DTYPEY = int16_t;
  using DTYPEP = int32_t;
  #define SAME_DTYPE_SIZES  1
  #define DTYPEX_SZ 2
#elif BENCH_VARIANT == 44
  // 32 bit -> 32 bit ; 4 byte -> 4 byte
  using DTYPEX = int32_t;
  using DTYPEY = int32_t;
  using DTYPEP = int32_t;
  #define SAME_DTYPE_SIZES  1
  #define DTYPEX_SZ 4
#elif BENCH_VARIANT == 48
  // 32 bit -> 64 bit ; 4 byte -> 8 byte
  using DTYPEX = int32_t;
  using DTYPEY = int64_t;
  using DTYPEP = int64_t;
  #define SAME_DTYPE_SIZES  0
  #define DTYPEX_SZ 4
#elif BENCH_VARIANT == 88
  // 64 bit -> 64 bit ; 8 byte -> 8 byte
  using DTYPEX = int64_t;
  using DTYPEY = int64_t;
  using DTYPEP = int64_t;
  #define SAME_DTYPE_SIZES  1
  #define DTYPEX_SZ 8
#endif


///////////////////////////////////////////

#if defined(CPU_FEATURES_ARCH_AARCH64)
static const auto cpufx = cpu_features::GetAarch64Info().features;
#elif defined(CPU_FEATURES_ARCH_ARM)
static const auto cpufx = cpu_features::GetArmInfo().features();
#elif defined(CPU_FEATURES_ARCH_X86_64)
static const auto cpufx = cpu_features::GetX86Info().features;
#endif

static inline bool have_SSE() {
  #if defined(CPU_FEATURES_ARCH_X86_64)
    return cpufx.sse;
  #elif defined(CPU_FEATURES_ARCH_AARCH64)
    return cpufx.asimd;  // through sse2neon
  #elif defined(CPU_FEATURES_ARCH_ARM)
    return cpufx.neon && cpufx.vfpv3;  // through sse2neon
  #endif
  return false;
}

static inline bool have_SSE2() {
  #if defined(CPU_FEATURES_ARCH_X86_64)
    return have_SSE() && cpufx.sse2;
  #endif
  return false;
}

static inline bool have_SSSE3() {
  #if defined(CPU_FEATURES_ARCH_X86_64)
    return have_SSE2() && cpufx.sse3 && cpufx.ssse3;
  #endif
  return false;
}

static inline bool have_SSE41() {
  #if defined(CPU_FEATURES_ARCH_X86_64)
    return have_SSSE3() && cpufx.sse4_1;
  #endif
  return false;
}

static inline bool have_SSE42() {
  #if defined(CPU_FEATURES_ARCH_X86_64)
    return have_SSE41() && cpufx.sse4_2;
  #endif
  return false;
}

static inline bool have_SSE4() {
  return have_SSE42();
}

static inline bool have_AVX() {
  #if defined(CPU_FEATURES_ARCH_X86_64)
    return have_SSE4() && cpufx.avx;
  #endif
  return false;
}

HEDLEY_DIAGNOSTIC_PUSH
HEDLEY_DIAGNOSTIC_DISABLE_UNUSED_FUNCTION

static inline bool have_AVX2() {
  #if defined(CPU_FEATURES_ARCH_X86_64)
    return have_AVX() && cpufx.avx2;
  #endif
  return false;
}

HEDLEY_DIAGNOSTIC_POP

// have_SSE() have_SSE2() have_SSSE3() have_SSE4() have_AVX() have_AVX2()


///////////////////////////////////////////

// type for the transpose function pointers
//   function is HEDLEY_NO_THROW - but gcc warns, that function attribute is ignored (here)
typedef void (*transpose_function)(
  const transpose::mat_info &, NO_ESCAPE const DTYPEX * RESTRICT,
  const transpose::mat_info &, NO_ESCAPE DTYPEY * RESTRICT );

// identity function - could be used for type conversion or other transforms
template <class X, class Y>
struct FuncId
{
  ALWAYS_INLINE HEDLEY_CONST
  Y operator()(X in) const {
    return in;
  }
};

template <class T, class U, class FUNC>
HEDLEY_NO_THROW
static void trans_fake(
  const transpose::mat_info &, NO_ESCAPE const T * RESTRICT pin,
  const transpose::mat_info &out, NO_ESCAPE U * RESTRICT pout )
{
  // iterate linearly through output matrix indices
  const unsigned N = out.nRows;
  const unsigned M = out.nCols;
  const unsigned out_RS = out.rowSize;
  unsigned out_off;
  FUNC f;

  for( unsigned c = out_off = 0; c < N; ++c, out_off += out_RS ) {
    for( unsigned r = 0; r < M; ++r ) {
      pout[out_off+r] = f( (T)( (N * r) + ( c + 1 ) ) );
    }
  }

  // suppress warnings
  (void)pin;
}


struct test_s
{
  const char * n;
  transpose_function f;
  double cycles;
  uint32_t duration;
  int order;
};

static test_s tests[32];
static int n_tests = 0;

//////////////////////////////////////////////////////

class StopWatch
{
public:

  void start() {
    gettimeofday( &tvs, NULL );
    cycles_start = _rdtsc();
  }

  void stop() {
    cycles_end = _rdtsc();
    gettimeofday( &tve, NULL );
    tstart = tvs.tv_sec * 1000 + tvs.tv_usec / 1000;
    tend   = tve.tv_sec * 1000 + tve.tv_usec / 1000;
    tduration = tend - tstart;
    cycles_measured = (cycles_end - cycles_start);
  }

  uint32_t millis() const { return tduration; }
  double cycles() const { return cycles_measured; }

private:
  double cycles_measured;
  struct timeval tvs, tve;
  uint64_t cycles_start, cycles_end;
  uint32_t tstart, tend, tduration;
};

static StopWatch sw_bench;

//////////////////////////////////////////////////////


// time/bench given transpose function
//   microbenchmark repeats iter times, to get good ms values
static void time_and_test( int test_idx, int iters,
  const matrix<DTYPEX> &in,
  matrix<DTYPEY> &out,
  int verbose = 0,
  bool verify = true
) {
  const transpose::mat_info in_info  {  in.nRows,  in.nCols,  in.rowSize };
  const transpose::mat_info out_info { out.nRows, out.nCols, out.rowSize };

  std::cout << std::setw(2) << test_idx + 1 << " " << tests[test_idx].n << ": ";
  for( int t = 0; t < 2; ++t ) {
    sw_bench.start();
    for( int i = 0; i < iters; i++ )
      tests[test_idx].f( in_info, in.data, out_info, out.data );
    sw_bench.stop();
    tests[test_idx].duration = sw_bench.millis();
    tests[test_idx].cycles = sw_bench.cycles();
    std::cout << std::setw(4) << tests[test_idx].duration << " ms ";
    std::cout << "(= " << std::scientific << std::setprecision(2) << tests[test_idx].cycles << " cycles)\t";
  }
  std::cout << "\n";

  if (verbose >= 2)
    out.print<DTYPEP>();
  if (verify)
    out.verify_transposed();
}

static void enqueue(const char *n, transpose_function f) {
  assert( n_tests < 32 );
  tests[n_tests].n = n;
  tests[n_tests].f = f;
  tests[n_tests].cycles = -1.0;
  tests[n_tests].duration = 0;
  tests[n_tests].order = -1;
  ++n_tests;
}

static void enqueue_versatile_tests(
  const transpose::mat_info &in, NO_ESCAPE const DTYPEX * RESTRICT pin,
  const transpose::mat_info &out, NO_ESCAPE DTYPEY * RESTRICT pout )
{
  enqueue( "fake                      ", trans_fake<DTYPEX, DTYPEY, FuncId<DTYPEX, DTYPEY> > );

  enqueue( "naive_in                  ", transpose::naive_in<DTYPEX,   DTYPEY,  FuncId<DTYPEX, DTYPEY> > );
  enqueue( "naive_out                 ", transpose::naive_out<DTYPEX,  DTYPEY,  FuncId<DTYPEX, DTYPEY> > );
  enqueue( "naive_meta                ", transpose::naive_meta<DTYPEX,  DTYPEY, FuncId<DTYPEX, DTYPEY> > );

  enqueue( "cache_oblivious_in        ", transpose::cache_oblivious_in  <DTYPEX, DTYPEY, FuncId<DTYPEX, DTYPEY> > );
  enqueue( "cache_oblivious_out       ", transpose::cache_oblivious_out <DTYPEX, DTYPEY, FuncId<DTYPEX, DTYPEY> > );
  enqueue( "cache_oblivious_meta      ", transpose::cache_oblivious_meta<DTYPEX, DTYPEY, FuncId<DTYPEX, DTYPEY> > );

  enqueue( "cache_aware_in            ", transpose::caware_in< DTYPEX,  DTYPEY, FuncId<DTYPEX, DTYPEY> > );
  enqueue( "cache_aware_out           ", transpose::caware_out<DTYPEX,  DTYPEY, FuncId<DTYPEX, DTYPEY> > );
  enqueue( "cache_aware_meta          ", transpose::caware_meta<DTYPEX, DTYPEY, FuncId<DTYPEX, DTYPEY> > );

#if SAME_DTYPE_SIZES
  using TRANSPOSE_CLASS = transpose::caware_kernel<DTYPEX, transpose_kernels::Naive4x4Kernel<DTYPEX> >;
  enqueue( "kernel_in  <naive>_uu     ", TRANSPOSE_CLASS::uu_in );
  enqueue( "kernel_out <naive>_uu     ", TRANSPOSE_CLASS::uu_out );
  enqueue( "kernel_meta<naive>_uu     ", TRANSPOSE_CLASS::uu_meta );
  if ( TRANSPOSE_CLASS::aa_possible(in, pin, out, pout) ) {
    enqueue( "kernel_in  <naive>_aa     ", TRANSPOSE_CLASS::aa_in );
    enqueue( "kernel_out <naive>_aa     ", TRANSPOSE_CLASS::aa_out );
    enqueue( "kernel_meta<naive>_aa     ", TRANSPOSE_CLASS::aa_meta );
  }
#else
  // suppress warnings
  (void)in;
  (void)out;
  (void)pin;
  (void)pout;
#endif
}


static void enqueue_8bit_tests(
  const transpose::mat_info &in, NO_ESCAPE const DTYPEX * RESTRICT pin,
  const transpose::mat_info &out, NO_ESCAPE DTYPEY * RESTRICT pout )
{
#if SAME_DTYPE_SIZES && DTYPEX_SZ == 1

#  ifdef HAVE_SSE41_8x8x8_KERNEL
     if ( have_SSE41() ) {
       using TRANSPOSE_CLASS = transpose::caware_kernel<DTYPEX, transpose_kernels::SSE41_8x8x8Kernel<DTYPEX> >;
       enqueue( "kernel_in  <SSE41_8x8>_uu ", TRANSPOSE_CLASS::uu_in );
       enqueue( "kernel_out <SSE41_8x8>_uu ", TRANSPOSE_CLASS::uu_out );
       if ( TRANSPOSE_CLASS::aa_possible(in, pin, out, pout) ) {
         enqueue( "kernel_in  <SSE41_8x8>_aa ", TRANSPOSE_CLASS::aa_in );
         enqueue( "kernel_out <SSE41_8x8>_aa ", TRANSPOSE_CLASS::aa_out );
       }
     }
#  endif

#  if defined(HAVE_ONEAPI_IPP)
#    if defined(HAVE_IPP_KERNEL)
       enqueue( "one/ippiTranspose         ", transpose::trans_ipp8 );
#    endif
#  endif

#endif
  // suppress warnings
  (void)in;
  (void)out;
  (void)pin;
  (void)pout;
}


static void enqueue_16bit_tests(
  const transpose::mat_info &in, NO_ESCAPE const DTYPEX * RESTRICT pin,
  const transpose::mat_info &out, NO_ESCAPE DTYPEY * RESTRICT pout )
{
#if SAME_DTYPE_SIZES && DTYPEX_SZ == 2

#  ifdef HAVE_SSE2_8x8x16_KERNEL
     if ( have_SSE2() ) {
       using TRANSPOSE_CLASS = transpose::caware_kernel<DTYPEX, transpose_kernels::SSE2_8x8x16Kernel<DTYPEX> >;
       enqueue( "kernel_in  <SSE2_8x8>_uu  ", TRANSPOSE_CLASS::uu_in );
       enqueue( "kernel_out <SSE2_8x8>_uu  ", TRANSPOSE_CLASS::uu_out );
       if ( TRANSPOSE_CLASS::aa_possible(in, pin, out, pout) ) {
         enqueue( "kernel_in  <SSE2_8x8>_aa  ", TRANSPOSE_CLASS::aa_in );
         enqueue( "kernel_out <SSE2_8x8>_aa  ", TRANSPOSE_CLASS::aa_out );
       }
     }
#  endif

#  if defined(HAVE_ONEAPI_IPP)
#    if defined(HAVE_IPP_KERNEL)
       enqueue( "one/ippiTranspose         ", transpose::trans_ipp16 );
#    endif
#  endif

#endif
  // suppress warnings
  (void)in;
  (void)out;
  (void)pin;
  (void)pout;
}


static void enqueue_32bit_tests(
  const transpose::mat_info &in, NO_ESCAPE const DTYPEX * RESTRICT pin,
  const transpose::mat_info &out, NO_ESCAPE DTYPEY * RESTRICT pout )
{
#if SAME_DTYPE_SIZES && DTYPEX_SZ == 4

#  ifdef HAVE_SSE_4X4X32_KERNEL
     if ( have_SSE() ) {
       using SSE_TRANSPOSE_CLASS = transpose::caware_kernel<DTYPEX, transpose_kernels::SSE_4x4x32Kernel<DTYPEX> >;
       enqueue( "kernel_in  <SSE_4x4>_uu   ", SSE_TRANSPOSE_CLASS::uu_in );
       enqueue( "kernel_out <SSE_4x4>_uu   ", SSE_TRANSPOSE_CLASS::uu_out );
       if ( SSE_TRANSPOSE_CLASS::aa_possible(in, pin, out, pout) ) {
         enqueue( "kernel_in  <SSE_4x4>_aa   ", SSE_TRANSPOSE_CLASS::aa_in );
         enqueue( "kernel_out <SSE_4x4>_aa   ", SSE_TRANSPOSE_CLASS::aa_out );
       }
     }

     // looks, that out order is always faster with the SSE kernel
     // time_and_test( "kernel_meta<SSE>  ", transpose::caware_kernel<DTYPEX, SSE_4x4x32Kernel<DTYPEX> >::uu_meta, iters, in, out );
#  endif

#  ifdef HAVE_AVX_4X4X32_KERNEL
     if ( have_AVX() ) {
       using AVX44_TRANSPOSE_CLASS = transpose::caware_kernel<DTYPEX, transpose_kernels::AVX_4x4x32Kernel<DTYPEX> >;
       enqueue( "kernel_in  <AVX_4x4>_uu   ", AVX44_TRANSPOSE_CLASS::uu_in );
       enqueue( "kernel_out <AVX_4x4>_uu   ", AVX44_TRANSPOSE_CLASS::uu_out );
       if ( AVX44_TRANSPOSE_CLASS::aa_possible(in, pin, out, pout) ) {
         enqueue( "kernel_in  <AVX_4x4>_aa   ", AVX44_TRANSPOSE_CLASS::aa_in );
         enqueue( "kernel_out <AVX_4x4>_aa   ", AVX44_TRANSPOSE_CLASS::aa_out );
       }
     }
#  endif

#  ifdef HAVE_AVX_8X8X32_KERNEL
     if ( have_AVX() ) {
       using AVX88_TRANSPOSE_CLASS = transpose::caware_kernel<DTYPEX, transpose_kernels::AVX_8x8x32Kernel<DTYPEX> >;
       enqueue( "kernel_in  <AVX_8x8>_uu   ", AVX88_TRANSPOSE_CLASS::uu_in );
       enqueue( "kernel_out <AVX_8x8>_uu   ", AVX88_TRANSPOSE_CLASS::uu_out );
       if ( AVX88_TRANSPOSE_CLASS::aa_possible(in, pin, out, pout) ) {
         enqueue( "kernel_in  <AVX_8x8>_aa   ", AVX88_TRANSPOSE_CLASS::aa_in );
         enqueue( "kernel_out <AVX_8x8>_aa   ", AVX88_TRANSPOSE_CLASS::aa_out );
       }
     }

     if ( have_AVX() ) {
       using AVX88I_TRANSPOSE_CLASS = transpose::caware_kernel<DTYPEX, transpose_kernels::AVX_8x8x32VINS_Kernel<DTYPEX> >;
       enqueue( "kernel_in  <AVX_8x8I>_uu  ", AVX88I_TRANSPOSE_CLASS::uu_in );
       enqueue( "kernel_out <AVX_8x8I>_uu  ", AVX88I_TRANSPOSE_CLASS::uu_out );
       if ( AVX88I_TRANSPOSE_CLASS::aa_possible(in, pin, out, pout) ) {
         enqueue( "kernel_in  <AVX_8x8I>_aa  ", AVX88I_TRANSPOSE_CLASS::aa_in );
         enqueue( "kernel_out <AVX_8x8I>_aa  ", AVX88I_TRANSPOSE_CLASS::aa_out );
       }
     }
#  endif

#  ifdef HAVE_MKL_KERNEL
#    if defined(HAVE_SYSTEM_MKL)
       enqueue( "sys/mkl_somatcopy         ", transpose::trans_mkl32 );
#    elif defined(HAVE_ONEAPI_MKL)
       enqueue( "one/mkl_somatcopy         ", transpose::trans_mkl32 );
#    else
       enqueue( "mkl_somatcopy             ", transpose::trans_mkl32 );
#    endif
#  endif

#  if defined(HAVE_ONEAPI_IPP)
#    if defined(HAVE_IPP_KERNEL)
       enqueue( "one/ippiTranspose         ", transpose::trans_ipp32 );
#    endif
#  endif

#endif
  // suppress warnings
  (void)in;
  (void)out;
  (void)pin;
  (void)pout;
}


static void enqueue_64bit_tests(
  const transpose::mat_info &in, NO_ESCAPE const DTYPEX * RESTRICT pin,
  const transpose::mat_info &out, NO_ESCAPE DTYPEY * RESTRICT pout )
{
#if SAME_DTYPE_SIZES && DTYPEX_SZ == 8

#  ifdef HAVE_AVX_4X4X64_KERNEL
     if ( have_AVX() ) {
       using TRANSPOSE_CLASS = transpose::caware_kernel<DTYPEX, transpose_kernels::AVX_4x4x64Kernel<DTYPEX> >;
       enqueue( "kernel_in  <AVX_4x4>_uu   ", TRANSPOSE_CLASS::uu_in );
       enqueue( "kernel_out <AVX_4x4>_uu   ", TRANSPOSE_CLASS::uu_out );
       if ( TRANSPOSE_CLASS::aa_possible(in, pin, out, pout) ) {
         enqueue( "kernel_in  <AVX_4x4>_aa   ", TRANSPOSE_CLASS::aa_in );
         enqueue( "kernel_out <AVX_4x4>_aa   ", TRANSPOSE_CLASS::aa_out );
       }
     }
#  endif

#  ifdef HAVE_MKL_KERNEL
#    if defined(HAVE_SYSTEM_MKL)
       enqueue( "sys/mkl_domatcopy         ", transpose::trans_mkl64 );
#    elif defined(HAVE_ONEAPI_MKL)
       enqueue( "one/mkl_domatcopy         ", transpose::trans_mkl64 );
#    else
       enqueue( "mkl_domatcopy             ", transpose::trans_mkl64 );
#    endif
#  endif

#endif
  // suppress warnings
  (void)in;
  (void)out;
  (void)pin;
  (void)pout;
}


int main( int argc, char* argv[] ) {
  if ( 1 < argc && ( !strcmp(argv[1], "-h") || !strcmp(argv[1], "/h") || !strcmp(argv[1], "--help") || !strcmp(argv[1], "/help") ) ) {
    std::cout << "usage: " << argv[0] << " [-v] [-v] [<iters> [<plot_msamples> [<nRows> [<ncols> [<y_min> [ <y_max> [<input rowSize> [<output rowSize>] ] ] ] ] ] ] ]\n";
    std::cout << "  -v | -vv          verbose output\n";
    std::cout << "  <iters>           number of iterations for transpose benchmark; default: 100\n";
    std::cout << "  <plot_msamples>   MSamples of plot data in plot benchmark (only in 44); default: 16\n";
    std::cout << "  <nRows>           number of rows    for input matrix = plot width; default: 300\n";
    std::cout << "  <nCols>           number of columns for input matrix = plot height; default: 1024\n";
    std::cout << "  <plot_y_min>      minimum visible value on y-axis for data in [-1024 .. 1023]; default: -1200\n";
    std::cout << "  <plot_y_max>      maximum visible value on y-axis for data in [-1024 .. 1023]; default:  1200\n";
    std::cout << "  <input  rowSize>  row size (space for number of columns) for input  matrix; default: <nRows>\n";
    std::cout << "  <output rowSize>  row size (space for number of columns) for output matrix; default: <nCols>\n";
    return 1;
  }
  int verbose = 0;
  verbose             = verbose + ( (verbose+1 < argc) ? (strcmp(argv[verbose+1], "-v") ? 0 : 1) : 0 );
  verbose             = verbose + ( (verbose+1 < argc) ? (strcmp(argv[verbose+1], "-v") ? 0 : 1) : 0 );
  int iters           = (verbose+1 < argc) ? atoi(argv[verbose+1]) : 100;
  int msamples        = (verbose+2 < argc) ? atoi(argv[verbose+2]) : 16;
  unsigned N          = (verbose+3 < argc) ? atoi(argv[verbose+3]) : 1024U;
  unsigned M          = (verbose+4 < argc) ? atoi(argv[verbose+4]) : 300U;
  float    plot_y_min = (verbose+5 < argc) ? atof(argv[verbose+5]) : -1200.0F;
  float    plot_y_max = (verbose+6 < argc) ? atof(argv[verbose+6]) :  1200.0F;
  unsigned RSM        = (verbose+7 < argc) ? atoi(argv[verbose+7]) : M;
  unsigned RSN        = (verbose+8 < argc) ? atoi(argv[verbose+8]) : N;

  if (iters < 0)      iters = 0;
  if (msamples < 0)   msamples = 1;
  if (N <= 0)         N = 300U;
  if (M <= 0)         M = 1024U;
  if (RSM < M)        RSM = M;
  if (RSN < N)        RSN = N;

  matrix<DTYPEX> in{  N, M, RSM };
  matrix<DTYPEY> out{ M, N, RSN };

  const unsigned LT = transpose::numElemsInCacheLine<DTYPEX>();
  const unsigned LU = transpose::numElemsInCacheLine<DTYPEY>();
  unsigned RSMA = (( M + LT -1 ) / LT) * LT;
  unsigned RSNA = (( N + LU -1 ) / LU) * LU;

  std::cout << "running bench" << int(BENCH_VARIANT) << ": transposing matrix of depth "
    << 8*int(sizeof(DTYPEX)) << " bit  to matrix of depth "
    << 8*int(sizeof(DTYPEY)) << " bit\n";

  if (verbose >= 1) {
    std::cout << "input  matrix shape: " << N << " rows x " << M << " cols  with row size " << RSM << "\n";
    std::cout << "  aligned row size would be " << RSMA << "\n";
    std::cout << "output matrix shape: " << M << " rows x " << N << " cols  with row size " << RSN << "\n";
    std::cout << "  aligned row size would be " << RSNA << "\n\n";
    std::cout << "iters:    " << iters << " for testing of each transpose algorithm\n";
    std::cout << "  LT (in):  " << LT << " input  elements fit in one cache line\n";
    std::cout << "  LU (out): " << LU << " output elements fit in one cache line\n\n";
    std::cout << "msamples: " << msamples << " for comparison of plotting\n";
    std::cout << "  plot_y_min: " << plot_y_min << "\n";
    std::cout << "  plot_y_max: " << plot_y_max << "\n\n";
    // https://www.appsloveworld.com/cplus/100/124/detect-the-availability-of-sse-sse2-instruction-set-in-visual-studio
    // https://stackoverflow.com/questions/28939652/how-to-detect-sse-sse2-avx-avx2-avx-512-avx-128-fma-kcvi-availability-at-compile
    // https://en.wikipedia.org/wiki/AVX-512
    // https://en.wikichip.org/wiki/x86/avx-512
#if defined(__AVX512F__)
    const char maxSIMD[] = "AVX512F (=AVX512 Base)";
#elif defined(__AVX2__)
    const char maxSIMD[] = "AVX2";
#elif defined ( __AVX__ )
    const char maxSIMD[] = "AVX";
#elif (defined(__x86_64__) || defined(_M_AMD64) || defined(_M_X64))
    const char maxSIMD[] = "SSE2 x64";
#elif _M_IX86_FP == 2
    const char maxSIMD[] = "SSE2 x32";
#elif _M_IX86_FP == 1
    const char maxSIMD[] = "SSE x32";
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    const char maxSIMD[] = "NEON (ARM)";
#else
    const char maxSIMD[] = "no x86 SIMD available";
#endif
    std::cout << "maximum supported SIMD at compile time: " << maxSIMD << "\n\n";
  }

  in.init();
  if (verbose >= 2) {
    std::cout << "input:\n";
    in.print<DTYPEP>();
  }

  transpose::mat_info in_info  {  in.nRows,  in.nCols,  in.rowSize };
  transpose::mat_info out_info { out.nRows, out.nCols, out.rowSize };
  enqueue_versatile_tests(in_info, in.data, out_info, out.data);
  enqueue_8bit_tests(in_info,  in.data, out_info, out.data);
  enqueue_16bit_tests(in_info, in.data, out_info, out.data);
  enqueue_32bit_tests(in_info, in.data, out_info, out.data);
  enqueue_64bit_tests(in_info, in.data, out_info, out.data);


#if 0
  n_tests = 0;
  enqueue( "kernel_in <SSE_8>  ", transpose::caware_kernel<DTYPEX,  transpose_kernels::SSE41_8x8x8Kernel<DTYPEX> >::uu_in );
  // enqueue( "kernel_out<SSE_8>  ", transpose::caware_kernel<DTYPEX, transpose_kernels::SSE8x8x64Kernel<DTYPEX> >::uu_out );
#endif

  // run all tests
  if (iters > 0) {
    for (int k = 0; k < n_tests; ++k) {
      time_and_test( k, iters, in, out, verbose );
    }

    for (int n = 0; n < 3; ++n) {
      int min_idx = -1;
      for (int k = 1; k < n_tests; ++k) {
        if ( tests[k].order < 0 && ( min_idx < 0 || tests[k].cycles < tests[min_idx].cycles ) )
          min_idx = k;
      }
      if ( min_idx >= 0 ) {
        tests[min_idx].order = n;
        std::cout << "order " << n + 1 << ": "
          << std::setw(2) << min_idx + 1 << " " << tests[min_idx].n << ": "
          << std::setw(4) << tests[min_idx].duration << " ms "
          << std::scientific << std::setprecision(2) << tests[min_idx].cycles << " cycles\n";
      }
    }
  }

#if SAME_DTYPE_SIZES && DTYPEX_SZ == 4
#  ifdef HAVE_SSE_4X4X32_KERNEL
  if ( have_SSE() && msamples > 0 ) {

    /// plot a big vector into an 32-bit RGB(A) image
    const unsigned plotDataLen = msamples * 1024U * 1024U;
    float * plotData = new float[plotDataLen];
    // const unsigned x_per_pixel = plotDataLen / ( N - 1U );  // assume plotDataLen >> N
    const float Mflt = M;

    const double ix1 = 0.0;
    const double ix2 = plotDataLen - 1;
    const double iy1 = 0.0;
    const double iy2 = N - 0.01; // N - 1;
    const double i_m = (iy2 - iy1) / (ix2 - ix1);
    const double inv_m = (ix2 - ix1) / (iy2 - iy1);
    // const double i_c = -ix1 * i_m + iy1; == 0.0

    // y(x1 = plot_y_max) = 0    = y1
    // y(x2 = plot_y_min) = M-1  = y2
    const float x1 = plot_y_max;
    const float x2 = plot_y_min;
    const float y1 = 0;
    const float y2 = M - 1;
    const float y_m = (y2 - y1) / (x2 - x1);
    const float y_c = -x1 * y_m + y1;

    std::cout << "\nprepare plot data of " << msamples << " MSamples = "
              << std::scientific << std::setprecision(2) << double(plotDataLen) << " samples\n";

    {
      // generate plot data
      // LCG of Numerical Recipes, see https://en.wikipedia.org/wiki/Linear_congruential_generator
      constexpr int32_t a = 1664525;
      constexpr int32_t c = 1013904223;
      //constexpr int32_t m = 1 << 32;  // = 2^32
      int32_t seed = 1;
      for ( unsigned k = 0; k < plotDataLen; ++k ) {
        seed = ( a * seed + c );
        plotData[k] = float( seed >> (32 - 11) );  // keep 11 bits => +/- 1024
      }

      auto mima = std::minmax_element( plotData, plotData + plotDataLen );
      float h_min = *(mima.first)  * y_m + y_c;
      float h_max = *(mima.second) * y_m + y_c;
      std::cout << std::fixed << std::setprecision(2)
                << "generated:    min = " << *(mima.first) << "  max = " << *(mima.second) << "\n"
                << "  transform to rows " << h_min << " .. " << h_max << "\n";
    }

    /////////////

    out.fill(0);
    std::cout << "direct plot A into " << M << " rows x " << N << " cols output:       " << std::flush;
    sw_bench.start();
    {
      for ( unsigned k = 0; k < plotDataLen; ++k ) {
        float h = plotData[k] * y_m + y_c;
        if ( h < 0.0F || h >= Mflt )
          continue;
        unsigned hu = unsigned(h);
        // unsigned xu = k / x_per_pixel;
        unsigned xu = unsigned(std::floor(k * i_m));
        //   [ 0 .. plotDataLen - 1 ] / ( plotDataLen / ( N - 1) )
        // = [ 0 .. 0.99 ] / ( 1.0 / ( N - 1 ) )
        // = [ 0 .. 0.99 ] * ( N - 1 )
        out(hu, xu) += 1;
      }
      // transpose() not necessary
    }
    sw_bench.stop();
    std::cout << std::setw(4) << sw_bench.millis() << " ms "
      << "(= " << std::scientific << std::setprecision(2) << sw_bench.cycles() << " cycles)\n";
    DTYPEX sum_out = out.sum();
    std::cout << double(sum_out) << " points were set in matrix (= "
      << std::fixed << std::setprecision(2)
      << double(sum_out) * 100.0 / double(plotDataLen) << "%)\n" << std::flush;

    /////////////

    out.fill(0);
    std::cout << "direct plot B into " << M << " rows x " << N << " cols output:       " << std::flush;
    sw_bench.start();
    {
      unsigned k = 0;
      while ( k < plotDataLen ) {
        unsigned xu = unsigned(std::floor(k * i_m));
        // xu = k * i_m
        // xu + 1 = kk * i_m
        // k_stop = (xu + 1) / i_m
        // k_stop = (xu + 1) * inv_m
        unsigned k_stop = std::min( unsigned(std::ceil((xu + 1) * inv_m)), plotDataLen );
        for ( ; k < k_stop; ++k ) {
          float h = plotData[k] * y_m + y_c;
          if ( h < 0.0F || h >= Mflt )
            continue;
          unsigned hu = unsigned(h);
          out(hu, xu) += 1;
        }
      }
      // transpose() not necessary
    }
    sw_bench.stop();
    std::cout << std::setw(4) << sw_bench.millis() << " ms "
      << "(= " << std::scientific << std::setprecision(2) << sw_bench.cycles() << " cycles)\n";
    sum_out = out.sum();
    std::cout << double(sum_out) << " points were set in matrix (= "
      << std::fixed << std::setprecision(2)
      << double(sum_out) * 100.0 / double(plotDataLen) << "%)\n" << std::flush;

    /////////////

    in.fill(0);
    std::cout << "plot A into " << N << " rows x " << M << " cols input and transpose: " << std::flush;
    sw_bench.start();
    {
#if 0
      unsigned last_xu = 0xFFFFFFFFU;
      DTYPEX * row_ptr = nullptr;
#endif
      for ( unsigned k = 0; k < plotDataLen; ++k ) {
        float h = plotData[k] * y_m + y_c;
        if ( h < 0.0F || h >= Mflt )
          continue;
        unsigned hu = unsigned(h);
        unsigned xu = unsigned(std::floor(k * i_m));
#if 0
        if ( last_xu != xu ) {
          row_ptr = in.row(xu);
          last_xu = xu;
        }
        row_ptr[hu] += 1;
#else
        in(xu, hu) += 1;
#endif
      }
      // transpose() is necessary
      using SSE_TRANSPOSE_CLASS = transpose::caware_kernel<DTYPEX, transpose_kernels::SSE_4x4x32Kernel<DTYPEX> >;
      // transpose_function ftrans = ( SSE_TRANSPOSE_CLASS::aa_possible(out_info, in_info) )
      //   ? SSE_TRANSPOSE_CLASS::aa_out
      //   : SSE_TRANSPOSE_CLASS::uu_out;
      // ftrans( out_info, in_info );
      SSE_TRANSPOSE_CLASS::uu_out( in_info, in.data, out_info, out.data );
    }
    sw_bench.stop();
    std::cout << std::setw(4) << sw_bench.millis() << " ms "
      << "(= " << std::scientific << std::setprecision(2) << sw_bench.cycles() << " cycles)\n";
    DTYPEX sum_in = in.sum();
    std::cout << double(sum_in) << " points were set in matrix (= "
      << std::fixed << std::setprecision(2)
      << double(sum_in) * 100.0 / double(plotDataLen) << "%)\n" << std::flush;

    /////////////

    in.fill(0);
    std::cout << "plot B into " << N << " rows x " << M << " cols input and transpose: " << std::flush;
    sw_bench.start();
    {
      unsigned k = 0;
      while ( k < plotDataLen ) {
        unsigned xu = unsigned(std::floor(k * i_m));
        unsigned row_off = xu * in.rowSize;
        unsigned k_stop = std::min( unsigned(std::ceil((xu + 1) * inv_m)), plotDataLen );
        for ( ; k < k_stop; ++k ) {
          float h = plotData[k] * y_m + y_c;
          if ( h < 0.0F || h >= Mflt )
            continue;
          unsigned hu = unsigned(h);
          // in(xu, hu) += 1;
          in.data[ row_off + hu ] += 1;
        }
      }
      using SSE_TRANSPOSE_CLASS = transpose::caware_kernel<DTYPEX, transpose_kernels::SSE_4x4x32Kernel<DTYPEX> >;
      SSE_TRANSPOSE_CLASS::uu_out( in_info, in.data, out_info, out.data );
    }
    sw_bench.stop();
    std::cout << std::setw(4) << sw_bench.millis() << " ms "
      << "(= " << std::scientific << std::setprecision(2) << sw_bench.cycles() << " cycles)\n";
    sum_in = in.sum();
    std::cout << double(sum_in) << " points were set in matrix (= "
      << std::fixed << std::setprecision(2)
      << double(sum_in) * 100.0 / double(plotDataLen) << "%)\n" << std::flush;

  }
#endif
#endif

  return 0;
}
