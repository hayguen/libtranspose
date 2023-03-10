
#pragma once

// see https://en.wikipedia.org/wiki/Transpose

#include "transpose_defs.hpp"

#include "transpose_cache_aware_kernels.hpp"

#include "trans_kernel_AVX_8x8x32bit.hpp"
#include "trans_kernel_AVX_4x4x32bit.hpp"
#include "trans_kernel_SSE_4x4x32bit.hpp"

#include "trans_kernel_SSE41_8x8x8bit_macros.hpp"
#ifdef HAVE_SSE41_8x8x8_KERNEL
// need to use template specialization with C macros
#  include "transpose_cache_aware_kernel_specialization.hpp"
#  undef KERNEL_NAME
#  undef KERNEL_INIT
#  undef KERNEL_OP_UU
#  undef KERNEL_OP_AA
#endif

#include "trans_kernel_AVX_4x4x64bit_macros.hpp"
#ifdef HAVE_AVX_4X4X64_KERNEL
// need to use template specialization with C macros
#  include "transpose_cache_aware_kernel_specialization.hpp"
#  undef KERNEL_NAME
#  undef KERNEL_INIT
#  undef KERNEL_OP_UU
#  undef KERNEL_OP_AA
#endif

#include "trans_kernel_AVX_4x4x128bit_macros.hpp"
#ifdef HAVE_AVX_4X4X128_KERNEL
// need to use template specialization with C macros
#  include "transpose_cache_aware_kernel_specialization.hpp"
#  undef KERNEL_NAME
#  undef KERNEL_INIT
#  undef KERNEL_OP_UU
#  undef KERNEL_OP_AA
#endif

#include "trans_kernel_SSE2_8x8x16bit.hpp"

#include "trans_kernel_naive.hpp"

#include "transpose_cache_aware_non_simd.hpp"
#include "transpose_cache_oblivious.hpp"
#include "transpose_naive.hpp"
