
#pragma once

#include "transpose_defs.hpp"

namespace transpose_kernels
{

template <class T>
struct Naive4x4Kernel
{
  static constexpr unsigned KERNEL_SZ = 4;
  static constexpr bool HAS_AA = true;  // same operation

  ALWAYS_INLINE static void op_uu(const T * RESTRICT A, T * RESTRICT B, const unsigned rowSizeA, const unsigned rowSizeB) {
    // https://stackoverflow.com/questions/16941098/fast-memory-transpose-with-sse-avx-and-openmp
    const T r0[] = { A[0], A[1], A[2], A[3] }; // memcpy instead?
    A += rowSizeA;
    const T r1[] = { A[0], A[1], A[2], A[3] };
    A += rowSizeA;
    const T r2[] = { A[0], A[1], A[2], A[3] };
    A += rowSizeA;
    const T r3[] = { A[0], A[1], A[2], A[3] };
    B[0] = r0[0];
    B[1] = r1[0];
    B[2] = r2[0];
    B[3] = r3[0];
    B += rowSizeB;
    B[0] = r0[1];
    B[1] = r1[1];
    B[2] = r2[1];
    B[3] = r3[1];
    B += rowSizeB;
    B[0] = r0[2];
    B[1] = r1[2];
    B[2] = r2[2];
    B[3] = r3[2];
    B += rowSizeB;
    B[0] = r0[3];
    B[1] = r1[3];
    B[2] = r2[3];
    B[3] = r3[3];
  }

  ALWAYS_INLINE static void op_aa(const T * RESTRICT A, T * RESTRICT B, const unsigned rowSizeA, const unsigned rowSizeB) {
    op_uu(A, B, rowSizeA, rowSizeB);
  }

};

}
