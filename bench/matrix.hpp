
#pragma once

#include "transpose_defs.hpp"

#include <iostream>
#include <memory>
#include <cassert>


template <class T>
struct matrix
{
  matrix(unsigned rows_, unsigned cols_, unsigned rowSize_)
    : nRows(rows_), nCols(cols_), rowSize(rowSize_)
  {
#if 0
    data = raw_data = new T[rows_ * rowSize_];
#else
    // constexpr unsigned alignment = 256U / 8U;  // AVX/AVX2 with max. 256 bits == 32 bytes
    // allocation of 32 byte aligned memory does fail!
    constexpr unsigned alignment = 512U / 8U;  // == 64 bytes
    unsigned nAdd = (alignment + sizeof(T) - 1U) / sizeof(T);  // ceil() division result
    if ( !nAdd )
      nAdd = 1;
    std::size_t space = (rows_ * rowSize_ + nAdd) * sizeof(T);
    raw_data = new T[rows_ * rowSize_ + nAdd];
    void * raw = raw_data;
    void * res = std::align( 64U, rows_ * rowSize_ * sizeof(T), raw, space );
    if (!res) {
      std::cerr << "Error allocating aligned memory!\n";
      assert(0);
    }
    data = reinterpret_cast<T*>( res );
#endif
  }

  matrix() = delete;
  matrix(const matrix<T> &) = delete;
  matrix(matrix<T> &&) = delete;

  ~matrix() {
    delete []raw_data;
  }

  ALWAYS_INLINE T operator()(unsigned row, unsigned col) const { return data[row*rowSize+col]; }
  ALWAYS_INLINE T& operator()(unsigned row, unsigned col) { return data[row*rowSize+col]; }

  ALWAYS_INLINE T* row(unsigned row) { return &data[row*rowSize]; }
  ALWAYS_INLINE const T* row(unsigned row) const { return &data[row*rowSize]; }

  template <class PRINT>
  void print() const {
    for( unsigned r = 0; r < nRows; ++r ) {
      for( unsigned c = 0; c < nCols; ++c ) {
        std::cout << PRINT(operator()(r,c)) << " ";
      }
      std::cout << "\n";
    }
  }

  void init() {
    for( unsigned r = 0; r < nRows; ++r ) {
      for( unsigned c = 0; c < rowSize; ++c ) {
        operator()(r,c) = (nCols * r) + ( c + 1 );
      }
    }
  }

  void fill(T v) {
    T * RESTRICT vec = data;
    for( unsigned k = 0; k < nRows * rowSize; ++k )
      vec[k] = v;
  }

  T sum() const {
    T * RESTRICT vec = data;
    T s = T(0);
    for( unsigned k = 0; k < nRows * rowSize; ++k )
      s += vec[k];
    return s;
  }

  // remember its out of situ so output of transpose is always the same no matter how many iterations
  void verify_transposed() const {
    for( unsigned c = 0; c < nRows; ++c ) {
      for( unsigned r = 0; r < nCols; ++r ) {
        T e = (T)( (nRows * r) + ( c + 1 ) );
        if ( operator()(c,r) != e ) {
          std::cerr << "error at verification of transposed row " << c << " column " << r << ": expected = " << e << " actual = " << operator()(c,r) << "\n";
          assert( 0 );
          return;
        }
      }
    }
  }

  T * RESTRICT data;
  T * RESTRICT raw_data;
  const unsigned nRows;
  const unsigned nCols;
  const unsigned rowSize; // row size >= nCols

};
