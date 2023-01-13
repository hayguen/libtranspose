#!/bin/bash

ITERS="4"

if [ -z "$1" ]; then
  T="bench_clang"
else
  T="$1"
fi

# Ubuntu 22.04 provides clang-14
CX=$(which clang++-14)
CV="14"
if [ -z "${CX}" ]; then
  # Ubuntu 20.04 provides clang-12
  CX=$(which clang++-12)
  CV="12"
fi

if [ -z "${CX}" ]; then
  echo "error: could not find clang++"
  exit 1
fi

echo "clang++-${VC}:    ${CX}"
echo ""

mkdir -p ${T}
for N in $(echo "11 22 44 48 88"); do
  echo "clang version: $( ${CX} --version )" >${T}/bench${N}.log
  for K in $(seq 3) ; do
    echo "===="  >>${T}/bench${N}.log
    build_perf_hints/bench${N} ${ITERS} >/dev/shm/bench_tmp.log
    cat /dev/shm/bench_tmp.log >>${T}/bench${N}.log
    cat /dev/shm/bench_tmp.log
  done
done
