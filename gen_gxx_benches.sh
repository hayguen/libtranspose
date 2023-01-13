#!/bin/bash

ITERS="4"

if [ -z "$1" ]; then
  T="bench_gxx"
else
  T="$1"
fi

# Ubuntu 22.04 provides g++-11
CX=$(which g++)
if [ -z "${CX}" ]; then
  echo "error: could not find g++"
  exit 1
fi

echo "g++: ${CX}"
echo ""

mkdir -p ${T}
for N in $(echo "11 22 44 48 88"); do
  echo "g++ version: $( ${CX} --version )" >${T}/bench${N}.log
  for K in $(seq 3) ; do
    echo "===="  >>${T}/bench${N}.log
    build/bench${N} ${ITERS} >/dev/shm/bench_tmp.log
    cat /dev/shm/bench_tmp.log >>${T}/bench${N}.log
    cat /dev/shm/bench_tmp.log
  done
done
