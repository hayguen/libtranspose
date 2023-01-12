#!/bin/bash

# Ubuntu 22.04 provides clang-14
CX=$(which clang++-14)
CC=$(which clang-14)
CV="14"
if [ -z "${CX}" ]; then
  # Ubuntu 20.04 provides clang-12
  CX=$(which clang++-12)
  CC=$(which clang-12)
  CV="12"
fi

if [ -z "${CX}" ]; then
  echo "error: could not find clang++"
  exit 1
fi

# use opt-viewer.py, if we can find it in env OV  or in PATH,
#   e.g. optview2 from https://github.com/OfekShilon/optview2
if [ -z "${OV}" ]; then
  OV=$(which opt-viewer.py)
fi
if [ -z "${OV}" ]; then
  OV="/usr/lib/llvm-${CV}/share/opt-viewer/opt-viewer.py"
fi

echo "clang++:    ${CX}"
echo "clang:      ${CC}"
echo "opt-viewer: ${OV}"
echo ""

# cmake -DCMAKE_TOOLCHAIN_FILE=~/tests/cmake/CLANG${CV}-Toolchain.cmake -DCMAKE_BUILD_TYPE=Release -S . -B perf_hint_build

if [ -z "$1" ] || [ "$1" = "c" ]; then
  cmake -DCMAKE_CXX_COMPILER=${CX} -DCMAKE_C_COMPILER=${CC} -DCMAKE_BUILD_TYPE=Release -S . -B build_perf_hints
  cmake --build build_perf_hints
fi

if [ -z "$1" ] || [ "$1" = "h" ] || [ "$1" = "html" ]; then
  if [ -z "$2" ]; then
    T="perf_hints"
  else
    T="$2"
  fi

  for N in $(echo "11 22 44 48 88"); do
    mkdir -p "${T}/bench${N}"
    echo "generating html for bench${N} in  ${T}/bench${N}/ .."
    $OV --output-dir "${T}/bench${N}" --source-dir . build_perf_hints/CMakeFiles/bench${N}.dir/bench/
  done
fi
