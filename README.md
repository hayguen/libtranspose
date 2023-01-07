# libtranspose

## about

[libtranspose](https://github.com/hayguen/libtranspose) should provide an efficient transpose function for matrices or images consisting of basic data types.

supported are x86/64 and aarm64 platforms. later one through [sse2neon](https://github.com/DLTcollab/sse2neon)

for now, it's no readily re-usable library.

## license

see [MIT License](LICENSE)

## dependencies

### mandatory

CMake and a C++11 compiler - ideally g++

### optional - for comparison in the benchmark

on intel x86/64-Bit Ubuntu Linux, you could/can install intel MKL (Math Kernel Library):

```
sudo apt install intel-mkl
```

on debian, the package name is `intel-mkl-full`


Depending on the Distribution and it's version, the version might be outdated.
Installation of latest OneAPI libraries from intel's APT repository is recommended to achieve better performance:
see https://www.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/install-using-package-managers/apt.html

```
sudo apt install intel-oneapi-mkl intel-oneapi-mkl-devel
sudo apt install intel-oneapi-ipp intel-oneapi-ipp-devel
```

be warned: https://www.intel.com/content/www/us/en/developer/articles/guide/installing-free-libraries-and-python-apt-repo.html
is outdated and does not provide the `intel-oneapi-*` packages

installing `intel-hpckit` as described at https://gist.github.com/DaisukeMiyamoto/c2493c36b929002785e43a3588e7d45a
installs far more! that isn't necessary.


## build / test

```
git clone --recursive https://github.com/hayguen/libtranspose.git

cd libtranspose
cmake -S . -B build
ccmake build  # optional to modify cmake options
cmake --build build
./bench.sh
VERBOSE="-v" ITERSA=4 ITERSB=4 ./bench.sh   # with some options through environment variables
```

## wiki documents

  * https://codingspirit.de/dokuwiki/doku.php?id=development:numeric_math#fast_cache-efficient_matrix_transposition

## todos

  * have a cmake library target
  * have a dispatch mechanism to select SIMD-variant/implementation at runtime
