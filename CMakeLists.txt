cmake_minimum_required(VERSION 3.13)
project(libtranspose)

option(DISABLE_LINK_WITH_M "Disables linking with m library to build with clangCL from MSVC" OFF)
option(USE_BENCH_SYS_MKL    "use Intel MKL from system/distribution" OFF)

if ( (CMAKE_SYSTEM_PROCESSOR STREQUAL "i686") OR (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64") OR (CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64") )
    option(USE_BENCH_ONEAPI_MKL "use Intel MKL from OneAPI (extra package source)" ON)
    option(USE_BENCH_ONEAPI_IPP "use Intel IPP from OneAPI (extra package source)" ON)
else()
    option(USE_BENCH_ONEAPI_MKL "use Intel MKL from OneAPI (extra package source)" OFF)
    option(USE_BENCH_ONEAPI_IPP "use Intel IPP from OneAPI (extra package source)" OFF)
endif()


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
include(cmake/compiler_warnings.cmake)


# cmake -S . -B build -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DBUILD_GMOCK=OFF -DGTEST_HAS_ABSL=OFF -DINSTALL_GTEST=OFF
# https://stackoverflow.com/questions/3766740/overriding-a-default-option-value-in-cmake-from-a-parent-cmakelists-txt
SET(BUILD_SHARED_LIBS OFF CACHE BOOL "Build cpu_features library as shared.")
SET(BUILD_TESTING     OFF CACHE BOOL "Build cpu_features tests")
SET(BUILD_GMOCK       OFF CACHE BOOL "Build cpu_features gmock")
SET(GTEST_HAS_ABSL    OFF CACHE BOOL "Build cpu_features: GTEST_HAS_ABSL")
SET(INSTALL_GTEST     OFF CACHE BOOL "Build cpu_features: INSTALL_GTEST")
add_subdirectory( cpu_features EXCLUDE_FROM_ALL )
# now, we have the library cpu_features


# C90 requires the gcc extensions for function attributes like always_inline
# C99 provides the function attributes: no gcc extensions required
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 98)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "Build type not specified: defaulting to release.")
endif(NOT CMAKE_BUILD_TYPE)

if ( CMAKE_C_COMPILER_ID MATCHES "MSVC" )
    # using Visual Studio C++
    message(STATUS "INFO: detected MSVC: will not link math lib m")
    set(MATHLIB "")
    add_definitions("/D_CRT_SECURE_NO_WARNINGS")
    set(MSVC_DISABLED_WARNINGS_LIST "C4996")
else()
    if(DISABLE_LINK_WITH_M)
    else()
        message(STATUS "INFO: detected NO MSVC: ${CMAKE_C_COMPILER_ID}: will link math lib m")
        set(MATHLIB "m")
    endif()
endif()

set(STDCXXLIB "")
if (MINGW)
    set(STDCXXLIB "stdc++")
endif()

######################################################

set(BENCH_VARIANTS "11;22;44;48;88")

# template headers
set( LIB_TPL_SOURCES
    src/transpose_defs.hpp
    src/transpose_tpl.hpp
    # transpose variants / algorithms
    src/transpose_naive.hpp
    src/transpose_cache_oblivious.hpp
    src/transpose_cache_aware_non_simd.hpp
    src/transpose_cache_aware_kernels.hpp
    src/transpose_cache_aware_kernel_specialization.hpp
    # kernels of 4x4 and 8x8 blocks, insertable as template class in
    # transpose_cache_aware_kernels.hpp or transpose_cache_aware_kernel_specialization.hpp
    src/trans_kernel_naive.hpp
    src/trans_kernel_SSE_4x4x32bit.hpp
    src/trans_kernel_SSE2_8x8x16bit.hpp
    src/trans_kernel_SSE41_8x8x8bit.hpp
    src/trans_kernel_AVX_4x4x32bit.hpp
    src/trans_kernel_AVX_4x4x64bit.hpp
    src/trans_kernel_AVX_8x8x32bit.hpp
)

set( BENCH_SOURCES
    bench/transpose_mkl.hpp
    bench/transpose_ipp.hpp
    bench/matrix.hpp
    bench/bench.cpp
)

message(STATUS "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "detected Clang compiler; activating -fsave-optimization-record")
endif()

foreach (VARIANT ${BENCH_VARIANTS})
    add_executable( bench${VARIANT} ${BENCH_SOURCES} ${LIB_TPL_SOURCES} )
    target_activate_cxx_compiler_warnings(bench${VARIANT})
    set_property(TARGET bench${VARIANT} PROPERTY CXX_STANDARD 11)
    set_property(TARGET bench${VARIANT} PROPERTY CXX_STANDARD_REQUIRED ON)
    target_compile_definitions(bench${VARIANT} PRIVATE BENCH_VARIANT=${VARIANT})
    # target_compile_definitions(bench${VARIANT} PRIVATE _USE_MATH_DEFINES)
    target_include_directories(bench${VARIANT} PRIVATE src)
    target_include_directories(bench${VARIANT} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} )
    target_include_directories(bench${VARIANT} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpu_features/include )
    target_include_directories(bench${VARIANT} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/hedley )
    target_link_libraries(bench${VARIANT} PRIVATE cpu_features  ${MATHLIB})
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(bench${VARIANT}  PRIVATE -fsave-optimization-record)
    endif()
endforeach()

if ( (CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR (CMAKE_C_COMPILER_ID STREQUAL "Clang") )
    if ( (CMAKE_SYSTEM_PROCESSOR STREQUAL "i686") OR (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64") OR (CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64") )
        foreach (VARIANT ${BENCH_VARIANTS})
            # target_compile_options(bench${VARIANT} PRIVATE "-march=ivybridge")
            # target_compile_options(bench${VARIANT} PRIVATE "-march=core2")
            target_compile_options(bench${VARIANT} PRIVATE "-march=native")
        endforeach()
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        foreach (VARIANT ${BENCH_VARIANTS})
            # target_compile_options(bench${VARIANT} PRIVATE "-march=armv8-a")
            target_compile_options(bench${VARIANT} PRIVATE "-march=native")
        endforeach()
    else()
        message(WARNING "unknown/unsupported processor '${CMAKE_SYSTEM_PROCESSOR}'")
    endif()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    if ( (CMAKE_SYSTEM_PROCESSOR STREQUAL "i686") OR (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64") OR (CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64") )
        # [SSE2]/AVX/AVX2/AVX512
        foreach (VARIANT ${BENCH_VARIANTS})
            target_compile_options(bench${VARIANT} PRIVATE "/arch:AVX")
        endforeach()
    else()
        message(WARNING "unknown/unsupported processor '${CMAKE_SYSTEM_PROCESSOR}'")
    endif()
else()
    message(WARNING "unsupported C/C++ compiler '${CMAKE_CXX_COMPILER_ID}', see https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER_ID.html")
endif()


if (USE_BENCH_SYS_MKL)
    if ( (CMAKE_SYSTEM_PROCESSOR STREQUAL "i686") OR (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64") OR (CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64") )
        # has chances to work
    else()
        # other PROCESSORs could be "ppc", "ppc64",  "arm", "aarch64", "armv7l" - or something else?!
        message(WARNING "using Intel MKL on '${CMAKE_SYSTEM_PROCESSOR}' might fail.")
    endif()
    message(STATUS "In case compiling/linking with Intel MKL fails, check CMakeLists.txt or deactivate USE_BENCH_MKL")
    foreach (VARIANT ${BENCH_VARIANTS})
        target_compile_definitions(bench${VARIANT} PRIVATE HAVE_SYSTEM_MKL=1)
        target_link_libraries(bench${VARIANT} PRIVATE mkl_intel_lp64 mkl_sequential -lmkl_core)
    endforeach()
    message(STATUS "Trying to use/link against system's Intel MKL")
endif()

if (USE_BENCH_ONEAPI_MKL)
    # https://www.intel.com/content/www/us/en/develop/documentation/onemkl-macos-developer-guide/top/getting-started/cmake-config-for-onemkl.html
    find_package(MKL CONFIG)
    if (MKL_FOUND)
        foreach (VARIANT ${BENCH_VARIANTS})
            target_compile_definitions(bench${VARIANT} PRIVATE HAVE_ONEAPI_MKL=1)
            target_compile_options(bench${VARIANT} PRIVATE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
            target_include_directories(bench${VARIANT} PRIVATE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
            target_link_libraries(bench${VARIANT} PRIVATE $<LINK_ONLY:MKL::MKL>)
        endforeach()
        message(STATUS "Found/Using Intel OneAPI MKL")
    else()
        message(WARNING "Could NOT Find Intel OneAPI MKL")
    endif()
endif()

if (USE_BENCH_ONEAPI_IPP)
    # https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-cmake-with-oneapi-applications.html
    find_package(IPP CONFIG)
    if (IPP_FOUND)
        # IPP components: ipp_iw ippcore ippcc ippdc ippch ippcv ippe ippi ipps ippvm
        foreach (VARIANT ${BENCH_VARIANTS})
            target_compile_definitions(bench${VARIANT} PRIVATE HAVE_ONEAPI_IPP=1)
            target_compile_options(bench${VARIANT} PRIVATE $<TARGET_PROPERTY:IPP::ippi,INTERFACE_COMPILE_OPTIONS> $<TARGET_PROPERTY:IPP::ipps,INTERFACE_COMPILE_OPTIONS>)
            target_include_directories(bench${VARIANT} PRIVATE $<TARGET_PROPERTY:IPP::ippi,INTERFACE_INCLUDE_DIRECTORIES> $<TARGET_PROPERTY:IPP::ipps,INTERFACE_INCLUDE_DIRECTORIES>)
            target_link_libraries(bench${VARIANT} PRIVATE $<LINK_ONLY:IPP::ippi> $<LINK_ONLY:IPP::ipps>)
        endforeach()
        message(STATUS "Found/Using Intel OneAPI IPP")
    else()
        message(WARNING "Could NOT Find Intel OneAPI IPP")
    endif()
endif()

######################################################
