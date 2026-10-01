cmake_minimum_required(VERSION 3.24)
project(vecsearch VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(VECSEARCH_BUILD_TESTS "Build GoogleTest unit tests" ON)
option(VECSEARCH_BUILD_BENCH "Build Google Benchmark microbenchmarks and tools" OFF)
option(VECSEARCH_BUILD_PYTHON "Build the pybind11 extension module" OFF)
option(VECSEARCH_NATIVE "Compile everything with -march=native (benchmarks only)" OFF)

# ---- Build types -----------------------------------------------------------
# Release / Debug / RelWithDebInfo are built in. We add:
#   Sanitizer : ASan + UBSan
#   TSan      : ThreadSanitizer (cannot be combined with ASan)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(_san_flags "-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=undefined")
set(CMAKE_CXX_FLAGS_SANITIZER "${_san_flags}" CACHE STRING "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_SANITIZER "-fsanitize=address,undefined" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_SANITIZER "-fsanitize=address,undefined" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_TSAN "-O1 -g -fno-omit-frame-pointer -fsanitize=thread" CACHE STRING "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_TSAN "-fsanitize=thread" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_TSAN "-fsanitize=thread" CACHE STRING "" FORCE)

add_compile_options(-Wall -Wextra -Wpedantic)
if(VECSEARCH_NATIVE)
  add_compile_options(-march=native)
endif()

# ---- Core library ----------------------------------------------------------
find_package(Threads REQUIRED)

add_library(vecsearch STATIC
  src/version.cpp
  src/distance/scalar.cpp
  src/distance/dispatch.cpp
  src/index/flat_index.cpp
)
target_include_directories(vecsearch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vecsearch PUBLIC Threads::Threads)

# ---- Dependencies (fetched) ------------------------------------------------
include(FetchContent)
if(VECSEARCH_BUILD_TESTS)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()

if(VECSEARCH_BUILD_BENCH)
  FetchContent_Declare(benchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(benchmark)
  add_subdirectory(bench/micro)
endif()

if(VECSEARCH_BUILD_PYTHON)
  add_subdirectory(python)
endif()
