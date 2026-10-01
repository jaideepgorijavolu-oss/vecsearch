add_library(autovec_plain OBJECT autovec.cpp)
target_compile_definitions(autovec_plain PRIVATE VECSEARCH_AUTOVEC_NS=autovec)
target_compile_options(autovec_plain PRIVATE -O3 -march=native)

add_library(autovec_fastmath OBJECT autovec.cpp)
target_compile_definitions(autovec_fastmath PRIVATE VECSEARCH_AUTOVEC_NS=autovec_fastmath)
target_compile_options(autovec_fastmath PRIVATE -O3 -march=native -ffast-math)

add_executable(bench_distance bench_distance.cpp)
target_link_libraries(bench_distance PRIVATE vecsearch benchmark::benchmark
  autovec_plain autovec_fastmath)
