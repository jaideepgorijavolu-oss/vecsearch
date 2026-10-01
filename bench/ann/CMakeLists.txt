add_executable(hnsw_eval hnsw_eval.cpp)
target_link_libraries(hnsw_eval PRIVATE vecsearch)

add_executable(parallel_build_quality parallel_build_quality.cpp)
target_link_libraries(parallel_build_quality PRIVATE vecsearch)

add_executable(profile_search profile_search.cpp)
target_link_libraries(profile_search PRIVATE vecsearch)
# Frame pointers make perf call graphs reliable without DWARF unwinding.
target_compile_options(profile_search PRIVATE -fno-omit-frame-pointer)
