include(GoogleTest)
add_executable(vecsearch_tests
  test_smoke.cpp
  test_flat.cpp
  test_parallel.cpp
  test_distance.cpp
  test_hnsw.cpp
  test_corrupt.cpp
)
target_link_libraries(vecsearch_tests PRIVATE vecsearch GTest::gtest_main)
target_compile_definitions(vecsearch_tests PRIVATE VECSEARCH_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
gtest_discover_tests(vecsearch_tests DISCOVERY_TIMEOUT 60)
