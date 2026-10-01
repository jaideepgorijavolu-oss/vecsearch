include(GoogleTest)
add_executable(vecsearch_tests
  test_smoke.cpp
)
target_link_libraries(vecsearch_tests PRIVATE vecsearch GTest::gtest_main)
gtest_discover_tests(vecsearch_tests DISCOVERY_TIMEOUT 60)
