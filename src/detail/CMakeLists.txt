enable_testing()

add_executable(
  census_ops_test
  census_ops_test.cpp
)
target_link_libraries(
  census_ops_test
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(census_ops_test)
