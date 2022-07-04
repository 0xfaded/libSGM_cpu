enable_testing()

add_executable(
  census_ops_test
  census_ops_test.cpp
)
target_link_libraries(
  census_ops_test
  gtest_main
)

add_executable(
  path_aggregation_ops_test
  test_util.cpp
  path_aggregation_ops_test.cpp
)
target_link_libraries(
  path_aggregation_ops_test
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(census_ops_test path_aggregation_ops_test)
