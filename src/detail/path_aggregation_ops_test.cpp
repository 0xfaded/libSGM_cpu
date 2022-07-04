#include <random>
#include <iostream>

#include <tune/array128_tune.hpp>
#include <detail/path_aggregation_ops.hpp>

#include <gtest/gtest.h>

#include "test_util.hpp"

namespace sgm_cpu {
namespace test {

TEST(PathAggregationOps, AggregatePatch16x16) {
  std::minstd_rand0 rng;

  using Ops = detail::PathAggregationOps<tune::Array128>;

  int W = 32;
  int H = 1;
  int B = 0;
  int D = 16;

  std::vector<uint32_t> left = random_descriptors(W, H, B, rng);
  std::vector<uint32_t> right = random_descriptors(W, H, B, rng);

  std::vector<uint8_t> output(16*16);

  Ops::PatchLayout layout;

  Ops::tune::simd::load_w4(layout.left, left.data());
  Ops::tune::simd::load_w4(layout.right[0], right.data());
  Ops::tune::simd::load_w4(layout.right[1], right.data() + 16);

  Ops::aggregate_patch_16x16(layout, output.data(), D);

  for (int d = 0; d < 16; d += 1) {
    for (int l = 0; l < 16; l += 1) {
      int i = D*d + l;
      ASSERT_EQ(output[i], __builtin_popcount(left[l] ^ right[l+D-d])) << "i = " << i << "\n";
    }
  }
}


TEST(PathAggregationOps, AggregateEdgePatch16x16) {
  std::minstd_rand0 rng;

  using Ops = detail::PathAggregationOps<tune::Array128>;

  int W = 16;
  int H = 1;
  int B = 0;
  int D = 16;

  std::vector<uint32_t> left = random_descriptors(W, H, B, rng);
  std::vector<uint32_t> right = random_descriptors(W, H, B, rng);

  std::vector<uint8_t> output(16*16);

  Ops::PatchLayout layout;

  Ops::tune::simd::load_w4(layout.left, left.data());
  Ops::tune::simd::clear(layout.right[0]);
  Ops::tune::simd::load_w4(layout.right[1], right.data());

  Ops::aggregate_edge_patch_16x16(layout, output.data(), D);

  for (int d = 0; d < 16; d += 1) {
    for (int l = 0; l < 16; l += 1) {
      int i = D*d + l;
      int r = l-d;
      if (r >= 0) {
        ASSERT_EQ(output[i], __builtin_popcount(left[l] ^ right[r])) << "i = " << i << "\n";
      } else {
        ASSERT_EQ(output[i], 0) << "i = " << i << "\n";
      }
    }
  }
}

} // namespace test
} // namespace sgm_cpu

