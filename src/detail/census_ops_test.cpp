#include <random>
#include <iostream>

#include <tune/array128_tune.hpp>
#include <detail/census_ops.hpp>

#include <gtest/gtest.h>

/*
namespace sgm_cpu {
namespace test {
*/
using namespace sgm_cpu;

static uint32_t compute_census(const uint8_t *src, int src_pitch);
static std::ostream &print_descriptor(std::ostream &os, uint32_t desc);

TEST(CensusOpsTest, ExecutePatchX2) {
  std::minstd_rand0 rng;

  constexpr int src_pitch = 16;
  constexpr int dst_pitch = 8;

  std::array<uint8_t, src_pitch*8> patch;
  std::array<uint32_t, dst_pitch*2> output;

  std::generate(patch.begin(), patch.end(), [&rng]() {
      return rng();
  });
  std::fill(output.begin(), output.end(), 0);

  /*
  for (int y = 0; y < 8; y += 1) {
    for (int x = 0; x < 16; x += 1) {
      patch[y*src_pitch+x] = y*src_pitch+x;
    }
  }
  */

  using Ops = detail::CensusOps<tune::Array128>;

  Ops::PatchLayout r;

  for (size_t i = 0; i < r.row2.size(); i += 1) {
    const uint8_t *src = patch.data()+2*src_pitch*i;
    Ops::tune::simd::load_row2_init(r.row2[i], src, src_pitch);
  }

  Ops::execute_patch_x2(r, output.data(), 8);

  for (int y = 0; y < 2; y += 1) {
    for (int x = 0; x < 8; x += 1) {
      uint32_t ref = compute_census(patch.data() + y*src_pitch + x, src_pitch);
      uint32_t desc = output[y*dst_pitch+x];

      print_descriptor(std::cout << "ref: ", ref) << "\n";
      print_descriptor(std::cout << "out: ", desc) << "\n\n";

      ASSERT_EQ(ref, desc);
    }
  }
}

uint32_t compute_census(const uint8_t *src, int src_pitch) {

  struct Comparison {
    int x0, y0;
    int x1, y1;
    int bit;
  };

  static std::vector<Comparison> comparisons = {
    {0, 0, 8, 6, 0},
    {0, 2, 8, 4, 1},
    {0, 4, 8, 2, 2},
    {0, 6, 8, 0, 3},
    {0, 1, 8, 5, 4},
    {0, 3, 8, 3, 5},
    {0, 5, 8, 1, 6},
    {4, 0, 4, 6, 7},

    {1, 0, 7, 6, 8},
    {1, 2, 7, 4, 9},
    {1, 4, 7, 2, 10},
    {1, 6, 7, 0, 11},
    {1, 1, 7, 5, 12},
    {1, 3, 7, 3, 13},
    {1, 5, 7, 1, 14},
    {4, 2, 4, 4, 15},

    {2, 0, 6, 6, 16},
    {2, 2, 6, 4, 17},
    {2, 4, 6, 2, 18},
    {2, 6, 6, 0, 19},
    {2, 1, 6, 5, 20},
    {2, 3, 6, 3, 21},
    {2, 5, 6, 1, 22},
    {4, 1, 4, 5, 23},

    {3, 0, 5, 6, 24},
    {3, 2, 5, 4, 25},
    {3, 4, 5, 2, 26},
    {3, 6, 5, 0, 27},
    {3, 1, 5, 5, 28},
    {3, 3, 5, 3, 29},
    {3, 5, 5, 1, 30} };

  uint32_t result = 0;

  for (Comparison c : comparisons) {
    auto [x0, y0, x1, y1, bit] = c;
    bool b = src[y0 * src_pitch + x0] < src[y1 * src_pitch + x1];
    result |= (b ? 1 : 0) << bit;
  }

  return result;
}

static std::ostream &print_descriptor(std::ostream &os, uint32_t desc) {
  for (int i = 31; i >= 0; i -= 1) {
    os << ((desc & (1 << i)) ? 1 : 0);
    if ((i % 8) == 0) os << " ";
  }
  return os;
}

//} // namespace test
//} // namespace sgm_cpu

