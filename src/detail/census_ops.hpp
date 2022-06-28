#pragma once

#include <census_transform.hpp>

namespace sgm_cpu {
namespace detail {

template <class Tune>
class CensusOps {

 public:
  using tune = Tune;
  using input_type = typename CensusTransform<Tune>::input_type;

  struct PatchLayout;

  static void execute_block(
      const input_type *src,
      feature_type *dst,
      int width,
      int height,
      int src_pitch,
      int dst_pitch);

  static void execute_block_x2(
      const input_type *src,
      feature_type *dst,
      int width,
      int height,
      int src_pitch,
      int dst_pitch);

  static inline void execute_patch_x2(
      PatchLayout &r,
      feature_type *dst,
      int dst_pitch);

  struct consts {
    static constexpr int feature_width = 9;

    static constexpr int v_patch = Tune::census::v_step + 6;
    static constexpr int h_patch = 16;
  };

  struct PatchLayout {
    std::array<typename Tune::simd::reg::x2_t, consts::v_patch / 2> row2;
  };

  static_assert((Tune::census::v_step % 2) == 0,
      "Tune::census::v_step must be even");

  static_assert(Tune::census::v_block >= consts::v_patch,
      "Tune::census::v_block must be greater than patch height (v_patch)");

  static_assert(Tune::census::h_block >= consts::h_patch,
      "Tune::census::h_block must be greater than patch width (h_patch)");

};

} // detail
} // sgm_cpu

#include <detail/census_ops_impl.hpp>
