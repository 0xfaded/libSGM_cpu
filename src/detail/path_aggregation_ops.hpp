#pragma once

#include <types.hpp>

namespace sgm_cpu {
namespace detail {

template <class Tune>
class PathAggregationOps {

 public:
  using tune = Tune;

  struct PatchLayout;

  static inline void aggregate_patch_16x16_(
      PatchLayout &input,
      uint8_t *dst,
      int dst_pitch);

  template <bool is_edge_block>
  static inline void aggregate_patch_16x16_(
      PatchLayout &input,
      uint8_t *dst,
      int dst_pitch);

  static inline void aggregate_patch_16x16(
      PatchLayout &input,
      uint8_t *dst,
      int dst_pitch) {

    aggregate_patch_16x16_<false>(input, dst, dst_pitch);
  }

  static inline void aggregate_edge_patch_16x16(
      PatchLayout &input,
      uint8_t *dst,
      int dst_pitch) {

    aggregate_patch_16x16_<true>(input, dst, dst_pitch);
  }

  static inline void aggregate_patch_16x1(
    const typename Tune::simd::reg::x1_t &cost,
    uint8_t *dst);

  struct consts {
  };

  struct PatchLayout {
    typename Tune::simd::reg::w4_t left;
    std::array<typename Tune::simd::reg::w4_t, 2> right;
  };

};

} // detail
} // sgm_cpu

#include <detail/path_aggregation_ops_impl.hpp>
