namespace sgm_cpu {
namespace detail {


/*
void do_it() {
  for (int by = 0; by < height; by += 1) {
    for (int bx = 0; bx < width; bx += Tune::aggregation::block_x) {
      aggregate_block();
    }
  }
}

void aggregate_block() {

  using simd = typename Tune::simd;
  using w1_t = typename simd::reg::w1_t;

  using aggr = typename Tune::aggregate;

  // TODO[carl]: Abstract pixels_per_reg concept into tune
  constexpr int n_pixels_per_w1 = 4;
  static_assert((aggr::block_x % n_pixels_per_w1) != 0,
        "aggregation::block_x must be divisible by 4");

  static_assert((aggr::block_d % n_pixels_per_w1) != 0,
        "aggregation::block_d must be divisible by 4");

  constexpr int n_left = aggr::block_x / n_pixels_per_w1;
  constexpr int n_right = (aggr::block_x + aggr::block_d) / n_pixels_per_w1;

  std::array<w1_t, n_left> left;
  std::array<w1_t, n_right> right;

  for (size_t i = right.size() - 1; i >= 1; i -= 1) {
    simd::shift_right(right[i], right[i-1]);
  }
  simd::shift_right_zero(right[0]);

}
*/

// TODO[carl]: Abstract the patch size once the
// hard coded implementation is figured out.
template <class Tune>
template <bool is_edge_block>
void PathAggregationOps<Tune>::aggregate_patch_16x16_(
    PathAggregationOps<Tune>::PatchLayout &input,
    uint8_t *dst,
    int dst_pitch) {

  using simd = typename Tune::simd;
  using x1_t = typename simd::reg::x1_t;
  using w4_t = typename simd::reg::w4_t;

  w4_t &left = input.left;
  std::array<w4_t, 2> &right = input.right;

  for (int shift = 0; shift < 4; shift += 1) {
    x1_t cost0 = simd::template popcnt_xor_w4<0>(left, right[0], right[1]);
    x1_t cost1 = simd::template popcnt_xor_w4<1>(left, right[0], right[1]);
    x1_t cost2 = simd::template popcnt_xor_w4<2>(left, right[0], right[1]);
    x1_t cost3 = simd::template popcnt_xor_w4<3>(left, right[0], right[1]);

    if (is_edge_block) {
      x1_t mask = simd::fill_x1(0xff);
      mask = simd::shiftr_x1(mask, shift);
      cost0 = simd::and_x1(cost0, mask);

      mask = simd::shiftr_x1(mask, 4);
      cost1 = simd::and_x1(cost1, mask);

      mask = simd::shiftr_x1(mask, 4);
      cost2 = simd::and_x1(cost2, mask);

      mask = simd::shiftr_x1(mask, 4);
      cost3 = simd::and_x1(cost3, mask);
    }

    aggregate_patch_16x1(cost0, dst + 0*4*dst_pitch);
    aggregate_patch_16x1(cost1, dst + 1*4*dst_pitch);
    aggregate_patch_16x1(cost2, dst + 2*4*dst_pitch);
    aggregate_patch_16x1(cost3, dst + 3*4*dst_pitch);

    dst += dst_pitch;

    simd::shift_up_w4(right[0], right[1]);
  }

  /*
  s1_t best = penalty_2;

  for (int block = 0; block < depth; block += 8) {
    // there are 4 descriptors per register, which must be down shifted
    for (int shift = 0; shift < 4; shift += 1) {
      // in an 8x8 block, each group of 4 descriptors gets compared twice
      for (int sub_block = 0; sub_block < 8; block += 4) {

        const int storage = block | (sub_block / 4) | (shift * 4);
        const int r = sub_block / 4;

        s1_t cost = simd::xor_w2(left[0], left[1], right[r], right[r+1];

        // Vertical
        {
          const uint16_t *dp_base = v_prev.dp + depth * depth_pitch;
          const uint16_t *min_base = v_prev.min;

          s1_t min_prev = simd::load_s(min_base);

          s1_t prev_best = v_best.prev[x];

          s1_t cost_closer = v_cost.prev[x-1][d];
          s1_t cost_equal = v_cost.prev[x][d];
          s1_t cost_farther = v_cost.prev[x][d];

          s1_t cost = aggregate_min_cost(cost_closer, cost_equal, cost_farther,
              prev_best, penalty_1, penalty_2);

          best = simd::min_s(best, cost);
        }
    }
    */


  /*
  s_square_t cost_patch = simd::outer_product(w_left, w_right);

  }
  */
}

/*
template <class Arch>
typename Arch::simd::reg::s1_t AggregateOps<Arch>::aggregate_min_cost(
    const typename simd::reg::s1_t &cost_closer,
    const typename simd::reg::s1_t &cost_equal,
    const typename simd::reg::s1_t &cost_farther,
    const typename simd::reg::s1_t &cost_prev_best,
    const typename simd::reg::s1_t &penalty_1,
    const typename simd::reg::s1_t &penalty_1) {

  // cost = min(equal, closer + p1, farther + p1, prev_best + p2) - prev_best
  typename simd::reg::s1_t cost;

  cost = simd::min_s(cost_closer, cost_farther);
  cost = simd::add_s(cost, penalty_1);
  cost = simd::min_s(cost, cost_p0);
  cost = simd::sub_s(cost, prev_best);
  cost = simd::min_s(cost, penalty_2);

  return cost;
}

*/

template <class Tune>
void PathAggregationOps<Tune>::aggregate_patch_16x1(
    const typename Tune::simd::reg::x1_t &cost,
    uint8_t *dst) {

  using simd = typename Tune::simd;

  simd::store_x1(cost, dst);
}

} // detail
} // sgm_cpu

