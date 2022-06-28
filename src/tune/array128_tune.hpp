#pragma once

#include <array>

#include <detail/simd/array128_impl.hpp>

namespace sgm_cpu {
namespace tune {

struct Array128 {
  using simd = detail::simd::array128_impl;

  struct census {
    static constexpr int h_block = 64;
    static constexpr int v_block = 32;

    static constexpr int h_step = 8;
    static constexpr int v_step = 2;
  };
};

} // namespace tune
} // namespace sgm_cpu
