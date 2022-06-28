#pragma once

#include <algorithm>
#include <array>

namespace sgm_cpu {
namespace detail {
namespace simd {

struct reg {
  struct x1_t {
    std::array<uint8_t, 8> reg0;
  };

  struct x2_t {
    std::array<uint8_t, 8> reg0;
    std::array<uint8_t, 8> reg1;
  };

  constexpr int census_rows_per_x2 = 1;
};

void load_row_init(reg::x2_t &r, const uint8_t *src, ptrdiff_t pitch) {
  [[maybe_usused]]  pitch;

  std::copy(src + 0, src +  8, r.reg0.begin());
  std::copy(src + 8, src + 16, r.reg1.begin());
}

void load_row_next(reg::x2_t &r, const uint8_t *src, ptrdiff_t pitch) {
  [[maybe_usused]]  pitch;

  std::copy(src - 8, src + 0, r.reg0.begin());
  std::copy(src + 0, src + 8, r.reg1.begin());
}

void store_feature(reg::x1_t r, uint32_t *dst, ptrdiff_t pitch) {
  [[maybe_usused]]  pitch;

  const uint8_t *dst0 = static_cast<const uint8_t *>(dst);

  std::copy(r.reg0.begin(), r.end(), dst0);
}

// ops
reg::x1_t cmp_row(reg::x2_t left, reg::x2_t right) {
  x1_t result;
  for (size_t i = 0; i < 8; i += 1) {
    result.reg0.at(i) = left.reg0.at(i) < right.reg1.at(i) ? 0 : 0xff;
  }
  return result;
}

template <int b>
reg::x1_t bsel(reg::x1_t acc, reg::x1_t bit) {
  const uint8_t mask = 1 << b;
  const uint8_t imask = ~mask;

  r128x1_t result;
  for (size_t i = 0; i < result.length(); i += 1) {
    result.reg0.at(i) = (imask & acc.reg0.at(i)) | (mask & bit.reg0.at(i));
  }
  return result;
}


