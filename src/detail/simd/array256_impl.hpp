#pragma once

#include <algorithm>
#include <array>

namespace sgm_cpu {
namespace detail {
namespace simd {

struct reg {
  struct x1_t {
    std::array<uint8_t, 32> reg0;
  };

  struct x2_t {
    // row (16-byte) storage is reg0 = <row0 | row2>, reg1 = <row1 | row3>
    std::array<uint8_t, 32> reg0;
    std::array<uint8_t, 32> reg1;
  };

  static const int census_rows_per_reg = 2;
};

void load_row_init(reg::x2_t &r, const uint8_t *src, ptrdiff_t pitch) {
  const uint8_t *src0 = src + 0*pitch;
  const uint8_t *src1 = src + 1*pitch;
  const uint8_t *src2 = src + 2*pitch;
  const uint8_t *src3 = src + 3*pitch;

  std::copy(src0, src0 + 16, r.reg0.begin());
  std::copy(src1, src1 + 16, r.reg1.begin());
  std::copy(src2, src2 + 16, r.reg0.begin() + 16);
  std::copy(src3, src3 + 16, r.reg1.begin() + 16);
}

void load_row_next(reg::x2_t &r, const uint8_t *src, ptrdiff_t pitch) {
  const uint8_t *src0 = src + 0*pitch;
  const uint8_t *src1 = src + 1*pitch;
  const uint8_t *src2 = src + 2*pitch;
  const uint8_t *src3 = src + 3*pitch;

  std::copy(src0 - 8, src0 + 8, r.reg0.begin());
  std::copy(src1 - 8, src1 + 8, r.reg1.begin());
  std::copy(src2 - 8, src2 + 8, r.reg0.begin() + 16);
  std::copy(src3 - 8, src3 + 8, r.reg1.begin() + 16);
}

void store_feature(reg::x1_t r, uint32_t *dst, ptrdiff_t pitch) {
  const uint8_t *dst0 = static_cast<uint8_t *>(dst + 0*pitch);
  const uint8_t *dst1 = static_cast<uint8_t *>(dst + 1*pitch);

  std::copy(r.reg0.begin(), r.begin() + 16, dst0);
  std::copy(r.reg0.begin() + 16, r.end(), dst1);
}

// ops
reg::x1_t cmp_row(reg::x2_t left, reg::x2_t right) {
  x1_t result;
  size_t i = 0;
  for (; i < 8; i += 1) {
    result.reg0.at(i   ) = left.reg0.at(i   ) < right.reg0.at(i+ 8) ? 0 : 0xff;
    result.reg0.at(i+16) = left.reg0.at(i+16) < right.reg0.at(i+24) ? 0 : 0xff;
  }

  for (; i < result.length(); i += 1) {
    result.reg0.at(i   ) = left.reg1.at(i   ) < right.reg1.at(i+ 8) ? 0 : 0xff;
    result.reg0.at(i+16) = left.reg1.at(i+16) < right.reg1.at(i+24) ? 0 : 0xff;
  }

  return result;
}

template <int b>
r128x1_t bsel(r128x1_t acc, r128x1_t input) {
  const uint8_t mask = 1 << b;
  const uint8_t imask = ~mask;

  r128x1_t result;
  for (size_t i = 0; i < result.length(); i += 1) {
    result.reg0.at(i) = (imask & acc.reg0.at(i)) | (mask & input.reg0.at(i));
  }

  return result;
}
