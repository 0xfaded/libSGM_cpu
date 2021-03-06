#pragma once

#include <algorithm>
#include <array>
#include <bitset>
#include <cstdint>
#include <cstddef>

namespace sgm_cpu {
namespace detail {
namespace simd {

struct array128_impl {
  struct reg {
    struct x1_t {
      std::array<uint8_t, 16> reg0;
    };

    struct s1_t {
      std::array<uint16_t, 8> reg0;
    };

    struct w4_t {
      std::array<std::array<uint32_t, 4>, 4> reg;
    };

    struct p1_t {
      std::bitset<16> reg0;
    };

    struct x2_t {
      // one row (16-byte) per register
      std::array<uint8_t, 16> reg0;
      std::array<uint8_t, 16> reg1;
    };

    static constexpr int census_rows_per_x2 = 2;
  };

  inline static
  void clear(reg::x1_t &r) {
    std::fill(r.reg0.begin(), r.reg0.end(), 0);
  }

  inline static
  void clear(reg::s1_t &r) {
    std::fill(r.reg0.begin(), r.reg0.end(), 0);
  }

  inline static
  void clear(reg::w4_t &r) {
    std::for_each(r.reg.begin(), r.reg.end(), [](auto &reg_i) {
      std::fill(reg_i.begin(), reg_i.end(), 0);
    });
  }

  inline static
  void load_row2(reg::x2_t &r, const uint8_t *src, ptrdiff_t pitch) {
    const uint8_t *src0 = src + 0*pitch;
    const uint8_t *src1 = src + 1*pitch;

    std::copy(src0, src0 + 8, r.reg0.begin());
    std::copy(src1, src1 + 8, r.reg0.begin() + 8);

    std::copy(src0 + 8, src0 + 16, r.reg1.begin());
    std::copy(src1 + 8, src1 + 16, r.reg1.begin() + 8);
  }

  inline static
  void store_feature(reg::x1_t r, uint32_t *dst, ptrdiff_t pitch) {
    uint8_t *dst0 = reinterpret_cast<uint8_t *>(dst);

    std::copy(r.reg0.begin(), r.reg0.end(), dst0);
  }

  inline static
  void store_x1(const reg::x1_t &r, uint8_t *dst) {
    std::copy(r.reg0.begin(), r.reg0.end(), dst);
  }

  inline static
  void store_w4(const reg::w4_t &r, uint32_t *dst) {
    for_each(r.reg.begin(), r.reg.end(), [&dst](const auto &reg_i) {
      dst = std::copy(reg_i.begin(), reg_i.end(), dst);
    });
  }

  inline static
  void load_s1(reg::s1_t &r, const uint16_t *src) {
    std::copy(src, src + r.reg0.size(), r.reg0.begin());
  }

  inline static
  void load_w4(reg::w4_t &r, const uint32_t *src) {
    for_each(r.reg.begin(), r.reg.end(), [&src](auto &reg_i) {
      std::copy(src, src + reg_i.size(), reg_i.begin());
      src += reg_i.size();
    });
  }

  inline static
  reg::p1_t cmp_row2(reg::x2_t a, reg::x2_t b) {
    reg::p1_t result;
    for (size_t i = 0; i < result.reg0.size(); i += 1) {
      result.reg0.set(i, a.reg0.at(i) < b.reg1.at(i));
    }
    return result;
  }

  template <int b> static
  reg::x1_t bsel2(reg::x1_t acc, reg::p1_t pred) {
    const uint8_t bit = 1 << b;
    const uint8_t mask = static_cast<uint8_t>(~bit);

    reg::x1_t result;
    for (size_t i = 0; i < result.reg0.size(); i += 1) {
      result.reg0.at(i) = (mask & acc.reg0.at(i)) |
        (pred.reg0.test(i) ? bit : 0);
    }

    return result;
  }

  inline static
  void transpose_row2(reg::x2_t &r0, reg::x2_t &r1) {
    for (size_t i = 0; i < 8; i += 1) {
      std::swap(r0.reg0.at(8+i), r1.reg0.at(i));
      std::swap(r0.reg1.at(8+i), r1.reg1.at(i));
    }
  }

  inline static
  void zip4b2(reg::x1_t &r0, reg::x1_t &r1, reg::x1_t &r2, reg::x1_t &r3) {
    reg::x1_t *r[4] = { &r0, &r1, &r2, &r3 };

    // transpose inplace 4x4 byte blocks
    for (size_t block = 0; block < 16; block += 4) {
      for (int i = 0; i < 4; i += 1) {
        for (int j = i+1; j < 4; j += 1) {
          std::swap(r[i]->reg0.at(block + j), r[j]->reg0.at(block + i));
        }
      }
    }

    // transpose inplace 4x4 words
    for (int i = 0; i < 4; i += 1) {
      for (int j = i+1; j < 4; j += 1) {
        for (int k = 0; k < 4; k += 1) {
          std::swap(r[i]->reg0.at(4 * j + k), r[j]->reg0.at(4 * i + k));
        }
      }
    }
  }

  inline static
  reg::x2_t roll_outward2(reg::x2_t r, int x) {
    reg::x2_t result;

    std::copy_backward(r.reg0.begin() + 1, r.reg0.end(), result.reg0.end()-1);
    std::copy_backward(r.reg1.rbegin() + 1, r.reg1.rend(), result.reg1.rend()-1);

    result.reg0.at(7) = r.reg1.at(0+2*x);
    result.reg0.at(15) = r.reg1.at(8+2*x);

    result.reg1.at(0) = r.reg0.at(7-2*x);
    result.reg1.at(8) = r.reg0.at(15-2*x);

    return result;
  }

  inline static
  reg::x1_t fill_x1(uint8_t x) {
    reg::x1_t result;
    std::fill(result.reg0.begin(), result.reg0.end(), x);
    return result;
  }

  inline static
  reg::x1_t and_x1(const reg::x1_t &a, const reg::x1_t &b) {
    reg::x1_t result = a;
    for (size_t i = 0; i < a.reg0.size(); i += 1) {
      result.reg0[i] = a.reg0[i] & b.reg0[i];
    }
    return result;
  }

  inline static
  reg::x1_t shiftr_x1(const reg::x1_t &r, int n) {
    reg::x1_t result;
    std::fill(result.reg0.begin(), result.reg0.begin() + n, 0);
    std::copy(r.reg0.begin(), r.reg0.end() - n, result.reg0.begin() + n);
    return result;
  }

  template<int offset> static
  reg::x1_t popcnt_xor_w4(
      const reg::w4_t &left,
      const reg::w4_t &right0,
      const reg::w4_t &right1) {

    static_assert(0 <= offset && offset < 4,
        "popcnt_xor_w2 offset must be one of 0,1,2,3");

    reg::x1_t result;
    auto out_it = result.reg0.begin();

    for (size_t l = 0; l < left.reg.size(); l += 1) {
      auto &right = l >= offset ? right1.reg[l-offset] : right0.reg[l+4-offset];
      for (size_t i = 0; i < left.reg[l].size(); i += 1) {
        *out_it = __builtin_popcount(left.reg[l][i] ^ right[i]);
        ++out_it;
      }
    }

    return result;
  }

  inline static
  void shift_up_w4(
      reg::w4_t &right0,
      reg::w4_t &right1) {

    uint32_t overflow;
    auto shift_up = [&overflow](auto &reg_i) {
      uint32_t tmp = reg_i.back();
      std::copy(reg_i.begin(), reg_i.end() - 1, reg_i.begin() + 1);
      reg_i.front() = overflow;
      overflow = tmp;
    };

    overflow = 0;
    std::for_each(right0.reg.begin(), right0.reg.end(), shift_up);
    std::for_each(right1.reg.begin(), right1.reg.end(), shift_up);
  }

};

} // namespace simd
} // namespace detail
} // namespace sgm_cpu
