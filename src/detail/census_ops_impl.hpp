#pragma once

#include <iostream>
#include <iomanip>

namespace sgm_cpu {
namespace detail {

template <class Tune>
void CensusOps<Tune>::execute_block(
    const input_type *src,
    feature_type *dst,
    int width,
    int height,
    int src_pitch,
    int dst_pitch) {

  using simd = typename Tune::simd;

  dst_pitch = (dst_pitch == -1) ? width : dst_pitch;

  PatchLayout r;

  // Improve cache performance across rows (especially on architectures with
  // limited simd registers) by processing the image in blocks.
  for (int by = 0; by < Tune::census::v_block; by += Tune::census::v_step) {

    // change row0 reference to top of input patches
    const uint8_t *row0 = reinterpret_cast<const uint8_t *>(src);

    // Load the leftmost part of the patch, including 3 pixels above and below.
    for (size_t i = 0; i < r.row2.size(); i += 1) {
      simd::load_row2_init(r.row2[i], row0, src_pitch);
      row0 += 2*src_pitch;
    }

    int bx = 0;
    goto row_loaded;

    for (; bx < Tune::census::h_block; bx += Tune::census::h_step) {

    // load_row2_init has already loaded pixels and loadrow2_next must not
    // be called on first iteration. (implementations may assume h_step valid
    // values before input pointers).
    for (size_t i = 0; i < r.row2.size(); i += 1) {
      simd::load_row2_next(r.row2[i], row0, src_pitch);
      row0 += 2*src_pitch;
    }

  row_loaded:

      execute_patch_x2(r, dst, dst_pitch);
      row0 += bx;
    }
    src += Tune::census::v_step * src_pitch;
  }
}


// Implementation for 128-bit architectures registers where two rows
// (128-bits each) are processed simutaneously.
template <class Tune>
void CensusOps<Tune>::execute_patch_x2(
    CensusOps::PatchLayout &r,
    feature_type *dst,
    int dst_pitch) {

  // It's tempting for people to change the feature size thinking they will
  // get better descriptors. This would lead to memory errors below, so
  // put in an additional assert
  static_assert(consts::feature_width == 9, "Do not change the feature size!");

  using simd = typename Tune::simd;
  using p1_t = typename simd::reg::p1_t;
  using x1_t = typename simd::reg::x1_t;
  using x2_t = typename simd::reg::x2_t;

  constexpr int feature_half_width = consts::feature_width / 2;
  constexpr int n_iterations = Tune::census::v_step / 2;

  std::array<std::array<x1_t, 4>, n_iterations> out2;

  for (int x = 0; x < feature_half_width; x += 1) {

    std::cout << "in :\n";
    for (int y = 0; y < 4; y += 1) {
      for (int i = 0; i < 8; i += 1) {
        std::cout << std::setw(2) << std::hex << static_cast<int>(r.row2[y].reg0[i]) << " ";
      }
      for (int i = 0; i < 8; i += 1) {
        std::cout << " " << std::setw(2) << std::hex << static_cast<int>(r.row2[y].reg1[i]);
      }
      std::cout << "\n";
      for (int i = 0; i < 8; i += 1) {
        std::cout << std::setw(2) << std::hex << static_cast<int>(r.row2[y].reg0[8+i]) << " ";
      }
      for (int i = 0; i < 8; i += 1) {
        std::cout << " " << std::setw(2) << std::hex << static_cast<int>(r.row2[y].reg1[8+i]);
      }
      std::cout << "\n";
    }


    for (int y = 0; y < n_iterations; y += 1) {
      x1_t out;
      p1_t p;

      simd::clear(out);

      // p is divided into to halves, resulting from the comparisons described below
      //
      //   p.left  = row[0].left < row[6].right
      //   p.right = row[1].left < row[7].right
      p = simd::cmp_row2(r.row2[y+0], r.row2[y+3]);
      out = simd::template bsel2<0>(out, p);

      std::cout << "p0 " << p.reg0 << "\n";

      for (int i = 0; i < 16; i += 1) {
        std::cout << std::setw(2) << std::hex << static_cast<int>(out.reg0.at(i)) << " ";
      }
      std::cout << "\n";

      // The same is true for all other comparisons:
      //
      //   p[0] = row[2].left < row[4].right
      //   p[1] = row[3].left < row[5].right
      p = simd::cmp_row2(r.row2[y+1], r.row2[y+2]);
      out = simd::template bsel2<1>(out, p);

      std::cout << "p1 " << p.reg0 << "\n";

      for (int i = 0; i < 16; i += 1) {
        std::cout << std::setw(2) << std::hex << static_cast<int>(out.reg0.at(i)) << " ";
      }
      std::cout << "\n";

      //   p[0] = row[4].left < row[2].right
      //   p[1] = row[5].left < row[3].right
      p = simd::cmp_row2(r.row2[y+2], r.row2[y+1]);
      out = simd::template bsel2<2>(out, p);

      std::cout << "p2 " << p.reg0 << "\n";

      for (int i = 0; i < 16; i += 1) {
        std::cout << std::setw(2) << std::hex << static_cast<int>(out.reg0.at(i)) << " ";
      }
      std::cout << "\n";

      //   p[0] = row[6].left < row[0].right
      //   p[1] = row[7].left < row[1].right
      p = simd::cmp_row2(r.row2[y+3], r.row2[y+0]);
      out = simd::template bsel2<3>(out, p);

      std::cout << "p3 " << p.reg0 << "\n";

      for (int i = 0; i < 16; i += 1) {
        std::cout << std::setw(2) << std::hex << static_cast<int>(out.reg0.at(i)) << " ";
      }
      std::cout << "\n";

      // We've compared the even rows, now we need the odd rows.
      // Registers are renamed for clarity. After transpose:
      //   odd_0[0] = patch_row[1]
      //   odd_0[1] = patch_row[2]
      //   odd_1[0] = patch_row[3]
      //   odd_1[1] = patch_row[2]
      //   odd_2[0] = patch_row[1]
      //   odd_2[1] = patch_row[2]
      //   odd_x[0] = patch_row[0]
      //   odd_x[1] = patch_row[7]
      //
      // Note these are references, not copies. Copies would add an entire
      // patch-worth of register pressure. Instead, we transpose, then undo.
      x2_t &odd_x = r.row2[y];
      x2_t &odd_2 = r.row2[y+1];
      x2_t &odd_0 = r.row2[y+2];
      x2_t &odd_1 = r.row2[y+3];

      simd::transpose_row2(odd_1, odd_0);
      simd::transpose_row2(odd_x, odd_0);
      simd::transpose_row2(odd_2, odd_1);
      simd::transpose_row2(odd_0, odd_2);

    std::cout << "odd:\n";
    for (int y = 0; y < 4; y += 1) {
      for (int i = 0; i < 8; i += 1) {
        std::cout << std::setw(2) << std::hex << static_cast<int>(r.row2[y].reg0[i]) << " ";
      }
      for (int i = 0; i < 8; i += 1) {
        std::cout << " " << std::setw(2) << std::hex << static_cast<int>(r.row2[y].reg1[i]);
      }
      std::cout << "\n";
      for (int i = 0; i < 8; i += 1) {
        std::cout << std::setw(2) << std::hex << static_cast<int>(r.row2[y].reg0[8+i]) << " ";
      }
      for (int i = 0; i < 8; i += 1) {
        std::cout << " " << std::setw(2) << std::hex << static_cast<int>(r.row2[y].reg1[8+i]);
      }
      std::cout << "\n";
    }

      // self comparison for pixels left and right of center on same row:
      //   p[0] = row[3].left < row[3].right
      //   p[1] = row[4].left < row[4].right
      p = simd::cmp_row2(odd_1, odd_1);
      out = simd::template bsel2<5>(out, p);

      std::cout << "p5 " << p.reg0 << "\n";

      for (int i = 0; i < 16; i += 1) {
        std::cout << std::setw(2) << std::hex << static_cast<int>(out.reg0.at(i)) << " ";
      }
      std::cout << "\n";

      //   p[0] = row[1].left < row[5].right
      //   p[1] = row[2].left < row[6].right
      p = simd::cmp_row2(odd_0, odd_2);
      out = simd::template bsel2<4>(out, p);

      std::cout << "p4 " << p.reg0 << "\n";

      for (int i = 0; i < 16; i += 1) {
        std::cout << std::setw(2) << std::hex << static_cast<int>(out.reg0.at(i)) << " ";
      }
      std::cout << "\n";

      //   p[0] = row[5].left < row[1].right
      //   p[1] = row[6].left < row[2].right
      p = simd::cmp_row2(odd_2, odd_0);
      out = simd::template bsel2<6>(out, p);

      std::cout << "p6 " << p.reg0 << "\n";

      for (int i = 0; i < 16; i += 1) {
        std::cout << std::setw(2) << std::hex << static_cast<int>(out.reg0.at(i)) << " ";
      }
      std::cout << "\n";

      // undo the transpose
      simd::transpose_row2(odd_0, odd_2);
      simd::transpose_row2(odd_2, odd_1);
      simd::transpose_row2(odd_x, odd_0);
      simd::transpose_row2(odd_1, odd_0);

    std::cout << "even:\n";
    for (int y = 0; y < 4; y += 1) {
      for (int i = 0; i < 8; i += 1) {
        std::cout << std::setw(2) << std::hex << static_cast<int>(r.row2[y].reg0[i]) << " ";
      }
      for (int i = 0; i < 8; i += 1) {
        std::cout << " " << std::setw(2) << std::hex << static_cast<int>(r.row2[y].reg1[i]);
      }
      std::cout << "\n";
      for (int i = 0; i < 8; i += 1) {
        std::cout << std::setw(2) << std::hex << static_cast<int>(r.row2[y].reg0[8+i]) << " ";
      }
      for (int i = 0; i < 8; i += 1) {
        std::cout << " " << std::setw(2) << std::hex << static_cast<int>(r.row2[y].reg1[8+i]);
      }
      std::cout << "\n";
    }

      std::cout << "x, y = " << x << ", " << y << "\n";

      out2[y][x] = out;
    }

    // roll each row outward to process the next columns of pixels
    for (size_t i = 0; i < r.row2.size(); i += 1) {
      r.row2[i] = simd::roll_outward2(r.row2[i], x);
    }

    /*
    std::cout << "out: ";
    for (int i = 0; i < 8; i += 1) {
      std::cout << std::setw(2) << std::hex << static_cast<int>(r.row2[0].reg0[i]) << " ";
    }
    for (int i = 0; i < 8; i += 1) {
      std::cout << " " << std::setw(2) << std::hex << static_cast<int>(r.row2[0].reg1[i]);
    }
    */
    std::cout << "\n\n";
  }

  // Finish with comparisons above and below center pixel.
  for (int y = 0; y < n_iterations; y += 1) {
    // Three comparisons remain (not seven, since the center column
    // is vertically mirrored, and no self comparison). Compute the
    // last comparisons and store results in last bit (7) unused by
    // previous loop.
    p1_t p;

    p = simd::cmp_row2(r.row2[y+0], r.row2[y+3]);
    out2[y][0] = simd::template bsel2<7>(out2[y][0], p);

    p = simd::cmp_row2(r.row2[y+1], r.row2[y+2]);
    out2[y][1] = simd::template bsel2<7>(out2[y][1], p);

    x2_t &odd_x = r.row2[y];
    x2_t &odd_2 = r.row2[y+1];
    x2_t &odd_0 = r.row2[y+2];
    x2_t &odd_y = r.row2[y+3];

    // first transpose unessesary because odd_y unused in comparisons
    // simd::transpose_row2(odd_y, odd_0);
    simd::transpose_row2(odd_x, odd_0);
    simd::transpose_row2(odd_2, odd_y);
    simd::transpose_row2(odd_0, odd_2);

    p = simd::cmp_row2(odd_0, odd_2);
    out2[y][2] = simd::template bsel2<7>(out2[y][2], p);

    std::cout << "prezip\n";

    for (int j = 0; j < 4; j += 1) {
      for (int i = 0; i < 8; i += 1) {
        std::cout << std::setw(2) << std::hex << static_cast<int>(out2[0][j].reg0[i]) << " ";
      }
      for (int i = 0; i < 8; i += 1) {
        std::cout << " " << std::setw(2) << std::hex << static_cast<int>(out2[0][j].reg0[i]);
      }
      std::cout << "\n";
    }
    std::cout << "\n";

    // zip together results from each column into a single 32-bit descriptor.
    simd::zip4b2(out2[y][0], out2[y][1], out2[y][2], out2[y][3]);

    std::cout << "postzip\n";

    for (int j = 0; j < 4; j += 1) {
      for (int i = 0; i < 8; i += 1) {
        std::cout << std::setw(2) << std::hex << static_cast<int>(out2[0][j].reg0[i]) << " ";
      }
      for (int i = 0; i < 8; i += 1) {
        std::cout << " " << std::setw(2) << std::hex << static_cast<int>(out2[0][j].reg0[i]);
      }
      std::cout << "\n";
    }
    std::cout << "\n";

    // store descriptors for rows(2)*h_step(8) = (16) pixels
    uint32_t *dst0 = dst;
    uint32_t *dst1 = dst + dst_pitch;
    simd::store_feature(out2[y][0], dst0 + 0, dst_pitch);
    simd::store_feature(out2[y][1], dst0 + 4, dst_pitch);
    simd::store_feature(out2[y][2], dst1 + 0, dst_pitch);
    simd::store_feature(out2[y][3], dst1 + 4, dst_pitch);

    dst += 2*dst_pitch;
  }
}

} // detail
} // sgm_cpu
