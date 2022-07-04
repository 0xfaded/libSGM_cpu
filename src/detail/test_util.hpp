#include <random>
#include <iostream>

namespace sgm_cpu {
namespace test {

std::vector<uint8_t> random_patch(int w, int h, std::minstd_rand0 &rng);
std::vector<uint32_t> random_descriptors(int w, int h, int border,
    std::minstd_rand0 &rng);

uint32_t compute_census(const uint8_t *src, int src_pitch);

std::vector<uint32_t> apply_census(const uint8_t *src,
    int width, int height, int src_pitch);

std::ostream &print_descriptor(std::ostream &os, uint32_t desc);

} // namespace test
} // namespace sgm_cpu

