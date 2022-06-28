#pragma once

#include <memory>

#include <types.hpp>

namespace sgm_cpu {

template <class Arch>
class CensusTransform {

 public:
  // other input widths will require a new set of SIMD implementations
	using input_type = char;

 private:
  std::unique_ptr<feature_type[]> m_feature_buffer;

 public:
	CensusTransform();

	const feature_type *get_output() const {
		return m_feature_buffer.get();
	}
	
	void execute(
      const input_type *src,
      int width,
      int height,
      int src_pitch,
      int dst_pitch = -1);

 private:
};

}

