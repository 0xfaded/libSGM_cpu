#include <tune/array128_tune.hpp>
#include <detail/census_ops.hpp>

namespace sgm_cpu {

template struct detail::CensusOps<tune::Array128>;

}
