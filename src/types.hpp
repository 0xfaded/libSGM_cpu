/*
Copyright 2022 Carl Chatfield
Copyright 2016 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Note[carl]: Types are copied from libSGM for compatibility

#pragma once

#include <cstdint>

namespace sgm_cpu {

using feature_type = uint32_t;
using cost_type = uint8_t;
using cost_sum_type = uint16_t;
using output_type = uint16_t;

} // namespace sgm_cpu
