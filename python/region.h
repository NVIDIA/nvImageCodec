/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <nvimgcodec.h>
#include <pybind11/pybind11.h>

#include <iostream>

#include <sstream>
#include <stdexcept>
#include <variant>
#include <vector>

namespace nvimgcodec {

namespace py = pybind11;
using namespace py::literals;

// helper struct for variant visitor
template<class... Ts>
struct overloaded : Ts... { using Ts::operator()...; };

template<class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;


class Region
{
public:
    using fillSampleType_t = std::variant<uint32_t, int32_t, float>;

    Region() = default;
    Region(const nvimgcodecRegion_t& region)
      : impl_{region}
    {
    }

    template <typename Container>
    Region(Container&& start, Container&& end, const std::vector<fillSampleType_t>& fill_samples)
    {
        impl_.ndim = start.size();
        if (start.size() != end.size() && (!start.empty() && !end.empty())) {
            throw std::runtime_error("Dimension mismatch");
        } else if (impl_.ndim  > NVIMGCODEC_MAX_NUM_DIM) {
            throw std::runtime_error(
                "Too many dimensions: " + std::to_string(impl_.ndim) +
                ", at most " + std::to_string(NVIMGCODEC_MAX_NUM_DIM) + " are allowed."
            );
        }
        for (int i = 0; i < impl_.ndim; i++) {
            impl_.start[i] = start[i];
            impl_.end[i] = end[i];
            if (end[i] <= start[i]) {
              throw std::runtime_error(
                "Invalid dimension on index " + std::to_string(i) +
                "; start = " + std::to_string(start[i]) +
                ", end = " + std::to_string(end[i])
              );
            }
        }

        if (fill_samples.size() > NVIMGCODEC_MAX_ROI_FILL_CHANNELS) {
            throw std::runtime_error(
                "Too many fill values: " + std::to_string(fill_samples.size()) +
                ", at most " + std::to_string(NVIMGCODEC_MAX_ROI_FILL_CHANNELS) + " are allowed."
            );
        }

        impl_.out_of_bounds_policy = NVIMGCODEC_OUT_OF_BOUNDS_POLICY_CONSTANT;
        // by default out_of_bounds_samples.type will be set to 0, as value initialization is used for creating impl_ (line 151)
        for (size_t i = 0; i < fill_samples.size(); ++i) {
            std::visit(overloaded{
                [&](uint32_t val) {
                    impl_.out_of_bounds_samples[i].value.as_uint = val;
                    impl_.out_of_bounds_samples[i].type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT32;
                }, [&](int32_t val) {
                    impl_.out_of_bounds_samples[i].value.as_int = val;
                    impl_.out_of_bounds_samples[i].type = NVIMGCODEC_SAMPLE_DATA_TYPE_INT32;
                }, [&](float val) {
                    impl_.out_of_bounds_samples[i].value.as_float = val;
                    impl_.out_of_bounds_samples[i].type = NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32;
                }
            }, fill_samples[i]);
        }
    }

    template <typename Container>
    Region(Container&& start, Container&& end, fillSampleType_t fill_sample)
        : Region(start, end, std::vector(NVIMGCODEC_MAX_ROI_FILL_CHANNELS, fill_sample))
    {}

    operator nvimgcodecRegion_t() const { return impl_; }

    static void exportToPython(py::module& m);

    int ndim() const {
      return impl_.ndim;
    }

    py::tuple start() const {
      auto ret = py::tuple(ndim());
      for (int i = 0; i < ndim(); i++) {
        ret[i] = impl_.start[i];
      }
      return ret;
    }

    py::tuple end() const {
      auto ret = py::tuple(ndim());
      for (int i = 0; i < ndim(); i++) {
        ret[i] = impl_.end[i];
      }
      return ret;
    }

    std::vector<fillSampleType_t> out_of_bounds_samples() const {
        std::vector<fillSampleType_t> ret;
        for (const auto& sample : impl_.out_of_bounds_samples) {
            switch (sample.type) {
                case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT32:
                    ret.emplace_back(sample.value.as_uint);
                    break;

                case NVIMGCODEC_SAMPLE_DATA_TYPE_INT32:
                    ret.emplace_back(sample.value.as_int);
                    break;

                case NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32:
                    ret.emplace_back(sample.value.as_float);
                    break;

                default: // for channels unspecified by user, type will be 0 and fill value will also be 0
                    ret.emplace_back(0);
                    break;
            }
        }

        return ret;
    }
    nvimgcodecRegion_t impl_ = {NVIMGCODEC_STRUCTURE_TYPE_REGION, sizeof(nvimgcodecRegion_t), nullptr, 0};
};

std::ostream& operator<<(std::ostream& os, const Region& r);


} // namespace nvimgcodec
