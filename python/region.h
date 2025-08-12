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
#include <sstream>
#include <stdexcept>
#include <iostream>

namespace nvimgcodec {

namespace py = pybind11;
using namespace py::literals;

class Region
{
  public:
    Region() = default;
    Region(const nvimgcodecRegion_t& region)
      : impl_{region}
    {
    }

    template <typename Container>
    Region(Container&& start, Container&& end)
    {
        impl_.ndim = start.size();
        if (start.size() != end.size() && (!start.empty() && !end.empty())) {
            throw std::runtime_error("Dimension mismatch");
        } else if (impl_.ndim  > NVIMGCODEC_MAX_NUM_DIM) {
            throw std::runtime_error("Too many dimensions: " + std::to_string(impl_.ndim));
        }
        for (int i = 0; i < impl_.ndim; i++) {
            impl_.start[i] = start[i];
            impl_.end[i] = end[i];
        }
    }

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

    nvimgcodecRegion_t impl_ = {NVIMGCODEC_STRUCTURE_TYPE_REGION, sizeof(nvimgcodecRegion_t), nullptr, 0};
};

std::ostream& operator<<(std::ostream& os, const Region& r);


} // namespace nvimgcodec
