/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "region.h"

namespace nvimgcodec {

namespace py = pybind11;
using namespace py::literals;

class CodeStreamView
{
  public:
    CodeStreamView() = default;
    CodeStreamView(const nvimgcodecCodeStreamView_t& view)
      : impl_{view}
    {
        impl_.struct_next = nullptr;
    }

    CodeStreamView(size_t image_idx, const std::optional<Region>& region = std::nullopt,
                   size_t bitstream_offset = 0, uint32_t limit_images = 0)
    {
        impl_.image_idx = image_idx;
        impl_.region = region.value_or(Region());
        impl_.bitstream_offset = bitstream_offset;
        impl_.limit_images = limit_images;
    }

    operator nvimgcodecCodeStreamView_t() const { return impl_; }

    static void exportToPython(py::module& m);

    size_t imageIdx() const {
      return impl_.image_idx;
    }

    std::optional<Region> region() const {
        return impl_.region.ndim != 0 ? std::optional<Region>(Region(impl_.region)) : std::nullopt;
    }

    size_t bitstreamOffset() const {
        return impl_.bitstream_offset;
    }

    uint32_t limitImages() const {
        return impl_.limit_images;
    }

    nvimgcodecCodeStreamView_t impl_ = {NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_VIEW, sizeof(nvimgcodecCodeStreamView_t), nullptr, 0, {}, 0, 0};
};

std::ostream& operator<<(std::ostream& os, const CodeStreamView& v);

} // namespace nvimgcodec
