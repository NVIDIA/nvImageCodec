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
#include <vector>
#include "metadata_kind.h"
#include "metadata_format.h"

namespace nvimgcodec {

namespace py = pybind11;
using namespace py::literals;

class Metadata
{
  public:
    Metadata() = default;
    Metadata(const nvimgcodecMetadata_t& metadata);
    Metadata(nvimgcodecMetadataKind_t kind, nvimgcodecMetadataFormat_t format, const py::bytes& buffer);

    operator nvimgcodecMetadata_t() const { return impl_; }
    nvimgcodecMetadata_t* handle() { return &impl_; }

    void allocateBuffer();

    static void exportToPython(py::module& m);

    nvimgcodecMetadataKind_t kind() const {
        return impl_.kind;
    }

    nvimgcodecMetadataFormat_t format() const {
        return impl_.format;
    }

    py::bytes buffer() const {
        if (buffer_.empty()) {
            return py::bytes();
        }
        return py::bytes(reinterpret_cast<const char*>(buffer_.data()), buffer_.size());
    }

    size_t buffer_size() const {
        return buffer_.size();
    }

  private:
    nvimgcodecMetadata_t impl_ = {NVIMGCODEC_STRUCTURE_TYPE_METADATA, sizeof(nvimgcodecMetadata_t), nullptr};
    std::vector<unsigned char> buffer_;
};

std::ostream& operator<<(std::ostream& os, const Metadata& m);

} // namespace nvimgcodec 