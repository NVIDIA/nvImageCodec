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
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <memory>
#include <sstream>
#include <optional>

namespace nvimgcodec {

namespace py = pybind11;
using namespace py::literals;
class CodeStream
{
  public:
    CodeStream(nvimgcodecInstance_t instance, const std::filesystem::path& filename);
    CodeStream(nvimgcodecInstance_t instance, const unsigned char* data, size_t length);
    CodeStream(nvimgcodecInstance_t instance, py::bytes);
    CodeStream(nvimgcodecInstance_t instance, py::array_t<uint8_t>);
    static void exportToPython(py::module& m, nvimgcodecInstance_t instance);
    nvimgcodecCodeStream_t handle() const;

    int width() const;
    int height() const;
    int channels() const;

    std::optional<int> num_tiles_x() const;
    std::optional<int> num_tiles_y() const;
    std::optional<int> tile_width() const;
    std::optional<int> tile_height() const;

    py::dtype dtype() const;
    int precision() const;
    std::string codec_name() const;

    CodeStream();

    CodeStream(CodeStream&&) = default;
    CodeStream& operator=(CodeStream&&) = default;

    CodeStream(const CodeStream&) = delete;
    CodeStream& operator=(CodeStream const&) = delete;

    ~CodeStream();

  private:
    const nvimgcodecImageInfo_t& ImageInfo() const;
    mutable nvimgcodecJpeg2kImageInfo_t jp2_info_{NVIMGCODEC_STRUCTURE_TYPE_JPEG2K_IMAGE_INFO, sizeof(nvimgcodecJpeg2kImageInfo_t), 0};
    mutable nvimgcodecImageInfo_t info_{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), static_cast<void*>(&jp2_info_)};
    mutable bool info_read_ = false;

    nvimgcodecCodeStream_t code_stream_;
    // Using those to keep a reference to the argument data,
    // so that they are kept alive throughout the lifetime of the object
    py::bytes data_ref_bytes_;
    py::array_t<uint8_t> data_ref_arr_;
};

std::ostream& operator<<(std::ostream& os, const CodeStream& cs);

} // namespace nvimgcodec
