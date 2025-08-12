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
#include "code_stream_view.h"

namespace nvimgcodec {

namespace py = pybind11;
using namespace py::literals;
class Region;
class CodeStream
{
  public:
    CodeStream(nvimgcodecInstance_t instance, const std::filesystem::path& filename);
    CodeStream(nvimgcodecInstance_t instance, const unsigned char* data, size_t length);
    CodeStream(nvimgcodecInstance_t instance, py::bytes);
    CodeStream(nvimgcodecInstance_t instance, py::array_t<uint8_t>);
    static void exportToPython(py::module& m, nvimgcodecInstance_t instance);
    nvimgcodecCodeStream_t handle() const;

    int num_images() const;
    int width() const;
    int height() const;

    int channels() const;

    std::optional<CodeStreamView> view() const;

    std::optional<int> num_tiles_x() const;
    std::optional<int> num_tiles_y() const;
    std::optional<int> tile_width() const;
    std::optional<int> tile_height() const;

    py::dtype dtype() const;
    int precision() const;
    std::string codec_name() const;

    CodeStream* getSubCodeStream(const CodeStreamView& code_stream_view);

    CodeStream();
    CodeStream(nvimgcodecCodeStream_t code_stream);

    CodeStream(CodeStream&&) = default;
    CodeStream& operator=(CodeStream&&) = default;

    CodeStream(const CodeStream&) = delete;
    CodeStream& operator=(CodeStream const&) = delete;

    ~CodeStream();
    
    const nvimgcodecCodeStreamInfo_t& getCodeStreamInfo() const;
    const nvimgcodecImageInfo_t& getImageInfo() const;
  private:
    mutable nvimgcodecTileGeometryInfo_t tile_geometry_info_{NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO, sizeof(nvimgcodecTileGeometryInfo_t), 0};
    mutable nvimgcodecImageInfo_t image_info_{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), static_cast<void*>(&tile_geometry_info_)};
    mutable nvimgcodecCodeStreamInfo_t codestream_info_{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr};
    mutable bool info_read_ = false;
    mutable bool codestream_info_read_ = false;

    nvimgcodecCodeStream_t code_stream_;
    // Using those to keep a reference to the argument data,
    // so that they are kept alive throughout the lifetime of the object
    py::bytes data_ref_bytes_;
    py::array_t<uint8_t> data_ref_arr_;
};

std::ostream& operator<<(std::ostream& os, const CodeStream& cs);

} // namespace nvimgcodec
