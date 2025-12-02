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
#include "imgproc/pinned_buffer.h"

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
class ILogger;
class CodeStream
{
  public:
    CodeStream(nvimgcodecInstance_t instance, ILogger* logger, py::bytes); //For FromMemHost provided by bytes
    CodeStream(nvimgcodecInstance_t instance, ILogger* logger, py::array_t<uint8_t>); //For FromMemHost provided by array

    CodeStream(nvimgcodecInstance_t instance, ILogger* logger, size_t pre_allocated_size, bool pin_memory = true); //For ToMemHost
    CodeStream(nvimgcodecInstance_t instance, ILogger* logger, nvimgcodecImageInfo_t& out_image_info, bool pin_memory = true); //For ToMemHost

    CodeStream(nvimgcodecInstance_t instance, ILogger* logger, const std::filesystem::path& filename); //For FromFile
    CodeStream(nvimgcodecInstance_t instance, ILogger* logger, const std::filesystem::path& filename, nvimgcodecImageInfo_t& out_image_info); //For ToFile

    CodeStream(nvimgcodecInstance_t instance, ILogger* logger, nvimgcodecCodeStream_t code_stream);// For SubCodeStream

    CodeStream(CodeStream&& other) noexcept = default;
    CodeStream& operator=(CodeStream&& other) noexcept = default;

    CodeStream(const CodeStream&) = delete;
    CodeStream& operator=(CodeStream const&) = delete;

    ~CodeStream();

    void reuse(nvimgcodecImageInfo_t& out_image_info);

    static void exportToPython(py::module& m, nvimgcodecInstance_t instance, ILogger* logger);
    nvimgcodecCodeStream_t handle() const;

    int num_images() const;
    int width() const;
    int height() const;

    int num_channels() const;

    std::optional<CodeStreamView> view() const;

    std::optional<int> num_tiles_x() const;
    std::optional<int> num_tiles_y() const;
    std::optional<int> tile_width() const;
    std::optional<int> tile_height() const;
    std::optional<int> tile_offset_x() const;
    std::optional<int> tile_offset_y() const;

    py::dtype dtype() const;
    int precision() const;
    std::string codec_name() const;
    size_t size() const;
    size_t capacity() const;
    bool pin_memory() const;

    nvimgcodecColorSpec_t getColorSpec() const;
    nvimgcodecSampleFormat_t getSampleFormat() const;

    CodeStream* getSubCodeStream(const CodeStreamView& code_stream_view);
    
    const nvimgcodecCodeStreamInfo_t& getCodeStreamInfo() const;
    const nvimgcodecImageInfo_t& getImageInfo() const;
  private:
    CodeStream(nvimgcodecInstance_t instance, ILogger* logger, const unsigned char* data, size_t length); //For FromMemHost
    unsigned char* resize_buffer(size_t bytes);
    static unsigned char* static_resize_buffer(void* ctx, size_t bytes);

    nvimgcodecInstance_t instance_ = nullptr;
    ILogger* logger_ = nullptr;

    mutable nvimgcodecTileGeometryInfo_t tile_geometry_info_{NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO, sizeof(nvimgcodecTileGeometryInfo_t), 0};
    mutable nvimgcodecImageInfo_t image_info_{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), static_cast<void*>(&tile_geometry_info_)};
    mutable nvimgcodecCodeStreamInfo_t codestream_info_{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr};

    mutable bool image_info_read_ = false;
    mutable bool codestream_info_read_ = false;

    nvimgcodecCodeStream_t code_stream_ = nullptr;

    // Referenced buffers
    // Using those to keep a reference to the argument data,
    // so that they are kept alive throughout the lifetime of the object
    std::optional<py::bytes> data_ref_bytes_ = std::nullopt;
    std::optional<py::array_t<uint8_t>> data_ref_arr_ = std::nullopt;

    //Owned buffers
    std::optional<PinnedBuffer> pinned_buffer_ = std::nullopt;
    std::optional<std::vector<unsigned char>> host_buffer_ = std::nullopt;
    bool pin_memory_ = true; //If true, the buffer will be pinned, otherwise it will be allocated on the host

};

std::ostream& operator<<(std::ostream& os, const CodeStream& cs);

} // namespace nvimgcodec
