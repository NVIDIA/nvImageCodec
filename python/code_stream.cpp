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

#include "code_stream.h"
#include <iostream>
#include "error_handling.h"
#include "type_utils.h"
#include "region.h"
#include <ilogger.h>
#include <log.h>

namespace nvimgcodec {

CodeStream::CodeStream(nvimgcodecInstance_t instance, ILogger* logger, const std::filesystem::path& filename)
    : instance_{instance}
    , logger_{logger}
    , code_stream_{nullptr}
{
    py::gil_scoped_release release;
    auto ret = nvimgcodecCodeStreamCreateFromFile(instance, &code_stream_, filename.string().c_str());
    if (ret != NVIMGCODEC_STATUS_SUCCESS)
        throw std::runtime_error("Failed to create code stream");
}

CodeStream::CodeStream(nvimgcodecInstance_t instance, ILogger* logger, const unsigned char * data, size_t len)
    : instance_{instance}
    , logger_{logger}
    , code_stream_{nullptr}
{
    py::gil_scoped_release release;
    auto ret = nvimgcodecCodeStreamCreateFromHostMem(instance, &code_stream_, data, len);
    if (ret != NVIMGCODEC_STATUS_SUCCESS)
        throw std::runtime_error("Failed to create code stream");
}

CodeStream::CodeStream(nvimgcodecInstance_t instance, ILogger* logger, py::bytes data)
    : CodeStream(
        instance, 
        logger, 
        reinterpret_cast<const unsigned char*>(
            static_cast<std::string_view>(data).data()), 
        static_cast<std::string_view>(data).size())
{
    data_ref_bytes_ = data;
}

CodeStream::CodeStream(nvimgcodecInstance_t instance, ILogger* logger, py::array_t<uint8_t> arr)
    : CodeStream(instance, logger, 
        arr.unchecked<1>().data(0), 
        arr.size())
{
    data_ref_arr_ = arr;
}

unsigned char* CodeStream::resize_buffer(size_t bytes)
{
    if (pin_memory_) {
        pinned_buffer_->resize(bytes, 0); // Use stream 0 for now
        return static_cast<unsigned char*>(pinned_buffer_->data);
    } else {
        host_buffer_->resize(bytes);
        return host_buffer_->data();
    }
}

unsigned char* CodeStream::static_resize_buffer(void* ctx, size_t bytes)
{
    py::gil_scoped_acquire acquire;
    auto handle = reinterpret_cast<CodeStream*>(ctx);
    return handle->resize_buffer(bytes);
}

CodeStream::CodeStream(nvimgcodecInstance_t instance, ILogger* logger, size_t pre_allocated_size, bool pin_memory)
    : instance_{instance}
    , logger_{logger}
    , code_stream_{nullptr}
    , pin_memory_{pin_memory}
{
    py::gil_scoped_release release;

    if (pin_memory_) {
        pinned_buffer_ = PinnedBuffer();    
        if (pre_allocated_size > 0) {
            pinned_buffer_->resize(pre_allocated_size, 0); 
        }
    } else {
        host_buffer_ = std::vector<unsigned char>();
        if (pre_allocated_size > 0) {
            host_buffer_->resize(pre_allocated_size);
        }
    }
}

CodeStream::CodeStream(nvimgcodecInstance_t instance, ILogger* logger, nvimgcodecImageInfo_t& out_image_info, bool pin_memory)
    : instance_{instance}
    , logger_{logger}
    , image_info_{out_image_info}
    , image_info_read_{true}
    , code_stream_{nullptr}
    , pin_memory_{pin_memory}
{
    if (pin_memory_) {
        pinned_buffer_ = PinnedBuffer();
    } else {
        host_buffer_ = std::vector<unsigned char>();
    }
    py::gil_scoped_release release;
    auto ret = nvimgcodecCodeStreamCreateToHostMem(instance, &code_stream_, (void*)this, &static_resize_buffer, &out_image_info);
    if (ret != NVIMGCODEC_STATUS_SUCCESS)
        throw std::runtime_error("Failed to create code stream");
}

CodeStream::CodeStream(nvimgcodecInstance_t instance, ILogger* logger, const std::filesystem::path& filename, nvimgcodecImageInfo_t& out_image_info)
    : instance_{instance}
    , logger_{logger}
    , code_stream_{nullptr}
{
    py::gil_scoped_release release;
    auto ret = nvimgcodecCodeStreamCreateToFile(instance, &code_stream_, filename.string().c_str(), &out_image_info);
    if (ret != NVIMGCODEC_STATUS_SUCCESS)
        throw std::runtime_error("Failed to create code stream");
}

CodeStream::CodeStream(nvimgcodecInstance_t instance, ILogger* logger, nvimgcodecCodeStream_t code_stream)
    : instance_{instance}
    , logger_{logger}
    , code_stream_{code_stream}
{
}

CodeStream::~CodeStream()
{
    if (code_stream_) {
        nvimgcodecCodeStreamDestroy(code_stream_);
    }
}

void CodeStream::reuse(nvimgcodecImageInfo_t& out_image_info)
{
    py::gil_scoped_release release;
    auto ret = nvimgcodecCodeStreamCreateToHostMem(instance_, &code_stream_, (void*)this, &static_resize_buffer, &out_image_info);
    if (ret != NVIMGCODEC_STATUS_SUCCESS)
        throw std::runtime_error("Failed to create code stream");
    image_info_ = out_image_info;
    image_info_read_ = true;
}

nvimgcodecCodeStream_t CodeStream::handle() const {
    return code_stream_;
}

const nvimgcodecCodeStreamInfo_t& CodeStream::getCodeStreamInfo() const {
    if (!codestream_info_read_) {
        py::gil_scoped_release release;
        codestream_info_ = {NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr, nullptr};
        auto ret = nvimgcodecCodeStreamGetCodeStreamInfo(code_stream_, &codestream_info_);
        if (ret != NVIMGCODEC_STATUS_SUCCESS)
            throw std::runtime_error("Failed to get code stream info");
        codestream_info_read_ = true;
    }
    return codestream_info_;
}

const nvimgcodecImageInfo_t& CodeStream::getImageInfo() const {
    if (!image_info_read_) {
        py::gil_scoped_release release;
        tile_geometry_info_ = {NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO, sizeof(nvimgcodecTileGeometryInfo_t), nullptr, 0, 0, 0, 0};
        image_info_ = {NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), static_cast<void*>(&tile_geometry_info_)};
        auto ret = nvimgcodecCodeStreamGetImageInfo(code_stream_, &image_info_);
        if (ret != NVIMGCODEC_STATUS_SUCCESS)
            throw std::runtime_error("Failed to get image info");
        image_info_read_ = true;
    }
    return image_info_;
}

int CodeStream::num_images() const
{
    auto& info = getCodeStreamInfo();

    return info.num_images;
}

int CodeStream::height() const {
    auto& info = getImageInfo();
    assert(info.num_planes > 0);
    return info.plane_info[0].height;
}

int CodeStream::width() const {
    auto& info = getImageInfo();
    assert(info.num_planes > 0);
    return info.plane_info[0].width;
}

std::optional<int> CodeStream::tile_height() const {
    [[maybe_unused]] auto& info = getImageInfo();
    assert(info.struct_next == &tile_geometry_info_);
    return tile_geometry_info_.tile_height > 0 ? tile_geometry_info_.tile_height : std::optional<int>{};
}

std::optional<int> CodeStream::tile_width() const {
    [[maybe_unused]] auto& info = getImageInfo();
    assert(info.struct_next == &tile_geometry_info_);
    return tile_geometry_info_.tile_width > 0 ? tile_geometry_info_.tile_width : std::optional<int>{};
}

std::optional<int> CodeStream::num_tiles_y() const {
    [[maybe_unused]] auto& info = getImageInfo();
    assert(info.struct_next == &tile_geometry_info_);
    return tile_geometry_info_.num_tiles_y > 0 ? tile_geometry_info_.num_tiles_y : std::optional<int>{};
}

std::optional<int> CodeStream::num_tiles_x() const {
    [[maybe_unused]] auto& info = getImageInfo();
    assert(info.struct_next == &tile_geometry_info_);
    return tile_geometry_info_.num_tiles_x > 0 ? tile_geometry_info_.num_tiles_x : std::optional<int>{};
}

std::optional<int> CodeStream::tile_offset_x() const {
    [[maybe_unused]] auto& info = getImageInfo();
    assert(info.struct_next == &tile_geometry_info_);
    return tile_geometry_info_.tile_width > 0 ? tile_geometry_info_.tile_offset_x : std::optional<int>{};
}

std::optional<int> CodeStream::tile_offset_y() const {
    [[maybe_unused]] auto& info = getImageInfo();
    assert(info.struct_next == &tile_geometry_info_);
    return tile_geometry_info_.tile_height > 0 ? tile_geometry_info_.tile_offset_y : std::optional<int>{};
}

std::optional<CodeStreamView> CodeStream::view() const
{
    auto& info = getCodeStreamInfo();
    if (!info.code_stream_view) {
        return std::nullopt;
    }
    return CodeStreamView(*info.code_stream_view);
}

int CodeStream::num_channels() const 
{
    auto& info = getImageInfo();
    int total_channels = 0;
    for (uint32_t i = 0; i < info.num_planes; ++i) {
        total_channels += info.plane_info[i].num_channels;
    }
    return total_channels;
}

py::dtype CodeStream::dtype() const 
{
    auto& info = getImageInfo();
    std::string format = format_str_from_type(info.plane_info[0].sample_type);

    return py::dtype(format);
}

int CodeStream::precision() const 
{
    auto& info = getImageInfo();
    return info.plane_info[0].precision;
}

std::string CodeStream::codec_name() const 
{
    auto& info = getImageInfo();
    return info.codec_name;
}

size_t CodeStream::capacity() const 
{
    if (data_ref_bytes_.has_value()) {
        auto data_view = static_cast<std::string_view>(data_ref_bytes_.value());
        return data_view.size();
    } else if (data_ref_arr_.has_value()) {
        auto data = data_ref_arr_->unchecked<1>();
        return data.size();
    } else if (pin_memory_ && pinned_buffer_.has_value()) {
        return pinned_buffer_->capacity;
    } else if (host_buffer_.has_value()) {
        return host_buffer_->capacity();
    } else {
        auto& info = getCodeStreamInfo();
        return info.size;
    }
}

size_t CodeStream::size() const 
{
    if (pin_memory_ && pinned_buffer_.has_value()) {
        return pinned_buffer_->size;
    } else if (host_buffer_.has_value()) {
        return host_buffer_->size();
    } else {
        auto& info = getCodeStreamInfo();
        return info.size;
    }
}

bool CodeStream::pin_memory() const 
{
    return pin_memory_;
}

nvimgcodecColorSpec_t CodeStream::getColorSpec() const
{
    auto& info = getImageInfo();
    return info.color_spec;
}

nvimgcodecSampleFormat_t CodeStream::getSampleFormat() const
{
    auto& info = getImageInfo();
    return info.sample_format;
}

CodeStream* CodeStream::getSubCodeStream(const CodeStreamView& code_stream_view)
{
    nvimgcodecCodeStream_t sub_code_stream{nullptr};
    {
        size_t nimg = num_images();

        if (code_stream_view.impl_.image_idx >= nimg) {
            throw std::runtime_error("Image index #" + std::to_string(code_stream_view.impl_.image_idx) + " out of range (0, " + std::to_string(nimg - 1) + ")");
        }
        auto already_have_region = view() && view()->impl_.region.ndim != 0;
        if (already_have_region && code_stream_view.impl_.region.ndim != 0) {
            throw std::runtime_error("Cannot create a sub code stream with nested regions. This is not supported.");
        }
        py::gil_scoped_release release;
        CHECK_NVIMGCODEC(nvimgcodecCodeStreamGetSubCodeStream(code_stream_, &sub_code_stream, &code_stream_view.impl_));
    }

    return new CodeStream(instance_, logger_, sub_code_stream);
}

void CodeStream::exportToPython(py::module& m, nvimgcodecInstance_t instance, ILogger* logger)
{
    // clang-format off
    py::class_<CodeStream>(m, "CodeStream",
        R"pbdoc(
        Class representing a coded stream of image data.

        This class provides access to image informations such as dimensions, codec,
        and tiling details. It supports initialization from bytes, numpy arrays, or file path.
        )pbdoc",
        py::buffer_protocol())
        .def(py::init([instance, logger](py::bytes bytes) {
                return new CodeStream(instance, logger, bytes);
            }),
            "bytes"_a, py::keep_alive<1, 2>(),
            R"pbdoc(
            Initialize a CodeStream using bytes as input.

            Args:
                bytes: The byte data representing the encoded stream.
            )pbdoc")
        .def(py::init([instance, logger](py::array_t<uint8_t> arr) {
                return new CodeStream(instance, logger, arr);
            }),
            "array"_a, py::keep_alive<1, 2>(),
            R"pbdoc(
            Initialize a CodeStream using a numpy array of uint8 as input.

            Args:
                array: The numpy array containing the encoded stream.
            )pbdoc")
        .def(py::init([instance, logger](const std::filesystem::path& filename) {
                return new CodeStream(instance, logger, filename);
            }),
            "filename"_a,
            R"pbdoc(
            Initialize a CodeStream using a file path as input.

            Args:
                filename: The file path to the encoded stream data.
            )pbdoc")
        .def(py::init([instance, logger](size_t pre_allocated_size, bool pin_memory) {
                return new CodeStream(instance, logger, pre_allocated_size, pin_memory);
            }),
            "pre_allocated_size"_a = 0, "pin_memory"_a = true,
            R"pbdoc(
            Initialize a CodeStream for encoding output.

            Args:
                pre_allocated_size: The size of the pre-allocated memory. (default: 0)
                
                pin_memory: If True, the output memory will be pinned. (default: True)
        )pbdoc")
        .def_buffer([](CodeStream& code_stream) -> py::buffer_info {

               auto make_buffer_info = [](void* ptr, size_t size) {
                   std::vector<size_t> shape{size};
                   std::vector<size_t> stride{1};
                   return py::buffer_info(
                       static_cast<unsigned char*>(ptr), /* Pointer to buffer */
                       1,                                /* Size of one scalar */
                       "B",                              /* Python struct-style format descriptor */
                       1,                                /* Number of dimensions */
                       shape,                            /* Buffer dimensions */
                       stride                            /* Strides (in bytes) for each index */
                   );
               };

               if (code_stream.pin_memory_ && code_stream.pinned_buffer_.has_value()) {
                   return make_buffer_info(code_stream.pinned_buffer_->data, code_stream.pinned_buffer_->size);
               } else if (!code_stream.pin_memory_ && code_stream.host_buffer_.has_value()) {
                   return make_buffer_info(code_stream.host_buffer_->data(), code_stream.host_buffer_->size());
               } else if (code_stream.data_ref_bytes_.has_value()) {
                   auto data_view = static_cast<std::string_view>(code_stream.data_ref_bytes_.value());
                   return make_buffer_info(const_cast<void*>(static_cast<const void*>(data_view.data())), data_view.size());
               } else if (code_stream.data_ref_arr_.has_value()) {
                   auto data = code_stream.data_ref_arr_->unchecked<1>();
                   return make_buffer_info(const_cast<void*>(static_cast<const void*>(data.data(0))), data.size());
               } else {
                   throw std::runtime_error("Not initialized buffer");
               }
            })
        .def("get_sub_code_stream", [](CodeStream& self, size_t image_idx, std::optional<Region> region) -> CodeStream* {
                return self.getSubCodeStream(CodeStreamView(image_idx, region.value_or(Region())));
            },
            "image_idx"_a = 0, "region"_a = std::nullopt,
            /* Keep this (1) CodeStream alive as long as newly created and returned (0) codeStream is alive.
             * This is required as this CodeStream may keep alive python object with data (like bytes).
            */
            py::keep_alive<0, 1>(),
            R"pbdoc(
            Get a sub code stream for a specific image index and optional region.

            Args:
                image_idx: Index of the image in the code stream. Defaults to 0.
                
                region: Optional region of interest within the image.

            Returns:
                A new CodeStream object representing the sub code stream.
            )pbdoc")
        .def("get_sub_code_stream", [](CodeStream& self, const CodeStreamView& view) -> CodeStream* {
                return self.getSubCodeStream(view);
            },
            "view"_a,
            /* Keep this (1) CodeStream alive as long as newly created and returned (0) codeStream is alive.
             * This is required as this CodeStream may keep alive python object with data (like bytes).
            */
            py::keep_alive<0, 1>(),
            R"pbdoc(
            Get a sub code stream using a CodeStreamView object.

            Args:
                view: A CodeStreamView object specifying the image index and optional region.

            Returns:
                A new CodeStream object representing the sub code stream.
            )pbdoc")
        .def_property_readonly("num_images", &CodeStream::num_images, 
            R"pbdoc(
            The number of images in the code stream.
            )pbdoc")
        .def_property_readonly("height", &CodeStream::height, 
            R"pbdoc(
            The vertical dimension of the entire image in pixels.
            )pbdoc")
        .def_property_readonly("width", &CodeStream::width, 
            R"pbdoc(
            The horizontal dimension of the entire image in pixels.
            )pbdoc")
        .def_property_readonly("num_channels", &CodeStream::num_channels, 
            R"pbdoc(
            The overall number of channels in the image across all planes.


            )pbdoc")
        .def_property_readonly("dtype", &CodeStream::dtype, 
            R"pbdoc(
            Data type of samples.
            )pbdoc")
        .def_property_readonly("precision", &CodeStream::precision, 
            R"pbdoc(
            Maximum number of significant bits in data type. Value 0 
            means that precision is equal to data type bit depth.
            )pbdoc")
        .def_property_readonly("codec_name", &CodeStream::codec_name, 
            R"pbdoc(
            Image format.
            )pbdoc")
        .def_property_readonly("capacity", &CodeStream::capacity, 
            R"pbdoc(
            The capacity of the internal buffer in bytes.
            )pbdoc")
        .def_property_readonly("size", &CodeStream::size, 
            R"pbdoc(
            The size of the compressed bitstream in bytes.
            )pbdoc")
        .def_property_readonly("pin_memory", &CodeStream::pin_memory, 
            R"pbdoc(
            Whether the internal buffer is pinned.
            )pbdoc")
        .def_property_readonly("color_spec", &CodeStream::getColorSpec,
            R"pbdoc(
            Property to get the color specification for the code stream.

            This determines the color space or color profile of the image data.
            For instance, sRGB is a common color specification. 
            
            Please notice that color specification of decoded Image depends of color_spec
            parameter used in decode function so can be different from the one of the code stream.
            )pbdoc")
        .def_property_readonly("sample_format", &CodeStream::getSampleFormat, 
            R"pbdoc(
            The sample format of the code stream indicating how color components are matched to channels and channels to planes.
            )pbdoc")
        .def_property_readonly("view", &CodeStream::view,
            R"pbdoc(
            The view of this code stream, if it was created as a sub code stream.
            Contains the image index and optional region of interest.

            Returns:
                CodeStreamView object if this is a sub code stream, None otherwise.
            )pbdoc")
        .def_property_readonly("num_tiles_y", &CodeStream::num_tiles_y, 
            R"pbdoc(
            The number of tiles arranged along the vertical axis of the image.
            )pbdoc")
        .def_property_readonly("num_tiles_x", &CodeStream::num_tiles_x, 
            R"pbdoc(
            The number of tiles arranged along the horizontal axis of the image.
            )pbdoc")
        .def_property_readonly("tile_height", &CodeStream::tile_height, 
            R"pbdoc(
            The vertical dimension of each individual tile within the image.
            )pbdoc")
        .def_property_readonly("tile_width", &CodeStream::tile_width, 
            R"pbdoc(
            The horizontal dimension of each individual tile within the image.
            )pbdoc")
        .def_property_readonly("tile_offset_x", &CodeStream::tile_offset_x, 
            R"pbdoc(
            The horizontal offset of the tile grid to the left of the image.
            )pbdoc")
        .def_property_readonly("tile_offset_y", &CodeStream::tile_offset_y, 
            R"pbdoc(
            The vertical offset of the tile grid to the top of the image.
            )pbdoc")
        .def("__repr__", [](const CodeStream* cs) {
            std::stringstream ss;
            ss << *cs;
            return ss.str();
        },
        R"pbdoc(
        Returns a string representation of the CodeStream object, displaying core attributes.
        )pbdoc");
    // clang-format on
    py::implicitly_convertible<py::bytes, CodeStream>();
    py::implicitly_convertible<py::array_t<uint8_t>, CodeStream>();
    py::implicitly_convertible<std::string, CodeStream>();
    py::implicitly_convertible<py::tuple, CodeStream>();
    py::implicitly_convertible<CodeStream, CodeStream>();
}


std::ostream& operator<<(std::ostream& os, const CodeStream& cs)
{
    os << "CodeStream("
        << " codec_name=" << cs.codec_name()
        << " num_images=" << cs.num_images();
    auto view_opt = cs.view();
    if (view_opt) {
        os << " view=" << *view_opt;
    }
    os << " height=" << cs.height()
        << " width=" << cs.width()
        << " num_channels=" << cs.num_channels()
        << " dtype=" << dtype_to_str(cs.dtype())
        << " precision=" << cs.precision()
        << " color_spec=" << cs.getColorSpec()
        << " sample_format=" << cs.getSampleFormat()
        << " size=" << cs.size()
        << " capacity=" << cs.capacity();
    auto num_tiles_y = cs.num_tiles_y();
    if (num_tiles_y)
        os << " num_tiles_y=" << num_tiles_y.value();
    auto num_tiles_x = cs.num_tiles_x();
    if (num_tiles_x)
        os << " num_tiles_x=" << num_tiles_x.value();
    auto tile_height = cs.tile_height();
    if (tile_height)
        os << " tile_height=" << tile_height.value();
    auto tile_width = cs.tile_width();
    if (tile_width)
        os << " tile_width=" << tile_width.value();
    os << ")";
    return os;
}


} // namespace nvimgcodec
