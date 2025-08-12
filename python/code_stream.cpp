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

namespace nvimgcodec {

CodeStream::CodeStream(nvimgcodecInstance_t instance, const std::filesystem::path& filename)
    : code_stream_{nullptr}
{
    py::gil_scoped_release release;
    auto ret = nvimgcodecCodeStreamCreateFromFile(instance, &code_stream_, filename.string().c_str());
    if (ret != NVIMGCODEC_STATUS_SUCCESS)
        throw std::runtime_error("Failed to create code stream");
}

CodeStream::CodeStream(nvimgcodecInstance_t instance, const unsigned char * data, size_t len)
    : code_stream_{nullptr}
{
    py::gil_scoped_release release;
    auto ret = nvimgcodecCodeStreamCreateFromHostMem(instance, &code_stream_, data, len);
    if (ret != NVIMGCODEC_STATUS_SUCCESS)
        throw std::runtime_error("Failed to create code stream");
}

CodeStream::CodeStream(nvimgcodecInstance_t instance, py::bytes data)
    : code_stream_{nullptr}
{
    data_ref_bytes_ = data;
    auto data_view = static_cast<std::string_view>(data_ref_bytes_);
    py::gil_scoped_release release;
    auto ret = nvimgcodecCodeStreamCreateFromHostMem(instance, &code_stream_, reinterpret_cast<const unsigned char*>(data_view.data()), data_view.size());
    if (ret != NVIMGCODEC_STATUS_SUCCESS)
        throw std::runtime_error("Failed to create code stream");}

CodeStream::CodeStream(nvimgcodecInstance_t instance, py::array_t<uint8_t> arr)
    : code_stream_{nullptr}
{
    data_ref_arr_ = arr;
    auto data = data_ref_arr_.unchecked<1>();
    py::gil_scoped_release release;
    auto ret = nvimgcodecCodeStreamCreateFromHostMem(instance, &code_stream_, data.data(0), data.size());
    if (ret != NVIMGCODEC_STATUS_SUCCESS)
        throw std::runtime_error("Failed to create code stream");
}

CodeStream::CodeStream()
    : code_stream_{nullptr}
{
}

CodeStream::CodeStream(nvimgcodecCodeStream_t code_stream)
    : code_stream_{code_stream}
{

}

CodeStream::~CodeStream()
{
    nvimgcodecCodeStreamDestroy(code_stream_);
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
    if (!info_read_) {
        py::gil_scoped_release release;
        tile_geometry_info_ = {NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO, sizeof(nvimgcodecTileGeometryInfo_t), nullptr, 0, 0, 0, 0};
        image_info_ = {NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), static_cast<void*>(&tile_geometry_info_)};
        auto ret = nvimgcodecCodeStreamGetImageInfo(code_stream_, &image_info_);
        if (ret != NVIMGCODEC_STATUS_SUCCESS)
            throw std::runtime_error("Failed to get image info");
        info_read_ = true;
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

std::optional<CodeStreamView> CodeStream::view() const
{
    auto& info = getCodeStreamInfo();
    if (!info.code_stream_view) {
        return std::nullopt;
    }
    return CodeStreamView(*info.code_stream_view);
}

int CodeStream::channels() const 
{
    auto& info = getImageInfo();
    return info.num_planes;
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

CodeStream* CodeStream::getSubCodeStream(const CodeStreamView& code_stream_view)
{
    nvimgcodecCodeStream_t sub_code_stream{nullptr};
    {
        size_t nimg = num_images();

        if (code_stream_view.impl_.image_idx >= nimg) {
            throw std::runtime_error("Image index #" + std::to_string(code_stream_view.impl_.image_idx) + " out of range (0, " + std::to_string(nimg - 1) + ")");
        }
        auto already_have_region = view() && view()->impl_.region.ndim > 0;
        if (already_have_region && code_stream_view.impl_.region.ndim > 0) {
            throw std::runtime_error("Cannot create a sub code stream with nested regions. This is not supported.");
        }
        py::gil_scoped_release release;
        CHECK_NVIMGCODEC(nvimgcodecCodeStreamGetSubCodeStream(code_stream_, &sub_code_stream, &code_stream_view.impl_));
    }

    return new CodeStream(sub_code_stream);
}

void CodeStream::exportToPython(py::module& m, nvimgcodecInstance_t instance)
{
    // clang-format off
    py::class_<CodeStream>(m, "CodeStream",
        R"pbdoc(
        Class representing a coded stream of image data.

        This class provides access to image informations such as dimensions, codec,
        and tiling details. It supports initialization from bytes, numpy arrays, or file path.
        )pbdoc")
        .def(py::init([instance](py::bytes bytes) {
            return new CodeStream(instance, bytes);
        }),
            "bytes"_a, py::keep_alive<1, 2>(),
            R"pbdoc(
            Initialize a CodeStream using bytes as input.

            Args:
                bytes: The byte data representing the encoded stream.
            )pbdoc")
        .def(py::init([instance](py::array_t<uint8_t> arr) {
            return new CodeStream(instance, arr);
        }),
            "array"_a, py::keep_alive<1, 2>(),
            R"pbdoc(
            Initialize a CodeStream using a numpy array of uint8 as input.

            Args:
                array: The numpy array containing the encoded stream.
            )pbdoc")
        .def(py::init([instance](const std::filesystem::path& filename) {
            return new CodeStream(instance, filename);
        }),
            "filename"_a,
            R"pbdoc(
            Initialize a CodeStream using a file path as input.

            Args:
                filename: The file path to the encoded stream data.
            )pbdoc")
        .def("getSubCodeStream", [](CodeStream& self, size_t image_idx, std::optional<Region> region) -> CodeStream* {
                return self.getSubCodeStream(CodeStreamView(image_idx, region.value_or(Region())));
            },
            "image_idx"_a = 0, "region"_a = std::nullopt,
            R"pbdoc(
            Get a sub code stream for a specific image index and optional region.

            Args:
                image_idx: Index of the image in the code stream. Defaults to 0.
                
                region: Optional region of interest within the image.

            Returns:
                A new CodeStream object representing the sub code stream.
            )pbdoc")
        .def("getSubCodeStream", [](CodeStream& self, const CodeStreamView& view) -> CodeStream* {
                return self.getSubCodeStream(view);
            },
            "view"_a,
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
        .def_property_readonly("channels", &CodeStream::channels, 
            R"pbdoc(
            The number of channels in the image.
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
        << " channels=" << cs.channels()
        << " dtype=" << dtype_to_str(cs.dtype())
        << " precision=" << cs.precision();
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
