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
#include <tiff_utils.h>
#include <ilogger.h>
#include <log.h>

namespace nvimgcodec {

CodeStream::CodeStream(nvimgcodecInstance_t instance, ILogger* logger, const std::filesystem::path& filename,
                       std::optional<size_t> bitstream_offset)
    : instance_{instance}
    , logger_{logger}
    , code_stream_{nullptr}
{
    py::gil_scoped_release release;
    const size_t offset = bitstream_offset.value_or(0);
    nvimgcodecCodeStreamView_t view{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_VIEW, sizeof(nvimgcodecCodeStreamView_t),
        nullptr, 0, {}, offset};
    auto ret = nvimgcodecCodeStreamCreateFromFile(instance, &code_stream_, filename.string().c_str(),
        offset != 0 ? &view : nullptr);
    if (ret != NVIMGCODEC_STATUS_SUCCESS)
        throw std::runtime_error("Failed to create code stream");
    if (offset != 0) {
        view_ = CodeStreamView(0, std::nullopt, offset);
    }
}

CodeStream::CodeStream(nvimgcodecInstance_t instance, ILogger* logger, const unsigned char * data, size_t len,
                       std::optional<size_t> bitstream_offset)
    : instance_{instance}
    , logger_{logger}
    , code_stream_{nullptr}
{
    py::gil_scoped_release release;
    const size_t offset = bitstream_offset.value_or(0);
    nvimgcodecCodeStreamView_t view{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_VIEW, sizeof(nvimgcodecCodeStreamView_t),
        nullptr, 0, {}, offset};
    auto ret = nvimgcodecCodeStreamCreateFromHostMem(instance, &code_stream_, data, len,
        offset != 0 ? &view : nullptr);
    if (ret != NVIMGCODEC_STATUS_SUCCESS)
        throw std::runtime_error("Failed to create code stream");
    if (offset != 0) {
        view_ = CodeStreamView(0, std::nullopt, offset);
    }
}

CodeStream::CodeStream(nvimgcodecInstance_t instance, ILogger* logger, py::bytes data,
                       std::optional<size_t> bitstream_offset)
    : CodeStream(
        instance,
        logger,
        reinterpret_cast<const unsigned char*>(
            static_cast<std::string_view>(data).data()),
        static_cast<std::string_view>(data).size(),
        bitstream_offset)
{
    data_ref_bytes_ = data;
}

CodeStream::CodeStream(nvimgcodecInstance_t instance, ILogger* logger, py::array_t<uint8_t> arr,
                       std::optional<size_t> bitstream_offset)
    : CodeStream(instance, logger,
        arr.unchecked<1>().data(0),
        arr.size(),
        bitstream_offset)
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
    // Encoder tasks keep the CodeStream alive until callbacks complete. The
    // callback only mutates native storage owned by this CodeStream.
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
    // out_image_info is the requested output layout. Query after encode so the
    // returned CodeStream exposes metadata from the produced codestream.
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

CodeStream::CodeStream(nvimgcodecInstance_t instance, ILogger* logger, nvimgcodecCodeStream_t code_stream,
                       std::optional<CodeStreamView> view)
    : instance_{instance}
    , logger_{logger}
    , code_stream_{code_stream}
    , view_{std::move(view)}
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
    view_ = std::nullopt;
    py::gil_scoped_release release;
    // From-file / from-bytes constructors leave the output buffer uninitialized;
    // prime whichever variant matches pin_memory_ so static_resize_buffer can grow it.
    if (pin_memory_) {
        if (!pinned_buffer_) pinned_buffer_ = PinnedBuffer();
    } else {
        if (!host_buffer_) host_buffer_ = std::vector<unsigned char>();
    }
    auto ret = nvimgcodecCodeStreamCreateToHostMem(instance_, &code_stream_, (void*)this, &static_resize_buffer, &out_image_info);
    if (ret != NVIMGCODEC_STATUS_SUCCESS)
        throw std::runtime_error("Failed to create code stream");
    // out_image_info is the requested output layout. Reused CodeStreams must
    // refresh metadata from the produced codestream after encoding.
    image_info_read_ = false;
    codestream_info_read_ = false;
}

nvimgcodecCodeStream_t CodeStream::handle() const {
    return code_stream_;
}

const nvimgcodecCodeStreamInfo_t& CodeStream::getCodeStreamInfo() const {
    if (!codestream_info_read_) {
        py::gil_scoped_release release;
        // Reset the TIFF extension and chain it to codestream_info_
        codestream_info_tiff_ext_ = {NVIMGCODEC_STRUCTURE_TYPE_TIFF_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfoTiffExt_t), nullptr, 0};
        codestream_info_ = {NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), 
                           static_cast<void*>(&codestream_info_tiff_ext_), nullptr};
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
        // Ensure our chain is used so tile accessors (and repr) never see a C++ internal pointer
        image_info_.struct_next = static_cast<void*>(&tile_geometry_info_);
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
    auto& info = getImageInfo();
    if (info.struct_next != &tile_geometry_info_)
        return std::nullopt;
    return tile_geometry_info_.tile_height > 0 ? tile_geometry_info_.tile_height : std::optional<int>{};
}

std::optional<int> CodeStream::tile_width() const {
    auto& info = getImageInfo();
    if (info.struct_next != &tile_geometry_info_)
        return std::nullopt;
    return tile_geometry_info_.tile_width > 0 ? tile_geometry_info_.tile_width : std::optional<int>{};
}

std::optional<int> CodeStream::num_tiles_y() const {
    auto& info = getImageInfo();
    if (info.struct_next != &tile_geometry_info_)
        return std::nullopt;
    return tile_geometry_info_.num_tiles_y > 0 ? tile_geometry_info_.num_tiles_y : std::optional<int>{};
}

std::optional<int> CodeStream::num_tiles_x() const {
    auto& info = getImageInfo();
    if (info.struct_next != &tile_geometry_info_)
        return std::nullopt;
    return tile_geometry_info_.num_tiles_x > 0 ? tile_geometry_info_.num_tiles_x : std::optional<int>{};
}

std::optional<int> CodeStream::tile_offset_x() const {
    auto& info = getImageInfo();
    if (info.struct_next != &tile_geometry_info_)
        return std::nullopt;
    return tile_geometry_info_.tile_width > 0 ? tile_geometry_info_.tile_offset_x : std::optional<int>{};
}

std::optional<int> CodeStream::tile_offset_y() const {
    auto& info = getImageInfo();
    if (info.struct_next != &tile_geometry_info_)
        return std::nullopt;
    return tile_geometry_info_.tile_height > 0 ? tile_geometry_info_.tile_offset_y : std::optional<int>{};
}

std::optional<CodeStreamView> CodeStream::view() const
{
    return view_;
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
    return resolve_precision(info.plane_info[0].precision, info.plane_info[0].sample_type);
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

nvimgcodecChromaSubsampling_t CodeStream::getChromaSubsampling() const
{
    auto& info = getImageInfo();
    return info.chroma_subsampling;
}

std::optional<size_t> CodeStream::ifd_offset() const
{
    auto& info = getCodeStreamInfo();
    const auto* tiff_ext = findInfoTiffExt(&info);
    if (tiff_ext && tiff_ext->ifd_offset != 0) {
        return tiff_ext->ifd_offset;
    }
    return std::nullopt;
}

std::optional<size_t> CodeStream::next_ifd_offset() const
{
    auto& info = getCodeStreamInfo();
    const auto* tiff_ext = findInfoTiffExt(&info);
    if (tiff_ext && tiff_ext->next_ifd_offset != 0) {
        return tiff_ext->next_ifd_offset;
    }
    return std::nullopt;
}

std::vector<size_t> CodeStream::subifd_offsets() const
{
    auto& info = getCodeStreamInfo();
    const auto* tiff_ext = findInfoTiffExt(&info);
    if (tiff_ext && tiff_ext->subifd_count > 0) {
        return std::vector<size_t>(tiff_ext->subifd_offsets,
                                    tiff_ext->subifd_offsets + tiff_ext->subifd_count);
    }
    return {};
}

CodeStream* CodeStream::getSubCodeStream(const CodeStreamView& code_stream_view)
{
    nvimgcodecCodeStream_t sub_code_stream{nullptr};
    auto effective_view = code_stream_view;
    {
        if (view_) {
            const auto& parent_view = view_->impl_;
            if (parent_view.region.ndim != 0) {
                throw std::runtime_error("Cannot create a sub code stream with nested regions. This is not supported.");
            }
            if (code_stream_view.impl_.image_idx != 0) {
                throw std::runtime_error("Cannot apply nonzero image_idx to a sub code stream.");
            }
            if (effective_view.impl_.bitstream_offset == 0) {
                if (parent_view.bitstream_offset != 0) {
                    effective_view.impl_.bitstream_offset = parent_view.bitstream_offset;
                } else if (effective_view.impl_.image_idx == 0) {
                    effective_view.impl_.image_idx = parent_view.image_idx;
                }
            }
        }
        py::gil_scoped_release release;
        CHECK_NVIMGCODEC(nvimgcodecCodeStreamGetSubCodeStream(code_stream_, &sub_code_stream, &code_stream_view.impl_));
    }

    return new CodeStream(instance_, logger_, sub_code_stream, effective_view);
}

void CodeStream::exportToPython(py::module& m, nvimgcodecInstance_t instance, ILogger* logger)
{
    // clang-format off
    py::class_<CodeStream>(m, "CodeStream",
        R"pbdoc(
        Class representing a coded stream of image data.

        This class provides access to image informations such as dimensions, codec,
        and tiling details. It supports initialization from bytes, numpy arrays, or file path.

        The codec is detected by registered parsers inspecting the encoded contents. A file
        path's extension is not used to select the codec, parser, or decoder.
        )pbdoc",
        py::buffer_protocol())
        .def(py::init([instance, logger](py::bytes bytes, std::optional<size_t> bitstream_offset) {
                return new CodeStream(instance, logger, bytes, bitstream_offset);
            }),
            "bytes"_a, "bitstream_offset"_a = py::none(), py::keep_alive<1, 2>(),
            R"pbdoc(
            Initialize a CodeStream using bytes as input.

            Note: image_idx and region are not supported here. Use get_sub_code_stream() to select
            a specific image or region after creation.

            Args:
                bytes: The byte data representing the encoded stream.

                bitstream_offset: For TIFF files, nonzero byte offset of one IFD to select.
                    Defaults to None (parse from file header); 0 is equivalent to None.
                    A nonzero bitstream_offset creates a one-image view and num_images
                    reports 1. Query the root CodeStream for the total root page count.

            )pbdoc")
        .def(py::init([instance, logger](py::array_t<uint8_t> arr, std::optional<size_t> bitstream_offset) {
                return new CodeStream(instance, logger, arr, bitstream_offset);
            }),
            "array"_a, "bitstream_offset"_a = py::none(), py::keep_alive<1, 2>(),
            R"pbdoc(
            Initialize a CodeStream using a numpy array of uint8 as input.

            Note: image_idx and region are not supported here. Use get_sub_code_stream() to select
            a specific image or region after creation.

            Args:
                array: The numpy array containing the encoded stream.

                bitstream_offset: For TIFF files, nonzero byte offset of one IFD to select.
                    Defaults to None (parse from file header); 0 is equivalent to None.
                    A nonzero bitstream_offset creates a one-image view and num_images
                    reports 1. Query the root CodeStream for the total root page count.

            )pbdoc")
        .def(py::init([instance, logger](const std::filesystem::path& filename,
                                          std::optional<size_t> bitstream_offset) {
                return new CodeStream(instance, logger, filename, bitstream_offset);
            }),
            "filename"_a, "bitstream_offset"_a = py::none(),
            R"pbdoc(
            Initialize a CodeStream using a file path as input.

            Note: image_idx and region are not supported here. Use get_sub_code_stream() to select
            a specific image or region after creation.

            Args:
                filename: The file path to the encoded stream data.

                bitstream_offset: For TIFF files, nonzero byte offset of one IFD to select.
                    Use offsets from next_ifd_offset or subifd_offsets. Defaults to None
                    (parse from file header); 0 is equivalent to None.
                    A nonzero bitstream_offset creates a one-image view and num_images
                    reports 1. Query the root CodeStream for the total root page count.

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
        .def("get_sub_code_stream", [](CodeStream& self, size_t image_idx, std::optional<Region> region,
                                         std::optional<size_t> bitstream_offset) -> py::object {
                CodeStream* sub = self.getSubCodeStream(CodeStreamView(image_idx, region, bitstream_offset));
                py::object result = py::cast(sub, py::return_value_policy::take_ownership);
                /* Keep this (self) CodeStream alive as long as the newly created and returned
                 * sub CodeStream is alive. This is required as this CodeStream may keep alive
                 * python object with data (like bytes).
                 *
                 * Applied manually rather than via py::keep_alive<0, 1>() because this is one
                 * of two overloads: pybind11 3.1.0 runs the keep_alive postcall unconditionally,
                 * including when THIS overload's argument binding fails and dispatch falls
                 * through to the CodeStreamView overload below, which segfaults dereferencing
                 * the PYBIND11_TRY_NEXT_OVERLOAD sentinel. Upstream fix, not yet released:
                 * https://github.com/pybind/pybind11/pull/6154. Doing this after a real result
                 * exists avoids that path regardless of version.
                */
                py::detail::keep_alive_impl(result, py::cast(self));
                return result;
            },
            "image_idx"_a = 0, "region"_a = std::nullopt,
            "bitstream_offset"_a = py::none(),
            R"pbdoc(
            Get a sub code stream for a specific image index, optional region, and TIFF-specific parameters.

            Args:
                image_idx: Index of the image in the code stream. Defaults to 0.
                
                region: Optional region of interest within the image.
                
                bitstream_offset: For TIFF files, nonzero byte offset of one IFD to select.
                    Use offsets from next_ifd_offset or subifd_offsets. Defaults to None;
                    0 is equivalent to None. A nonzero bitstream_offset cannot be
                    combined with a nonzero image_idx. A nonzero bitstream_offset creates
                    a one-image view and num_images reports 1. Query the root CodeStream
                    for the total root page count.
                
            Returns:
                A new CodeStream object representing the sub code stream.
            )pbdoc")
        .def("get_sub_code_stream", [](CodeStream& self, const CodeStreamView& view) -> py::object {
                CodeStream* sub = self.getSubCodeStream(view);
                py::object result = py::cast(sub, py::return_value_policy::take_ownership);
                /* See the comment on the other get_sub_code_stream overload above: this keep_alive
                 * is applied manually, not via py::keep_alive<0, 1>(), to avoid a pybind11 3.1.0
                 * postcall bug affecting overloaded defs.
                */
                py::detail::keep_alive_impl(result, py::cast(self));
                return result;
            },
            "view"_a,
            R"pbdoc(
            Get a sub code stream using a CodeStreamView object.

            Args:
                view: A CodeStreamView object specifying the image index, optional region,
                    or TIFF bitstream offset.

            Returns:
                A new CodeStream object representing the sub code stream.
            )pbdoc")
        .def_property_readonly("num_images", &CodeStream::num_images, 
            R"pbdoc(
            The number of images in the code stream.

            For TIFF root streams, this counts the full root IFD chain on first
            access and caches the result. TIFF substreams created with image_idx
            or bitstream_offset are one-image views and report 1. Query the root
            CodeStream when the total root page count is needed.
            )pbdoc")
        .def_property_readonly("ifd_offset", &CodeStream::ifd_offset,
            R"pbdoc(
            For TIFF files: the byte offset of the selected image's IFD.

            Returns None for non-TIFF files or when the offset is not available.

            Note: on a root CodeStream this triggers a one-time parse of the full root chain;
            use ``get_sub_code_stream(0).ifd_offset`` for lazy single-image access.
            )pbdoc")
        .def_property_readonly("next_ifd_offset", &CodeStream::next_ifd_offset,
            R"pbdoc(
            For TIFF files: the byte offset of the next sibling IFD.

            Returns None if there is no next sibling IFD or the file format is not TIFF.

            Note: on a root CodeStream this triggers a one-time parse of the full root chain;
            use ``get_sub_code_stream(0).next_ifd_offset`` for lazy single-image access.
            )pbdoc")
        .def_property_readonly("subifd_offsets", &CodeStream::subifd_offsets,
            R"pbdoc(
            For TIFF files: list of SubIFD byte offsets for the current image (Tag 330).

            SubIFDs typically contain reduced-resolution versions (thumbnails) or other
            related images. Use these offsets with the bitstream_offset parameter to
            create a CodeStream for the SubIFD.

            Returns an empty list if no SubIFDs exist or if the format is not TIFF.

            Note: on a root CodeStream this triggers a one-time parse of the full root chain;
            use ``get_sub_code_stream(0)`` for lazy single-image access.

            Example::

                cs = nvimgcodec.CodeStream(path)
                sub = cs.get_sub_code_stream(image_idx=0)
                for offset in sub.subifd_offsets:
                    subifd_cs = cs.get_sub_code_stream(bitstream_offset=offset)
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
            Number of significant bits per sample as declared by the encoded bitstream.

            For formats that record a precision below the storage type's bitdepth (e.g. a
            12-bit JPEG 2000 image stored as uint16), this returns the bitstream's declared
            precision. When the bitstream does not encode a separate precision, this returns
            the full bitdepth of the sample data type (8 for ``uint8``, 16 for ``uint16``, etc.).
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
        .def_property_readonly("chroma_subsampling", &CodeStream::getChromaSubsampling,
            R"pbdoc(
            The chroma subsampling of the encoded image.
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
