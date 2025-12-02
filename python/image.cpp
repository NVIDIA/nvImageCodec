/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "image.h"

#include <iostream>
#include <stdexcept>
#include <cstddef>
#include <optional>
#include <string>

#include <dlpack/dlpack.h>

#include <ilogger.h>
#include <log.h>

#include <imgproc/stream_device.h>
#include <imgproc/device_guard.h>
#include <imgproc/device_buffer.h>
#include <imgproc/pinned_buffer.h>
#include "imgproc/type_utils.h"
#include "dlpack_utils.h"
#include "error_handling.h"
#include "type_utils.h"
#include "sample_format.h"

namespace nvimgcodec {

Image::Image(nvimgcodecInstance_t instance, ILogger* logger, nvimgcodecImageInfo_t* image_info)
    : instance_(instance)
    , logger_(logger)
{
    py::gil_scoped_release release;
    initBuffer(image_info);

    nvimgcodecImage_t image = nullptr;
    CHECK_NVIMGCODEC(nvimgcodecImageCreate(instance, &image, image_info));
    image_ = std::shared_ptr<std::remove_pointer<nvimgcodecImage_t>::type>(
        image, [](nvimgcodecImage_t image) { nvimgcodecImageDestroy(image); });
    
    // Pass the variant buffer to DLPackTensor to keep it alive
    dlpack_tensor_ = std::make_shared<DLPackTensor>(logger_, *image_info, img_buffer_);
}

void Image::initBuffer(nvimgcodecImageInfo_t* image_info)
{
    if (image_info->buffer == nullptr) {
        if (image_info->buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
            initDeviceBuffer(image_info);
        } else if (image_info->buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST) {
            initHostBuffer(image_info);
        } else {
            throw std::runtime_error("Unsupported buffer type.");
        }
    }
}

void Image::initDeviceBuffer(nvimgcodecImageInfo_t* image_info)
{
    // Create shared_ptr<DeviceBuffer> and store in variant
    auto device_buffer = std::make_shared<DeviceBuffer>();
    device_buffer->resize(GetBufferSize(*image_info), image_info->cuda_stream);
    
    // Store in the variant
    img_buffer_ = device_buffer;
    
    // Set the buffer pointer for the image info
    image_info->buffer = static_cast<unsigned char*>(device_buffer->data);
}

void Image::initHostBuffer(nvimgcodecImageInfo_t* image_info)
{
    // Create shared_ptr<PinnedBuffer> and store in variant
    auto pinned_buffer = std::make_shared<PinnedBuffer>();
    pinned_buffer->resize(GetBufferSize(*image_info), image_info->cuda_stream);
    
    // Store in the variant
    img_buffer_ = pinned_buffer;
    
    // Set the buffer pointer for the image info
    image_info->buffer = static_cast<unsigned char*>(pinned_buffer->data);
}

void Image::initImageInfoFromDLPack(nvimgcodecImageInfo_t* image_info, py::capsule cap)
{
    if (auto* tensor = static_cast<DLManagedTensor*>(cap.get_pointer())) {
        check_cuda_buffer(tensor->dl_tensor.data);
        dlpack_tensor_ = std::make_shared<DLPackTensor>(logger_, tensor);
        // signal that producer don't have to call tensor's deleter, consumer will do it instead
        cap.set_name("used_dltensor");
        dlpack_tensor_->getImageInfo(image_info);
    } else {
        throw std::runtime_error("Unsupported dlpack PyCapsule object.");
    }
}

namespace {

// Helper function to get the minimum number of channels required for a sample format
int getMinChannelsForSampleFormat(nvimgcodecSampleFormat_t sample_format) {
    switch (sample_format) {
        case NVIMGCODEC_SAMPLEFORMAT_P_Y:
        case NVIMGCODEC_SAMPLEFORMAT_I_Y:
            return 1;
        case NVIMGCODEC_SAMPLEFORMAT_P_YA:
        case NVIMGCODEC_SAMPLEFORMAT_I_YA:
            return 2;
        case NVIMGCODEC_SAMPLEFORMAT_P_RGB:
        case NVIMGCODEC_SAMPLEFORMAT_I_RGB:
        case NVIMGCODEC_SAMPLEFORMAT_P_BGR:
        case NVIMGCODEC_SAMPLEFORMAT_I_BGR:
        case NVIMGCODEC_SAMPLEFORMAT_P_YUV:
        case NVIMGCODEC_SAMPLEFORMAT_I_YUV:
            return 3;
        case NVIMGCODEC_SAMPLEFORMAT_P_RGBA:
        case NVIMGCODEC_SAMPLEFORMAT_I_RGBA:
        case NVIMGCODEC_SAMPLEFORMAT_P_CMYK:
        case NVIMGCODEC_SAMPLEFORMAT_I_CMYK:
        case NVIMGCODEC_SAMPLEFORMAT_P_YCCK:
        case NVIMGCODEC_SAMPLEFORMAT_I_YCCK:
            return 4;
        case NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED:
        case NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED:
        case NVIMGCODEC_SAMPLEFORMAT_UNKNOWN:
        default:
            return -1; // -1 means any number is acceptable
    }
}

// Validate that the sample format is compatible with the number of channels
void validateSampleFormatChannels(nvimgcodecSampleFormat_t sample_format, int num_channels) {
    int min_required_channels = getMinChannelsForSampleFormat(sample_format);
    
    // If min_required_channels is -1, any number of channels is acceptable
    if (min_required_channels == -1) {
        return;
    }
    
    // Check if we have at least the minimum required channels
    if (num_channels < min_required_channels) {
        throw std::invalid_argument("Invalid sample_format for the number of channels. Sample format requires at least " + 
                                   std::to_string(min_required_channels) + " channel(s), but image has only " + 
                                   std::to_string(num_channels) + " channel(s).");
    }
}

} // anonymous namespace

void Image::initImageInfoFromInterfaceDict(const py::dict& iface, nvimgcodecImageInfo_t* image_info,
    std::optional<nvimgcodecSampleFormat_t> sample_format, std::optional<nvimgcodecColorSpec_t> color_spec)
{

    bool is_interleaved = true; //TODO detect interleaved if we have HWC layout

    std::vector<long> vshape;
    py::tuple shape = iface["shape"].cast<py::tuple>();
    for (auto& o : shape) {
        vshape.push_back(o.cast<long>());
    }
    if (vshape.size() < 2) {
        throw std::runtime_error("Unexpected number of dimensions. At least 2 dimensions are expected.");
    }
    if (vshape.size() > 3) {
        throw std::runtime_error("Unexpected number of dimensions. At most 3 dimensions are expected.");
    }
    if (!is_padding_correct(iface, is_interleaved)) {
        throw std::runtime_error("Unexpected array style. Padding is only allowed for rows. Other dimensions should have contiguous strides.");
    }
    std::vector<int> vstrides;
    if (iface.contains("strides")) {
        py::object strides = iface["strides"];
        if (!strides.is(py::none())) {
            strides = strides.cast<py::tuple>();
            for (auto& o : strides) {
                vstrides.push_back(o.cast<int>());
            }
        }
    }

    if (is_interleaved) {
        image_info->num_planes = 1;
        image_info->plane_info[0].height = vshape[0];
        image_info->plane_info[0].width = vshape[1];
        image_info->plane_info[0].num_channels = vshape.size() == 3 ? vshape[2] : 1;
    } else {
        image_info->num_planes = vshape[0];
        image_info->plane_info[0].height = vshape[1];
        image_info->plane_info[0].width = vshape[2];
        image_info->plane_info[0].num_channels = 1;
    }

    std::string typestr = iface["typestr"].cast<std::string>();
    auto sample_type = type_from_format_str(typestr);

    int bytes_per_element = sample_type_to_bytes_per_element(sample_type);

    image_info->chroma_subsampling = NVIMGCODEC_SAMPLING_444;

    // Validate user-provided sample_format against number of channels
    int num_channels = image_info->plane_info[0].num_channels;
    if (sample_format.has_value()) {
        validateSampleFormatChannels(sample_format.value(), num_channels);
    }

    // Set sample_format based on number of channels (if not provided by user)
    if (num_channels == 0) {
        // 0 channels  not allowed
        throw std::runtime_error("Unexpected number of channels. At least 1 channel is expected.");
    } else if (num_channels == 1) {
        // Single channel (grayscale) - default to interleaved
        image_info->color_spec = color_spec.value_or(NVIMGCODEC_COLORSPEC_GRAY);
        image_info->chroma_subsampling = NVIMGCODEC_SAMPLING_GRAY;
        image_info->sample_format = sample_format.value_or(NVIMGCODEC_SAMPLEFORMAT_I_Y);
    } else if (num_channels == 2) {
        // 2 channels (grayscale with alpha) - default to interleaved
        image_info->color_spec = color_spec.value_or(NVIMGCODEC_COLORSPEC_GRAY);
        image_info->chroma_subsampling = NVIMGCODEC_SAMPLING_GRAY;
        image_info->sample_format = sample_format.value_or(NVIMGCODEC_SAMPLEFORMAT_I_YA);
    } else if (num_channels == 3) {
        // 3 channels (typically RGB)
        image_info->color_spec = color_spec.value_or(NVIMGCODEC_COLORSPEC_SRGB);
        image_info->sample_format = sample_format.value_or(is_interleaved ? NVIMGCODEC_SAMPLEFORMAT_I_RGB : NVIMGCODEC_SAMPLEFORMAT_P_RGB);
    } else if (num_channels == 4) {
        // 4 channels (typically RGBA)
        image_info->color_spec = color_spec.value_or(NVIMGCODEC_COLORSPEC_SRGB);
        image_info->sample_format = sample_format.value_or(NVIMGCODEC_SAMPLEFORMAT_I_RGBA);
    } else {
        // More than 4 channels
        image_info->color_spec = color_spec.value_or(NVIMGCODEC_COLORSPEC_UNKNOWN);
        image_info->sample_format = sample_format.value_or(NVIMGCODEC_SAMPLEFORMAT_UNKNOWN);
    }

    int pitch_in_bytes = vstrides.size() > 1 ? (is_interleaved ? vstrides[0] : vstrides[1])
                                             : image_info->plane_info[0].width * image_info->plane_info[0].num_channels * bytes_per_element;
    for (size_t c = 0; c < image_info->num_planes; c++) {
        image_info->plane_info[c].width = image_info->plane_info[0].width;
        image_info->plane_info[c].height = image_info->plane_info[0].height;
        image_info->plane_info[c].row_stride = pitch_in_bytes;
        image_info->plane_info[c].sample_type = sample_type;
        image_info->plane_info[c].num_channels = image_info->plane_info[0].num_channels;
    }
    py::tuple tdata = iface["data"].cast<py::tuple>();
    void* buffer = PyLong_AsVoidPtr(tdata[0].ptr());
    image_info->buffer = buffer;
}

Image::Image(nvimgcodecInstance_t instance, ILogger* logger, PyObject* o, intptr_t cuda_stream,
             std::optional<nvimgcodecSampleFormat_t> sample_format,
             std::optional<nvimgcodecColorSpec_t> color_spec)
    : instance_(instance)
    , logger_(logger)
    , img_buffer_{nvimgcodecImageInfo_t{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0}}
{
    if (!o) {
        throw std::runtime_error("Object cannot be None");
    }
    py::object tmp = py::reinterpret_borrow<py::object>(o);
    auto& image_info = std::get<nvimgcodecImageInfo_t>(img_buffer_);
    image_info.cuda_stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    if (py::isinstance<py::capsule>(tmp)) {
        py::capsule cap = tmp.cast<py::capsule>();
        initImageInfoFromDLPack(&image_info, cap);
    } else if (hasattr(tmp, "__cuda_array_interface__")) {
        py::dict iface = tmp.attr("__cuda_array_interface__").cast<py::dict>();

        if (!iface.contains("shape") || !iface.contains("typestr") || !iface.contains("data") || !iface.contains("version")) {
            throw std::runtime_error("Unsupported __cuda_array_interface__ with missing field(s)");
        }

        int version = iface["version"].cast<int>();
        if (version < 2) {
            throw std::runtime_error("Unsupported __cuda_array_interface__ with version < 2");
        }
        initImageInfoFromInterfaceDict(iface, &image_info, sample_format, color_spec);
        std::optional<intptr_t> stream =
            version >= 3 && iface.contains("stream") ? iface["stream"].cast<std::optional<intptr_t>>() : std::optional<intptr_t>();

        if (stream.has_value()) {
            if (*stream == 0) {
                throw std::runtime_error("Invalid for stream to be 0");
            } else {
                image_info.cuda_stream = reinterpret_cast<cudaStream_t>(*stream);
            }
        }
        check_cuda_buffer(image_info.buffer);
        image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        
        dlpack_tensor_ = std::make_shared<DLPackTensor>(logger_, image_info);
    } else if (hasattr(tmp, "__array_interface__")) {
        py::dict iface = tmp.attr("__array_interface__").cast<py::dict>();

        if (!iface.contains("shape") || !iface.contains("typestr") || !iface.contains("data") || !iface.contains("version")) {
            throw std::runtime_error("Unsupported __array_interface__ with missing field(s)");
        }

        int version = iface["version"].cast<int>();
        if (version < 2) {
            throw std::runtime_error("Unsupported __array_interface__ with version < 2");
        }

        initImageInfoFromInterfaceDict(iface, &image_info, sample_format, color_spec);
        image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        
        dlpack_tensor_ = std::make_shared<DLPackTensor>(logger_, image_info);
    } else if (hasattr(tmp, "__dlpack__")) {
        // Quickly check if we support the device
        if (hasattr(tmp, "__dlpack_device__")) {
            py::tuple dlpack_device = tmp.attr("__dlpack_device__")().cast<py::tuple>();
            auto dev_type = static_cast<DLDeviceType>(dlpack_device[0].cast<int>());
            if (!is_cuda_accessible(dev_type)) {
                throw std::runtime_error("Unsupported device in DLTensor. Only CUDA-accessible memory buffers can be wrapped");
            }
        }
        py::object py_cuda_stream = cuda_stream ? py::int_((intptr_t)(cuda_stream)) : py::int_(1);
        py::capsule cap = tmp.attr("__dlpack__")("stream"_a = py_cuda_stream).cast<py::capsule>();
        initImageInfoFromDLPack(&image_info, cap);
    } else {
        throw std::runtime_error("Object does not support neither __cuda_array_interface__ nor __dlpack__");
    }

    py::gil_scoped_release release;
    nvimgcodecImage_t image = nullptr;
    CHECK_NVIMGCODEC(nvimgcodecImageCreate(instance, &image, &image_info));
    image_ = std::shared_ptr<std::remove_pointer<nvimgcodecImage_t>::type>(
        image, [](nvimgcodecImage_t image) { nvimgcodecImageDestroy(image); });
}

void Image::initInterfaceDictFromImageInfo(py::dict* d) const
{
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    {
        py::gil_scoped_release release;
        nvimgcodecImageGetImageInfo(image_.get(), &image_info);
    }

    const auto& plane_info = image_info.plane_info[0];
    size_t bytes_per_sample = plane_info.sample_type >> 11;
    size_t row_size = static_cast<size_t>(plane_info.width) * plane_info.num_channels * bytes_per_sample;
    bool is_continuous = plane_info.row_stride == row_size;

    (*d)["shape"] = shape();
    (*d)["strides"] = is_continuous? py::none() : py::object(strides());
    (*d)["typestr"] = format_str_from_type(plane_info.sample_type);
    (*d)["data"] = py::make_tuple(py::reinterpret_borrow<py::object>(PyLong_FromVoidPtr(image_info.buffer)), false);
    (*d)["version"] = 3;
}

int Image::getWidth() const
{
    py::gil_scoped_release release;
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    nvimgcodecImageGetImageInfo(image_.get(), &image_info);
    return image_info.plane_info[0].width;
}
int Image::getHeight() const
{
    py::gil_scoped_release release;
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    nvimgcodecImageGetImageInfo(image_.get(), &image_info);
    return image_info.plane_info[0].height;
}

int Image::getNdim() const
{
    //Shape has always 3 dimensions either WHC (interleaved) or CHW (planar)
    return 3;
}

py::dict Image::array_interface() const
{
    py::dict array_interface;
    try {
        initInterfaceDictFromImageInfo(&array_interface);
    } catch (...) {
        throw std::runtime_error("Unable to initialize __array_interface__");
    }
    return array_interface;
}

py::dict Image::cuda_interface() const
{
    py::dict cuda_array_interface;
    try {
        initInterfaceDictFromImageInfo(&cuda_array_interface);
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        {
            py::gil_scoped_release release;
            nvimgcodecImageGetImageInfo(image_.get(), &image_info);
        }
        py::object stream = image_info.cuda_stream ? py::int_((intptr_t)(image_info.cuda_stream)) : py::int_(1);
        cuda_array_interface["stream"] = stream;
    } catch (...) {
        throw std::runtime_error("Unable to initialize __cuda_array_interface__");
    }
    return cuda_array_interface;
}

py::tuple Image::shape() const
{
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    {
        py::gil_scoped_release release;
        nvimgcodecImageGetImageInfo(image_.get(), &image_info);
    }
    bool is_interleaved = is_sample_format_interleaved(image_info.sample_format) || image_info.num_planes == 1;
    py::tuple shape_tuple =
        is_interleaved
            ? py::make_tuple(image_info.plane_info[0].height, image_info.plane_info[0].width, image_info.plane_info[0].num_channels)
            : py::make_tuple(image_info.num_planes, image_info.plane_info[0].height, image_info.plane_info[0].width);
    return shape_tuple;
}

py::tuple Image::strides() const
{
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    {
        py::gil_scoped_release release;
        nvimgcodecImageGetImageInfo(image_.get(), &image_info);
    }
    int bytes_per_element = sample_type_to_bytes_per_element(image_info.plane_info[0].sample_type);
    bool is_interleaved = is_sample_format_interleaved(image_info.sample_format) || image_info.num_planes == 1;
    py::tuple strides_tuple = is_interleaved ? py::make_tuple(image_info.plane_info[0].row_stride,
                                                   image_info.plane_info[0].num_channels * bytes_per_element, bytes_per_element)
                                             : py::make_tuple(image_info.plane_info[0].row_stride * image_info.plane_info[0].height,
                                                   image_info.plane_info[0].row_stride, bytes_per_element);
    return strides_tuple;
}

py::object Image::dtype() const
{
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    {
        py::gil_scoped_release release;
        nvimgcodecImageGetImageInfo(image_.get(), &image_info);
    }
    std::string format = format_str_from_type(image_info.plane_info[0].sample_type);
    return py::dtype(format);
}

int Image::precision() const
{
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    {
        py::gil_scoped_release release;
        nvimgcodecImageGetImageInfo(image_.get(), &image_info);
    }
    return image_info.plane_info[0].precision;
}

nvimgcodecSampleFormat_t Image::getSampleFormat() const
{
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    {
        py::gil_scoped_release release;
        nvimgcodecImageGetImageInfo(image_.get(), &image_info);
    }
    return image_info.sample_format;
}

nvimgcodecColorSpec_t Image::getColorSpec() const
{
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    {
        py::gil_scoped_release release;
        nvimgcodecImageGetImageInfo(image_.get(), &image_info);
    }
    return image_info.color_spec;
}

nvimgcodecImageBufferKind_t Image::getBufferKind() const
{
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    {
        py::gil_scoped_release release;
        nvimgcodecImageGetImageInfo(image_.get(), &image_info);
    }
    return image_info.buffer_kind;
}

size_t Image::size() const
{
    if (std::holds_alternative<std::shared_ptr<DeviceBuffer>>(img_buffer_)) {
        auto device_buffer = std::get<std::shared_ptr<DeviceBuffer>>(img_buffer_);
        if (device_buffer) {
            return device_buffer->size;
        }
        return 0;
    } else if (std::holds_alternative<std::shared_ptr<PinnedBuffer>>(img_buffer_)) {
        auto pinned_buffer = std::get<std::shared_ptr<PinnedBuffer>>(img_buffer_);
        if (pinned_buffer) {
            return pinned_buffer->size;
        }
        return 0;
    } else {
        assert(std::holds_alternative<nvimgcodecImageInfo_t>(img_buffer_));
        // For externally managed buffers, get size from image info
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        {
            py::gil_scoped_release release;
            CHECK_NVIMGCODEC(nvimgcodecImageGetImageInfo(image_.get(), &image_info));
        }
        return GetImageSize(image_info);
    }
}

size_t Image::capacity() const
{
    if (std::holds_alternative<std::shared_ptr<DeviceBuffer>>(img_buffer_)) {
        auto device_buffer = std::get<std::shared_ptr<DeviceBuffer>>(img_buffer_);
        if (device_buffer) {
            return device_buffer->capacity;
        }
        return 0;
    } else if (std::holds_alternative<std::shared_ptr<PinnedBuffer>>(img_buffer_)) {
        auto pinned_buffer = std::get<std::shared_ptr<PinnedBuffer>>(img_buffer_);
        if (pinned_buffer) {
            return pinned_buffer->capacity;
        }
        return 0;
    } else {
        assert(std::holds_alternative<nvimgcodecImageInfo_t>(img_buffer_));
        auto original_image_info = std::get<nvimgcodecImageInfo_t>(img_buffer_);
        return GetImageSize(original_image_info);
    }
}

nvimgcodecImage_t Image::getNvImgCdcsImage() const
{
    return image_.get();
}

inline bool Image::hasInternallyManagedBuffer() const
{
    return !std::holds_alternative<nvimgcodecImageInfo_t>(img_buffer_);
}

py::capsule Image::dlpack(py::object stream_obj) const
{
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    {
        py::gil_scoped_release release;
        nvimgcodecImageGetImageInfo(image_.get(), &image_info);
    }
    std::optional<intptr_t> stream = stream_obj.cast<std::optional<intptr_t>>();
    intptr_t consumer_stream = stream.has_value() ? *stream : 0;

    py::capsule cap = dlpack_tensor_->getPyCapsule(consumer_stream, image_info.cuda_stream);
    if (std::string(cap.name()) != "dltensor") {
        throw std::runtime_error(
            "Could not get DLTensor capsules. It can be consumed only once, so you might have already constructed a tensor from it once.");
    }

    return cap;
}

const py::tuple Image::getDlpackDevice() const
{
    return py::make_tuple(
        py::int_(static_cast<int>((*dlpack_tensor_)->device.device_type)), py::int_(static_cast<int>((*dlpack_tensor_)->device.device_id)));
}

py::object Image::cpu()
{
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    {
        py::gil_scoped_release release;
        nvimgcodecImageGetImageInfo(image_.get(), &image_info);
    }
    if (image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
        nvimgcodecImageInfo_t cpu_image_info(image_info);
        cpu_image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        cpu_image_info.buffer = nullptr;
        for (unsigned int c = 0; c < cpu_image_info.num_planes; ++c) {
            auto& plane_info = cpu_image_info.plane_info[c];
            size_t bpp = TypeSize(plane_info.sample_type);
            plane_info.row_stride = static_cast<size_t>(plane_info.width) * bpp * plane_info.num_channels;
        }
        assert(GetBufferSize(cpu_image_info) == GetImageSize(cpu_image_info));

        auto image = new Image(instance_, logger_, &cpu_image_info);
        {
            py::gil_scoped_release release;
            if (cpu_image_info.plane_info[0].row_stride == image_info.plane_info[0].row_stride) {
                // new cpu_image_info is continuous, so original image also must be continuous
                // we can just memcpy whole image
                assert(GetBufferSize(cpu_image_info) == GetBufferSize(image_info));
                CHECK_CUDA(cudaMemcpyAsync(
                    cpu_image_info.buffer, image_info.buffer, GetBufferSize(image_info),
                    cudaMemcpyDeviceToHost, image_info.cuda_stream
                ));
            } else {
                CHECK_CUDA(cudaMemcpy2DAsync(
                    cpu_image_info.buffer, cpu_image_info.plane_info[0].row_stride,
                    image_info.buffer, image_info.plane_info[0].row_stride,
                    cpu_image_info.plane_info[0].row_stride, cpu_image_info.plane_info[0].height,
                    cudaMemcpyHostToDevice, image_info.cuda_stream
                ));
            }
            CHECK_CUDA(cudaStreamSynchronize(image_info.cuda_stream));
        }
        return py::cast(image,  py::return_value_policy::take_ownership);
    } else if (image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST) {
        return py::cast(this);
    } else {
        return py::none();
    }
}

py::object Image::cuda(bool synchronize)
{
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    {
        py::gil_scoped_release release;
        nvimgcodecImageGetImageInfo(image_.get(), &image_info);
    }
    if (image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST) {
        nvimgcodecImageInfo_t cuda_image_info(image_info);
        cuda_image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        cuda_image_info.buffer = nullptr;
        for (unsigned int c = 0; c < cuda_image_info.num_planes; ++c) {
            auto& plane_info = cuda_image_info.plane_info[c];
            size_t bpp = TypeSize(plane_info.sample_type);
            plane_info.row_stride = static_cast<size_t>(plane_info.width) * bpp * plane_info.num_channels;
        }
        assert(GetBufferSize(cuda_image_info) == GetImageSize(cuda_image_info));

        auto image = new Image(instance_, logger_, &cuda_image_info);
        {
            py::gil_scoped_release release;
            if (cuda_image_info.plane_info[0].row_stride == image_info.plane_info[0].row_stride) {
                // new cuda_image_info is continuous, so original image also must be continuous
                // we can just memcpy whole image
                assert(GetBufferSize(cuda_image_info) == GetBufferSize(image_info));
                CHECK_CUDA(cudaMemcpyAsync(
                    cuda_image_info.buffer, image_info.buffer, GetBufferSize(cuda_image_info),
                    cudaMemcpyHostToDevice, cuda_image_info.cuda_stream
                ));
            } else {
                CHECK_CUDA(cudaMemcpy2DAsync(
                    cuda_image_info.buffer, cuda_image_info.plane_info[0].row_stride,
                    image_info.buffer, image_info.plane_info[0].row_stride,
                    cuda_image_info.plane_info[0].row_stride, cuda_image_info.plane_info[0].height,
                    cudaMemcpyHostToDevice, cuda_image_info.cuda_stream
                ));
            }
            if (synchronize) {
                CHECK_CUDA(cudaStreamSynchronize(cuda_image_info.cuda_stream));
            }
        }
        return py::cast(image,  py::return_value_policy::take_ownership);
    } else if (image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
        return py::cast(this);
    } else {
        return py::none();
    }
}

void Image::reuse(nvimgcodecImageInfo_t* image_info)
{
    py::gil_scoped_release release;
    
    if (std::holds_alternative<std::shared_ptr<DeviceBuffer>>(img_buffer_)) {
        auto device_buffer = std::get<std::shared_ptr<DeviceBuffer>>(img_buffer_);
        device_buffer->resize(GetBufferSize(*image_info), image_info->cuda_stream);
        image_info->buffer = static_cast<unsigned char*>(device_buffer->data);
        image_info->buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
    } else if (std::holds_alternative<std::shared_ptr<PinnedBuffer>>(img_buffer_)) {
        auto pinned_buffer = std::get<std::shared_ptr<PinnedBuffer>>(img_buffer_);
        pinned_buffer->resize(GetBufferSize(*image_info), image_info->cuda_stream);
        image_info->buffer = static_cast<unsigned char*>(pinned_buffer->data);
        image_info->buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
    } else { // externally manager buffer
        assert(std::holds_alternative<nvimgcodecImageInfo_t>(img_buffer_));
        auto buffer_info = std::get<nvimgcodecImageInfo_t>(img_buffer_);
        if (buffer_info.num_planes != 1) {
            throw std::invalid_argument("Only single plane (interleaved format) reuse for external buffer is supported");
        }

        if (buffer_info.num_planes < image_info->num_planes) {
            throw std::invalid_argument("Number of planes in the buffer is not enough");
        }

        const auto& new_image_plane = image_info->plane_info[0];
        size_t new_image_row_size = (static_cast<size_t>(new_image_plane.width) *
            new_image_plane.num_channels *
            sample_type_to_bytes_per_element(new_image_plane.sample_type)
        );
        assert(new_image_row_size == new_image_plane.row_stride); // assuming that no row padding is needed for new image

        const auto& buffer_plane = buffer_info.plane_info[0];
        size_t buffer_row_size = (static_cast<size_t>(buffer_plane.width) *
            buffer_plane.num_channels *
            sample_type_to_bytes_per_element(buffer_plane.sample_type)
        );

        if (image_info->plane_info[0].height > buffer_info.plane_info[0].height ||
            new_image_row_size > buffer_row_size
        ) {
             // check if buffer is not continuous
            if (buffer_row_size != buffer_plane.row_stride) {
                throw std::invalid_argument("Existing buffer is not continuous. Row size or height are too small to fit new image.");
            }

            // buffer is continuous, but lets check if its size is enough
            if (GetBufferSize(*image_info) > GetBufferSize(buffer_info)) {
                throw std::invalid_argument("Existing buffer is too small to fit new image");
            }

            // new image fits inside the buffer, but it is outside of bounds (height or row size)
            // so we wil keep row_stride of the new image
        } else {
            // image fits in existing buffer bounds
            // so lets just keep original row stride, as buffer may not be continuous.
            image_info->plane_info[0].row_stride = buffer_info.plane_info[0].row_stride;
        }

        if (buffer_info.cuda_stream != image_info->cuda_stream) {
            // TODO: could be done via event sync, but we should also add it to device_buffer.cpp then
            CHECK_CUDA(cudaStreamSynchronize(buffer_info.cuda_stream));
        }

        image_info->buffer = buffer_info.buffer;
        image_info->buffer_kind = buffer_info.buffer_kind;
    }

    nvimgcodecImage_t new_image = image_.get();
    CHECK_NVIMGCODEC(nvimgcodecImageCreate(instance_, &new_image, image_info));
    assert(new_image == image_.get());

    // Update DLPackTensor with new image info and buffer
    dlpack_tensor_ = std::make_shared<DLPackTensor>(logger_, *image_info, img_buffer_);
}

void Image::exportToPython(py::module& m)
{
    // clang-format off
    py::class_<Image>(m, "Image", 
            R"pbdoc(Class which wraps buffer with pixels. It can be decoded pixels or pixels to encode.

            At present, the image must always have a three-dimensional shape in the HWC layout (height, width, channels), 
            which is also known as the interleaved format, and be stored as a contiguous array in C-style, but rows can have additional padding.
            )pbdoc")
        .def_property_readonly("__array_interface__", &Image::array_interface,
            R"pbdoc(
            The array interchange interface compatible with Numba v0.39.0 or later (see 
            `CUDA Array Interface <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_ for details)
            )pbdoc")
        .def_property_readonly("__cuda_array_interface__", &Image::cuda_interface,
            R"pbdoc(
            The CUDA array interchange interface compatible with Numba v0.39.0 or later (see 
            `CUDA Array Interface <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_ for details)
            )pbdoc")
        .def_property_readonly("shape", &Image::shape,
            R"pbdoc(
            The shape of the image.
            )pbdoc")
        .def_property_readonly("strides", &Image::strides, 
            R"pbdoc(
            Strides of axes in bytes.
            )pbdoc")
        .def_property_readonly("width", &Image::getWidth,
            R"pbdoc(
            The width of the image in pixels.
            )pbdoc")
        .def_property_readonly("height", &Image::getHeight,
            R"pbdoc(
            The height of the image in pixels.
            )pbdoc")
        .def_property_readonly("ndim", &Image::getNdim,
            R"pbdoc(
            The number of dimensions in the image.
            )pbdoc")
        .def_property_readonly("dtype", &Image::dtype,
            R"pbdoc(
            The data type (dtype) of the image samples.
            )pbdoc")
        .def_property_readonly("precision", &Image::precision, 
            R"pbdoc(
            Maximum number of significant bits in data type. Value 0 means that precision is equal to data type bit depth.
            )pbdoc")
        .def_property_readonly("sample_format", &Image::getSampleFormat, 
            R"pbdoc(
            The sample format of the image indicating how color components are matched to channels and channels to planes.
            )pbdoc")
        .def_property_readonly("color_spec", &Image::getColorSpec, 
            R"pbdoc(
            Color specification of the image indicating how the color information in image samples should be interpreted.
            )pbdoc")
        .def_property_readonly("buffer_kind", &Image::getBufferKind, 
            R"pbdoc(
            Buffer kind in which image data is stored. This indicates whether the data is stored as strided device or host memory.
            )pbdoc")
        .def_property_readonly("size", &Image::size, 
            R"pbdoc(
            The size of the image buffer in bytes.
            )pbdoc")
        .def_property_readonly("capacity", &Image::capacity, 
            R"pbdoc(
            The capacity of the image buffer in bytes.
            )pbdoc")
        .def("__dlpack__", &Image::dlpack, "stream"_a = py::none(), "Export the image as a DLPack tensor")
        .def("__dlpack_device__", &Image::getDlpackDevice, "Get the device associated with the buffer")
        .def("to_dlpack", &Image::dlpack,
            R"pbdoc(
            Export the image with zero-copy conversion to a DLPack tensor. 
            
            Args:
                cuda_stream: An optional cudaStream_t represented as a Python integer, 
                upon which synchronization must take place in created Image.

            Returns:
                DLPack tensor which is encapsulated in a PyCapsule object.
            )pbdoc",
            "cuda_stream"_a = py::none())
        .def("cpu", &Image::cpu,
            R"pbdoc(
            Returns a copy of this image in CPU memory. If this image is already in CPU memory, 
            than no copy is performed and the original object is returned. 
            
            Returns:
                Image object with content in CPU memory or None if copy could not be done.
            )pbdoc")
        .def("cuda", &Image::cuda,
            R"pbdoc(
            Returns a copy of this image in device memory. If this image is already in device memory, 
            than no copy is performed and the original object is returned.  
            
            Args:
                synchronize: If True (by default) it blocks and waits for copy from host to device to be finished, 
                             else no synchronization is executed and further synchronization needs to be done using
                             cuda stream provided by e.g. \_\_cuda_array_interface\_\_. 

            Returns:
                Image object with content in device memory or None if copy could not be done.
            )pbdoc",
            "synchronize"_a = true);
    // clang-format on
}


} // namespace nvimgcodec
