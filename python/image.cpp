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
#include <imgproc/pageable_host_buffer.h>
#include <imgproc/pinned_buffer.h>
#include "imgproc/type_utils.h"
#include "device_utils.h"
#include "dlpack_utils.h"
#include "error_handling.h"
#include "type_utils.h"
#include "sample_format.h"
// Included after error_handling.h so CopyImage uses this TU's CHECK_CUDA
// (error_handling.h's std::runtime_error variant) instead of redefining it.
#include "imgproc/copy_image.h"

namespace nvimgcodec {

Image::Image(nvimgcodecInstance_t instance, ILogger* logger, nvimgcodecImageInfo_t* image_info, int device_id)
    : instance_(instance)
    , logger_(logger)
    , device_id_(resolve_device_id(device_id))
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
    DeviceGuard device_guard(device_id_);
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
    const size_t buffer_size = GetBufferSize(*image_info);
    void* data = nullptr;
    // CPU-only callers may not have libcuda available, so cudaMallocHost
    // (which calls cudaGetDeviceCount internally) cannot be used. Fall back
    // to a plain pageable host buffer in that case.
    if (device_id_ == NVIMGCODEC_DEVICE_CPU_ONLY) {
        auto buffer = std::make_shared<PageableHostBuffer>();
        buffer->resize(buffer_size);
        data = buffer->data;
        img_buffer_ = std::move(buffer);
    } else {
        auto buffer = std::make_shared<PinnedBuffer>();
        buffer->resize(buffer_size, image_info->cuda_stream);
        data = buffer->data;
        img_buffer_ = std::move(buffer);
    }
    image_info->buffer = static_cast<unsigned char*>(data);
}

void Image::initImageInfoFromDLPack(nvimgcodecImageInfo_t* image_info, py::capsule cap,
    std::optional<nvimgcodecSampleFormat_t> sample_format, std::optional<nvimgcodecColorSpec_t> color_spec)
{
    if (auto* tensor = static_cast<DLManagedTensor*>(cap.get_pointer())) {
        // check_cuda_buffer validates CUDA registration, which a plain host (kDLCPU)
        // pointer will never have; only applies to CUDA-accessible tensors.
        if (is_cuda_accessible(tensor->dl_tensor.device.device_type)) {
            check_cuda_buffer(tensor->dl_tensor.data);
        }
        dlpack_tensor_ = std::make_shared<DLPackTensor>(logger_, tensor);
        // signal that producer don't have to call tensor's deleter, consumer will do it instead
        cap.set_name("used_dltensor");
        dlpack_tensor_->getImageInfo(image_info, sample_format, color_spec);
        // getImageInfo() sets buffer_kind from the tensor's own device; mirror that into
        // device_id_ the same way the __array_interface__ branch does explicitly, so a CPU
        // DLPack tensor doesn't fall through to the trailing cudaPointerGetAttributes()
        // block below (which is gated on STRIDED_DEVICE and would not touch it anyway, but
        // leaving device_id_ at NVIMGCODEC_DEVICE_CURRENT would misreport a CPU-only image).
        if (image_info->buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST) {
            device_id_ = NVIMGCODEC_DEVICE_CPU_ONLY;
        }
    } else {
        throw std::runtime_error("Unsupported dlpack PyCapsule object.");
    }
}

namespace {

// Validate that user-provided precision is compatible with the sample type bitdepth.
// precision == 0 (the default) means "use full bitdepth" and is always valid.
// For floating-point sample types, only the dtype's full bitdepth (or 0) is accepted,
// because a sub-bitdepth "significant bits" count is an integer-image concept and has
// no meaningful interpretation for float storage.
void validatePrecisionForSampleType(int precision, nvimgcodecSampleDataType_t sample_type) {
    if (precision == 0) {
        return;
    }
    if (precision < 0) {
        throw std::invalid_argument("Invalid precision " + std::to_string(precision) +
                                    ". Precision must be a non-negative integer.");
    }
    int bitdepth = static_cast<int>(sample_type_to_bytes_per_element(sample_type)) * 8;
    if (precision > bitdepth) {
        throw std::invalid_argument("Precision " + std::to_string(precision) +
                                    " exceeds the bitdepth of the sample type (" +
                                    std::to_string(bitdepth) + " bits).");
    }
    bool is_float = sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT16 ||
                    sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32 ||
                    sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT64;
    if (is_float && precision != bitdepth) {
        throw std::invalid_argument("Invalid precision " + std::to_string(precision) +
                                    " for floating-point sample type. Only the dtype's full bitdepth (" +
                                    std::to_string(bitdepth) +
                                    ") or 0 (meaning full bitdepth) is supported for float dtypes.");
    }
}

// The allocation on the target_device relies on cudaMallocAsync, which takes
// the memory pool from the stream's device rather than the current context.
// A stream from a different device would silently allocate on the wrong
// device and leave the resulting Image.device_id inconsistent with the actual
// buffer location. The default (null) stream, cudaStreamLegacy, and
// cudaStreamPerThread are per-current-device sentinels that are resolved
// correctly by the surrounding DeviceGuard, so skip them.
void validate_stream_device(cudaStream_t stream, int target_device_id)
{
    if (stream == nullptr || stream == cudaStreamLegacy || stream == cudaStreamPerThread) {
        return;
    }
    int stream_dev = nvimgcodec::get_stream_device_id(stream);
    if (stream_dev != target_device_id) {
        throw std::invalid_argument(
            "cuda_stream belongs to device " + std::to_string(stream_dev) +
            " but device_id is " + std::to_string(target_device_id) +
            ". cuda_stream must be created on the same device as device_id.");
    }
}

// A subsampled image is not one rectangle, so there is no honest single-array view of
// it. Say so rather than handing back plane 0's shape over the whole buffer.
void throwIfChromaSubsampled(const nvimgcodecImageInfo_t& image_info)
{
    if (!is_unsubsampled_chroma(image_info.chroma_subsampling)) {
        throw std::runtime_error(
            "A chroma-subsampled image has planes of differing sizes and cannot be exposed as a "
            "single array. Use Image.planes to get one view per plane.");
    }
}

} // anonymous namespace

void Image::initImageInfoFromInterfaceDict(const py::dict& iface, nvimgcodecImageInfo_t* image_info,
    std::optional<nvimgcodecSampleFormat_t> sample_format, std::optional<nvimgcodecColorSpec_t> color_spec)
{
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

    // Layout (HWC vs CHW) and the sample_format / color_spec labels follow the
    // caller's sample_format, resolved by the shared helpers in type_utils.h so
    // this path and the DLPack path stay in lockstep.
    AsImageLayout layout = resolveAsImageLayout(vshape, sample_format);
    const bool is_interleaved = layout.is_interleaved;

    if (!is_padding_correct(iface, is_interleaved)) {
        if (has_negative_stride(iface)) {
            throw std::runtime_error("Unexpected array style. Reversed (negative-stride) views are not supported.");
        }
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

    image_info->num_planes = layout.num_planes;
    image_info->plane_info[0].height = layout.height;
    image_info->plane_info[0].width = layout.width;
    image_info->plane_info[0].num_channels = layout.num_channels;

    std::string typestr = iface["typestr"].cast<std::string>();
    auto sample_type = type_from_format_str(typestr);

    int bytes_per_element = sample_type_to_bytes_per_element(sample_type);

    const int num_channels =
        is_interleaved ? static_cast<int>(layout.num_channels) : static_cast<int>(layout.num_planes);
    AsImageLabels labels = resolveAsImageLabels(num_channels, sample_format, color_spec);
    image_info->sample_format = labels.sample_format;
    image_info->color_spec = labels.color_spec;
    image_info->chroma_subsampling = labels.chroma_subsampling;

    // Row stride axis: HWC -> stride[0]; 3-D CHW planar -> stride[1] (rows
    // live inside each plane); 2-D HW planar (P_Y) -> stride[0].
    const int row_stride_axis = (vshape.size() == 3 && !is_interleaved) ? 1 : 0;
    size_t pitch_in_bytes = vstrides.size() > 1
        ? vstrides[row_stride_axis]
        : static_cast<size_t>(image_info->plane_info[0].width) * image_info->plane_info[0].num_channels * bytes_per_element;
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

Image::Image(nvimgcodecInstance_t instance, ILogger* logger, PyObject* o, CudaStreamArg cuda_stream,
             std::optional<nvimgcodecSampleFormat_t> sample_format,
             std::optional<nvimgcodecColorSpec_t> color_spec,
             std::optional<int> precision)
    : instance_(instance)
    , logger_(logger)
    , device_id_(NVIMGCODEC_DEVICE_CURRENT)
    , img_buffer_{nvimgcodecImageInfo_t{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0}}
    , cuda_stream_owner_(cuda_stream.owner)
{
    if (!o) {
        throw std::runtime_error("Object cannot be None");
    }
    py::object tmp = py::reinterpret_borrow<py::object>(o);
    auto& image_info = std::get<nvimgcodecImageInfo_t>(img_buffer_);
    image_info.cuda_stream = reinterpret_cast<cudaStream_t>(cuda_stream.handle);
    if (py::isinstance<py::capsule>(tmp)) {
        py::capsule cap = tmp.cast<py::capsule>();
        initImageInfoFromDLPack(&image_info, cap, sample_format, color_spec);
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
            } else if (!cuda_stream.specified) {
                // Unspecified (omitted or None). Inherit CAI's associated
                // stream. CAI only carries an integer handle, so there is no
                // protocol object to pin.
                image_info.cuda_stream = reinterpret_cast<cudaStream_t>(*stream);
            }
            // else: explicit cuda_stream= (including 0) wins over CAI's stream
            // (which is the current stream at access time, not necessarily
            // the producer). Keep handle and owner from the argument.
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

        device_id_ = NVIMGCODEC_DEVICE_CPU_ONLY;
        initImageInfoFromInterfaceDict(iface, &image_info, sample_format, color_spec);
        image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        // Host images do not retain a CUDA stream.
        cuda_stream_owner_ = py::object();
        
        dlpack_tensor_ = std::make_shared<DLPackTensor>(logger_, image_info);
    } else if (hasattr(tmp, "__dlpack__")) {
        // Quickly check if we support the device, and whether the `stream` argument to
        // __dlpack__() below applies - it is only meaningful for a producer with device-side
        // compute to synchronize with. DLPack forbids a non-None stream for a CPU exporter,
        // and PyTorch extends that same rule to CUDA-pinned host tensors (kDLCUDAHost):
        // although is_cuda_accessible() treats that device as CUDA-accessible for buffer
        // validation, PyTorch reports its DLPack device as host-resident (torch.Tensor.device
        // == "cpu") and raises if a stream is passed. Assume CUDA as before when
        // __dlpack_device__ is unavailable.
        bool source_is_cuda_accessible = true;
        bool needs_stream_arg = true;
        if (hasattr(tmp, "__dlpack_device__")) {
            py::tuple dlpack_device = tmp.attr("__dlpack_device__")().cast<py::tuple>();
            auto dev_type = static_cast<DLDeviceType>(dlpack_device[0].cast<int>());
            source_is_cuda_accessible = is_cuda_accessible(dev_type);
            if (!source_is_cuda_accessible && dev_type != kDLCPU) {
                throw std::runtime_error("Unsupported device in DLTensor. Only CPU or CUDA-accessible memory buffers can be wrapped");
            }
            needs_stream_arg = dev_type != kDLCPU && dev_type != kDLCUDAHost;
        }
        py::capsule cap;
        if (needs_stream_arg) {
            // DLPack uses 1 for the legacy default stream and forbids 0, so
            // unspecified and explicit cuda_stream=0 both become stream=1.
            py::object py_cuda_stream =
                (cuda_stream.specified && cuda_stream.handle) ? py::int_((intptr_t)(cuda_stream.handle)) : py::int_(1);
            cap = tmp.attr("__dlpack__")("stream"_a = py_cuda_stream).cast<py::capsule>();
        } else {
            cap = tmp.attr("__dlpack__")().cast<py::capsule>();
        }
        initImageInfoFromDLPack(&image_info, cap, sample_format, color_spec);
    } else {
        throw std::runtime_error("Object does not support neither __cuda_array_interface__ nor __dlpack__");
    }

    if (device_id_ == NVIMGCODEC_DEVICE_CURRENT && image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
        cudaPointerAttributes ptr_attrs{};
        CHECK_CUDA(cudaPointerGetAttributes(&ptr_attrs, image_info.buffer));
        device_id_ = ptr_attrs.device;
    }

    for (uint32_t c = 1; c < image_info.num_planes; ++c) {
        assert(image_info.plane_info[c].sample_type == image_info.plane_info[0].sample_type);
    }
    if (precision.has_value()) {
        validatePrecisionForSampleType(precision.value(), image_info.plane_info[0].sample_type);
        for (uint32_t c = 0; c < image_info.num_planes; ++c) {
            image_info.plane_info[c].precision = static_cast<uint8_t>(precision.value());
        }
    }

    py::gil_scoped_release release;
    nvimgcodecImage_t image = nullptr;
    CHECK_NVIMGCODEC(nvimgcodecImageCreate(instance, &image, &image_info));
    image_ = std::shared_ptr<std::remove_pointer<nvimgcodecImage_t>::type>(
        image, [](nvimgcodecImage_t image) { nvimgcodecImageDestroy(image); });
}

void Image::initInterfaceDictFromImageInfo(py::dict* d, bool is_cuda) const
{
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    {
        py::gil_scoped_release release;
        nvimgcodecImageGetImageInfo(image_.get(), &image_info);
    }

    throwIfChromaSubsampled(image_info);

    // Check that the buffer kind is compatible with interface (host for __array_interface__, device for __cuda_array_interface__)
    // If not, instruct user to call .cpu() or .cuda() accordingly.
    if (is_cuda) {
        if (image_info.buffer_kind != NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
            throw std::runtime_error("Image buffer is not on device (expected device buffer for __cuda_array_interface__). "
                                     "Call '.cuda()' on the image to obtain a device-backed image before using the CUDA array interface.");
        }
    } else {
        if (image_info.buffer_kind != NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST) {
            throw std::runtime_error("Image buffer is not in host memory (expected host buffer for __array_interface__). "
                                     "Call '.cpu()' on the image to obtain a host-backed image before using the array interface.");
        }
    }

    const auto& plane_info = image_info.plane_info[0];
    size_t bytes_per_sample = plane_info.sample_type >> 11;
    size_t row_size = static_cast<size_t>(plane_info.width) * plane_info.num_channels * bytes_per_sample;
    bool is_continuous = plane_info.row_stride == row_size;

    (*d)["shape"] = shape();
    (*d)["strides"] = is_continuous? py::none() : py::object(strides());
    (*d)["typestr"] = format_str_from_type(plane_info.sample_type);
    (*d)["data"] = py::make_tuple(py::reinterpret_steal<py::object>(PyLong_FromVoidPtr(image_info.buffer)), false);
    (*d)["version"] = 3;
    
    if (is_cuda) {
        py::object stream = image_info.cuda_stream ? py::int_((intptr_t)(image_info.cuda_stream)) : py::int_(1);
        (*d)["stream"] = stream;
    }
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
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    {
        py::gil_scoped_release release;
        nvimgcodecImageGetImageInfo(image_.get(), &image_info);
    }
    // See Image::shape: a chroma-subsampled image has no single-array shape to report a
    // rank for either.
    throwIfChromaSubsampled(image_info);
    // Shape has always 3 dimensions either WHC (interleaved) or CHW (planar)
    return 3;
}

int Image::num_channels() const
{
    py::gil_scoped_release release;
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    nvimgcodecImageGetImageInfo(image_.get(), &image_info);
    int total_channels = 0;
    for (uint32_t i = 0; i < image_info.num_planes; ++i) {
        total_channels += image_info.plane_info[i].num_channels;
    }
    return total_channels;
}

py::dict Image::array_interface() const
{
    py::dict array_interface;
    try {
        initInterfaceDictFromImageInfo(&array_interface, false);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Unable to initialize __array_interface__: ") + e.what());
    } catch (...) {
        throw std::runtime_error("Unable to initialize __array_interface__: unknown exception");
    }
    return array_interface;
}

py::dict Image::cuda_interface() const
{
    py::dict cuda_array_interface;
    try {
        initInterfaceDictFromImageInfo(&cuda_array_interface, true);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Unable to initialize __cuda_array_interface__: ") + e.what());
    } catch (...) {
        throw std::runtime_error("Unable to initialize __cuda_array_interface__: unknown exception");
    }
    return cuda_array_interface;
}

py::tuple Image::getPlanes(py::object self) const
{
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    {
        py::gil_scoped_release release;
        nvimgcodecImageGetImageInfo(image_.get(), &image_info);
    }

    bool is_cuda;
    switch (image_info.buffer_kind) {
    case NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE:
        is_cuda = true;
        break;
    case NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST:
        is_cuda = false;
        break;
    default:
        throw std::runtime_error(
            "Unable to expose planes: image buffer is not on the host or the device. Call .cpu() or .cuda() first.");
    }

    size_t offsets[NVIMGCODEC_MAX_NUM_PLANES];
    if (!SafePlaneByteOffsets(image_info, offsets))
        throw std::runtime_error("Unable to expose planes: plane byte offsets overflow size_t.");

    py::list planes;
    for (uint32_t p = 0; p < image_info.num_planes; ++p) {
        const auto& pi = image_info.plane_info[p];
        const size_t bytes_per_sample = sample_type_to_bytes_per_element(pi.sample_type);

        // Assigned rather than selected with ?:, because pybind11 3.0.4 makes
        // make_tuple return an arity-typed pybind11::typing::Tuple; two different
        // arities in the same conditional are then mutually convertible and the
        // ternary is ambiguous. (The vendored 3.0.1 still returns a plain tuple,
        // so this only broke the conda builds, which link the system pybind11.)
        py::tuple shape;
        py::tuple strides;
        if (pi.num_channels > 1) {
            shape = py::make_tuple(pi.height, pi.width, pi.num_channels);
            strides = py::make_tuple(pi.row_stride, bytes_per_sample * pi.num_channels, bytes_per_sample);
        } else {
            shape = py::make_tuple(pi.height, pi.width);
            strides = py::make_tuple(pi.row_stride, bytes_per_sample);
        }

        py::dict iface;
        iface["shape"] = shape;
        iface["strides"] = strides;
        iface["typestr"] = format_str_from_type(pi.sample_type);
        void* plane_ptr = static_cast<uint8_t*>(image_info.buffer) + offsets[p];
        iface["data"] = py::make_tuple(py::reinterpret_steal<py::object>(PyLong_FromVoidPtr(plane_ptr)), /*read_only=*/false);
        iface["version"] = 3;
        if (is_cuda) {
            iface["stream"] = image_info.cuda_stream ? py::int_((intptr_t)(image_info.cuda_stream)) : py::int_(1);
        }

        // A plain holder is enough: the consumer reads the interface dict off it. The
        // buffer described by that dict belongs to this Image, so each returned array
        // must keep this Image alive individually -- not just the tuple that carries it,
        // since callers commonly unpack (`y, u, v = img.planes`) and drop the tuple.
        py::object holder = py::module_::import("types").attr("SimpleNamespace")();
        holder.attr(is_cuda ? "__cuda_array_interface__" : "__array_interface__") = iface;

        py::module_ xp = py::module_::import(is_cuda ? "cupy" : "numpy");
        py::object arr = xp.attr("asarray")(holder);
        py::detail::keep_alive_impl(arr, self);
        planes.append(std::move(arr));
    }
    return py::tuple(planes);
}

py::object Image::toNumpyArray(py::object dtype_obj, py::object copy_obj) const
{
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    {
        py::gil_scoped_release release;
        nvimgcodecImageGetImageInfo(image_.get(), &image_info);
    }

    throwIfChromaSubsampled(image_info);

    if (image_info.buffer_kind != NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST) {
        throw std::runtime_error(
            "Cannot convert image to a NumPy array: image data is in GPU device memory.\n"
            "Call `.cpu()` first to copy the data to CPU memory, for example:\n"
            "    np.array(img.cpu())");
    }

    const auto& plane_info = image_info.plane_info[0];
    size_t bytes_per_sample = sample_type_to_bytes_per_element(plane_info.sample_type);
    bool is_interleaved = is_sample_format_interleaved(image_info.sample_format);

    py::dtype native_dtype(format_str_from_type(plane_info.sample_type));

    std::vector<py::ssize_t> arr_shape;
    std::vector<py::ssize_t> arr_strides;

    if (is_interleaved) {
        arr_shape   = {(py::ssize_t)plane_info.height, (py::ssize_t)plane_info.width, (py::ssize_t)plane_info.num_channels};
        arr_strides = {(py::ssize_t)plane_info.row_stride,
                       (py::ssize_t)(bytes_per_sample * plane_info.num_channels),
                       (py::ssize_t)bytes_per_sample};
    } else {
        py::ssize_t plane_size = (py::ssize_t)plane_info.row_stride * (py::ssize_t)plane_info.height;
        arr_shape   = {(py::ssize_t)image_info.num_planes, (py::ssize_t)plane_info.height, (py::ssize_t)plane_info.width};
        arr_strides = {plane_size, (py::ssize_t)plane_info.row_stride, (py::ssize_t)bytes_per_sample};
    }

    // Create a numpy array referencing this image's buffer; pass `this` as base so NumPy
    // keeps the Image alive for as long as the array exists.
    py::array arr(native_dtype, arr_shape, arr_strides, image_info.buffer,
                  py::cast(const_cast<Image*>(this), py::return_value_policy::reference));

    py::module_ np = py::module_::import("numpy");
    return np.attr("array")(arr, "dtype"_a = dtype_obj, "copy"_a = copy_obj);
}

py::tuple Image::shape() const
{
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    {
        py::gil_scoped_release release;
        nvimgcodecImageGetImageInfo(image_.get(), &image_info);
    }
    // A subsampled image's planes differ in size, so no single (C, H, W) tuple
    // describes the buffer; reporting plane 0's extents for every plane would
    // contradict Image.size, which is correctly computed per plane.
    throwIfChromaSubsampled(image_info);
    bool is_interleaved = is_sample_format_interleaved(image_info.sample_format);
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
    // See Image::shape: a uniform plane stride cannot describe planes of differing sizes.
    throwIfChromaSubsampled(image_info);
    int bytes_per_element = sample_type_to_bytes_per_element(image_info.plane_info[0].sample_type);
    bool is_interleaved = is_sample_format_interleaved(image_info.sample_format);
    py::tuple strides_tuple;
    if (is_interleaved) {
        strides_tuple = py::make_tuple(image_info.plane_info[0].row_stride,
                                      static_cast<size_t>(image_info.plane_info[0].num_channels) * static_cast<size_t>(bytes_per_element),
                                      bytes_per_element);
    } else {
        strides_tuple = py::make_tuple(image_info.plane_info[0].row_stride * image_info.plane_info[0].height,
                                      image_info.plane_info[0].row_stride, bytes_per_element);
    }
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
    return resolve_precision(image_info.plane_info[0].precision, image_info.plane_info[0].sample_type);
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

nvimgcodecChromaSubsampling_t Image::getChromaSubsampling() const
{
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    {
        py::gil_scoped_release release;
        nvimgcodecImageGetImageInfo(image_.get(), &image_info);
    }
    return image_info.chroma_subsampling;
}

std::optional<int> Image::getDeviceId() const
{
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    {
        py::gil_scoped_release release;
        nvimgcodecImageGetImageInfo(image_.get(), &image_info);
    }
    if (image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST) {
        return std::nullopt;
    }
    return device_id_;
}

std::optional<intptr_t> Image::getCudaStream() const
{
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    {
        py::gil_scoped_release release;
        nvimgcodecImageGetImageInfo(image_.get(), &image_info);
    }
    if (image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST) {
        return std::nullopt;
    }
    return reinterpret_cast<intptr_t>(image_info.cuda_stream);
}

void Image::setCudaStreamOwner(py::object owner)
{
    cuda_stream_owner_ = std::move(owner);
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
    } else if (std::holds_alternative<std::shared_ptr<PageableHostBuffer>>(img_buffer_)) {
        auto host_buffer = std::get<std::shared_ptr<PageableHostBuffer>>(img_buffer_);
        if (host_buffer) {
            return host_buffer->size;
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
    } else if (std::holds_alternative<std::shared_ptr<PageableHostBuffer>>(img_buffer_)) {
        auto host_buffer = std::get<std::shared_ptr<PageableHostBuffer>>(img_buffer_);
        if (host_buffer) {
            return host_buffer->capacity;
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
    // DLPack has no way to describe planes of differing sizes: the tensor would claim
    // (num_planes, H, W) full-size elements over a smaller, subsampled allocation and
    // any consumer reading it would run off the end of the buffer.
    throwIfChromaSubsampled(image_info);

    intptr_t consumer_stream = 0;
    if (!stream_obj.is_none()) {
        // None keeps the DLPack default (no explicit consumer stream).
        consumer_stream = as_cuda_stream_arg(stream_obj).handle;
    }

    py::capsule cap = dlpack_tensor_->getPyCapsule(consumer_stream, image_info.cuda_stream);
    if (std::string(cap.name()) != "dltensor") {
        throw std::runtime_error(
            "Could not get DLTensor capsule. The underlying image may have been destroyed or is invalid.");
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

        auto image = new Image(instance_, logger_, &cpu_image_info, NVIMGCODEC_DEVICE_CPU_ONLY);
        {
            py::gil_scoped_release release;
            DeviceGuard device_guard(device_id_);
            CopyImage(cpu_image_info, image_info, cudaMemcpyDeviceToHost, image_info.cuda_stream);
            CHECK_CUDA(cudaStreamSynchronize(image_info.cuda_stream));
        }
        return py::cast(image,  py::return_value_policy::take_ownership);
    } else if (image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST) {
        return py::cast(this);
    } else {
        return py::none();
    }
}

py::object Image::cuda(bool synchronize, int device_id, CudaStreamArg cuda_stream)
{
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    {
        py::gil_scoped_release release;
        nvimgcodecImageGetImageInfo(image_.get(), &image_info);
    }
    auto target_stream = reinterpret_cast<cudaStream_t>(cuda_stream.handle);

    if (image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST) {
        nvimgcodecImageInfo_t cuda_image_info(image_info);
        cuda_image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        cuda_image_info.buffer = nullptr;
        cuda_image_info.cuda_stream = target_stream;
        for (unsigned int c = 0; c < cuda_image_info.num_planes; ++c) {
            auto& plane_info = cuda_image_info.plane_info[c];
            size_t bpp = TypeSize(plane_info.sample_type);
            plane_info.row_stride = static_cast<size_t>(plane_info.width) * bpp * plane_info.num_channels;
        }
        assert(GetBufferSize(cuda_image_info) == GetImageSize(cuda_image_info));

        auto target_device_id = resolve_device_id(device_id);
        validate_stream_device(target_stream, target_device_id);
        auto image = new Image(instance_, logger_, &cuda_image_info, target_device_id);
        image->setCudaStreamOwner(cuda_stream.owner);
        {
            py::gil_scoped_release release;
            DeviceGuard device_guard(target_device_id);
            CopyImage(cuda_image_info, image_info, cudaMemcpyHostToDevice, target_stream);
            if (synchronize) {
                CHECK_CUDA(cudaStreamSynchronize(target_stream));
            }
        }
        // The async memcpy is enqueued on target_stream. Break the library's
        // dependency on target_stream so the caller may destroy it at any
        // time after this call returns.
        image->detachFromUserStream();
        return py::cast(image,  py::return_value_policy::take_ownership);
    } else if (image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
        int target_device_id = resolve_device_id(device_id);
        if (target_device_id == device_id_) {
            return py::cast(this);
        }
        validate_stream_device(target_stream, target_device_id);
        // Cross-device copy: create new image on target device with the same layout
        nvimgcodecImageInfo_t cuda_image_info(image_info);
        cuda_image_info.buffer = nullptr;
        cuda_image_info.cuda_stream = target_stream;
        auto image = new Image(instance_, logger_, &cuda_image_info, target_device_id);
        image->setCudaStreamOwner(cuda_stream.owner);
        {
            py::gil_scoped_release release;
            CHECK_CUDA(cudaMemcpyPeerAsync(
                cuda_image_info.buffer, target_device_id,
                image_info.buffer, device_id_,
                GetBufferSize(cuda_image_info),
                target_stream));
            if (synchronize) {
                CHECK_CUDA(cudaStreamSynchronize(target_stream));
            }
        }
        // The async peer copy is enqueued on target_stream. Break the
        // library's dependency on target_stream so the caller may destroy it
        // at any time after this call returns.
        image->detachFromUserStream();
        return py::cast(image, py::return_value_policy::take_ownership);
    } else {
        return py::none();
    }
}

void Image::detachFromUserStream() noexcept
{
    if (auto* db = std::get_if<std::shared_ptr<DeviceBuffer>>(&img_buffer_)) {
        if (*db) (*db)->detach_from_stream();
    } else if (auto* pb = std::get_if<std::shared_ptr<PinnedBuffer>>(&img_buffer_)) {
        if (*pb) (*pb)->detach_from_stream();
    }
    // PageableHostBuffer does not reference any stream, so nothing to do.
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
    } else if (std::holds_alternative<std::shared_ptr<PageableHostBuffer>>(img_buffer_)) {
        auto host_buffer = std::get<std::shared_ptr<PageableHostBuffer>>(img_buffer_);
        host_buffer->resize(GetBufferSize(*image_info));
        image_info->buffer = static_cast<unsigned char*>(host_buffer->data);
        image_info->buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
    } else { // externally manager buffer
        assert(std::holds_alternative<nvimgcodecImageInfo_t>(img_buffer_));
        auto buffer_info = std::get<nvimgcodecImageInfo_t>(img_buffer_);

        // True when every plane of `info` has its natural (packed) row stride
        // and the plane stride is implicit `height * row_stride` - i.e. the
        // buffer is one tightly-packed contiguous chunk. is_padding_correct
        // already rejects plane gaps for externally-managed buffers, so we
        // only need to check the row strides here.
        auto is_packed_contiguous = [](const nvimgcodecImageInfo_t& info) {
            for (uint32_t c = 0; c < info.num_planes; ++c) {
                const auto& p = info.plane_info[c];
                const size_t natural =
                    static_cast<size_t>(p.width) * p.num_channels *
                    sample_type_to_bytes_per_element(p.sample_type);
                if (p.row_stride != natural) {
                    return false;
                }
            }
            return true;
        };

        if (buffer_info.num_planes != image_info->num_planes) {
            // Cross-layout reuse (e.g. an HWC buffer reused as a CHW decode
            // target or vice versa). The bytes are reinterpreted under the
            // decoder's resolved layout, so this is only well-defined when
            // both sides are flat tightly-packed regions: any row or plane
            // padding would create gaps the new layout cannot represent.
            if (!is_packed_contiguous(buffer_info)) {
                throw std::invalid_argument(
                    "External buffer plane count (" + std::to_string(buffer_info.num_planes) +
                    ") does not match the decoder's plane count (" +
                    std::to_string(image_info->num_planes) +
                    "). Cross-layout reuse requires the buffer to be row- and "
                    "plane-contiguous (no padding); either remove the padding or "
                    "wrap the buffer with a sample_format that produces the same "
                    "layout as the decoded output.");
            }
            if (GetBufferSize(*image_info) > GetBufferSize(buffer_info)) {
                throw std::invalid_argument(
                    "Existing buffer is too small to fit new image after layout reinterpretation.");
            }
            // image_info's plane row strides are already the packed naturals
            // (decoder.cpp sets them that way before calling reuse), so no
            // adoption from the buffer is meaningful here.
        } else {
            // Same plane count: the existing row-padded reuse logic.
            const auto& new_image_plane = image_info->plane_info[0];
            size_t new_image_row_size = (static_cast<size_t>(new_image_plane.width) *
                new_image_plane.num_channels *
                sample_type_to_bytes_per_element(new_image_plane.sample_type)
            );
            assert(new_image_row_size == new_image_plane.row_stride); // no row padding on the new image

            const auto& buffer_plane = buffer_info.plane_info[0];
            size_t buffer_row_size = (static_cast<size_t>(buffer_plane.width) *
                buffer_plane.num_channels *
                sample_type_to_bytes_per_element(buffer_plane.sample_type)
            );

            if (image_info->plane_info[0].height > buffer_info.plane_info[0].height ||
                new_image_row_size > buffer_row_size
            ) {
                // Image is taller / wider than the buffer's per-plane bounds -
                // the buffer must be contiguous and large enough in total.
                if (buffer_row_size != buffer_plane.row_stride) {
                    throw std::invalid_argument(
                        "Existing buffer is not continuous. Row size or height are too small to fit new image.");
                }
                if (GetBufferSize(*image_info) > GetBufferSize(buffer_info)) {
                    throw std::invalid_argument("Existing buffer is too small to fit new image");
                }
                // Keep the new image's natural row_stride.
            } else {
                // Image fits in the existing buffer; adopt the buffer's row
                // stride (which may carry right-side padding) plane-by-plane.
                for (uint32_t c = 0; c < image_info->num_planes; ++c) {
                    image_info->plane_info[c].row_stride = buffer_info.plane_info[c].row_stride;
                }
            }
        }

        if (buffer_info.cuda_stream != image_info->cuda_stream) {
            // TODO: could be done via event sync, but we should also add it to device_buffer.cpp then
            CHECK_CUDA(cudaStreamSynchronize(buffer_info.cuda_stream));
        }

        image_info->buffer = buffer_info.buffer;
        image_info->buffer_kind = buffer_info.buffer_kind;
    }

    // Decoded Image reuse differs from encoded CodeStream reuse: this
    // image_info is the output buffer layout used immediately for decoding.
    //
    // nvimgcodecImageCreate's [in,out] image parameter contract (see
    // include/nvimgcodec.h): when *image is non-NULL on input, the existing
    // handle is reused in-place and no fresh allocation occurs. We pass
    // image_.get() (already non-NULL on this path), so new_image after the
    // call is the same handle owned by image_; nothing leaks. Coverity cannot
    // see the cross-function API contract and conservatively flags new_image
    // as allocated-and-leaked when it goes out of scope at the end of this function.
    nvimgcodecImage_t new_image = image_.get();
    CHECK_NVIMGCODEC(nvimgcodecImageCreate(instance_, &new_image, image_info));
    assert(new_image == image_.get());

    // Update DLPackTensor with new image info and buffer
    dlpack_tensor_ = std::make_shared<DLPackTensor>(logger_, *image_info, img_buffer_);
    // coverity[leaked_storage : FALSE]
}

void Image::exportToPython(py::module& m)
{
    // clang-format off
    py::class_<Image>(m, "Image",
            R"pbdoc(A pixel buffer with associated metadata. Holds either decoded pixels or pixels that are to be encoded.

            The buffer's layout is reflected in the image's ``shape`` and ``sample_format``:

            * Interleaved (``I_*`` sample formats): 3-D shape ``(H, W, C)``.
            * Planar (``P_*`` sample formats): 3-D shape ``(C, H, W)``.

            The buffer is C-style contiguous along all axes except the row stride,
            which may include padding.
            )pbdoc")
        .def_property_readonly("__array_interface__", &Image::array_interface,
            R"pbdoc(
            The array interchange interface compatible with Numba v0.39.0 or later (see
            `CUDA Array Interface <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_ for details)

            Raises ``RuntimeError`` for a chroma-subsampled image; use ``planes`` instead.
            )pbdoc")
        .def_property_readonly("__cuda_array_interface__", &Image::cuda_interface,
            R"pbdoc(
            The CUDA array interchange interface compatible with Numba v0.39.0 or later (see
            `CUDA Array Interface <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_ for details)

            Raises ``RuntimeError`` for a chroma-subsampled image; use ``planes`` instead.
            )pbdoc")
        // __array__ is defined even though __array_interface__ already raises a helpful error
        // for GPU images. The reason: NumPy only falls through to __array_interface__ and the
        // DLPack protocol (__dlpack__ / __dlpack_device__) when __array__ does *not exist*
        // (AttributeError). If __array__ exists and raises any other exception, NumPy stops
        // immediately and propagates it — no fallback to DLPack, no risk of a segfault from
        // NumPy trying to access GPU memory as if it were CPU memory.
        .def("__array__", &Image::toNumpyArray,
            "dtype"_a = py::none(), "copy"_a = py::none(),
            R"pbdoc(
            Convert image to a NumPy array.

            This method is called implicitly by ``numpy.array()`` / ``numpy.asarray()``.
            The image must be in CPU (host) memory. If the image was decoded to GPU memory
            (the default), call `.cpu()` first::

                arr = np.array(img.cpu())

            Args:
                dtype: Optional target dtype. If ``None``, the native dtype of the image is used.
                copy:  Copy semantics per the NumPy 2.x protocol. ``True`` always copies,
                       ``False`` raises ``ValueError`` if a copy would be required (e.g. dtype
                       conversion), ``None`` (default) copies only if needed.

            Returns:
                numpy.ndarray with the image data.

            Raises:
                RuntimeError: If the image data is in GPU device memory, or if the image is
                              chroma-subsampled (use ``planes`` instead).
            )pbdoc")
        .def_property_readonly("shape", &Image::shape,
            R"pbdoc(
            The shape of the image.

            Raises ``RuntimeError`` for a chroma-subsampled image, whose planes differ in size;
            use ``planes`` instead.
            )pbdoc")
        .def_property_readonly("strides", &Image::strides,
            R"pbdoc(
            Strides of axes in bytes.

            Raises ``RuntimeError`` for a chroma-subsampled image; use ``planes`` instead.
            )pbdoc")
        .def_property_readonly("width", &Image::getWidth,
            R"pbdoc(
            The width of the image in pixels.
            )pbdoc")
        .def_property_readonly("height", &Image::getHeight,
            R"pbdoc(
            The height of the image in pixels.
            )pbdoc")
        .def_property_readonly("num_channels", &Image::num_channels,
            R"pbdoc(
            The overall number of channels in the image across all planes.
            )pbdoc")
        .def_property_readonly("ndim", &Image::getNdim,
            R"pbdoc(
            The number of dimensions in the image.

            Raises ``RuntimeError`` for a chroma-subsampled image; use ``planes`` instead.
            )pbdoc")
        .def_property_readonly("dtype", &Image::dtype,
            R"pbdoc(
            The data type (dtype) of the image samples.
            )pbdoc")
        .def_property_readonly("precision", &Image::precision,
            R"pbdoc(
            Number of significant bits per sample.

            For lower-precision data stored in a wider container (e.g. 12-bit values held in a uint16 buffer)
            this is the bitdepth set explicitly by the caller (e.g. via ``nvimgcodec.as_image(arr, precision=12)``).
            When no explicit precision was provided, this returns the full bitdepth of the sample data type
            (8 for ``uint8``, 16 for ``uint16``, 32 for ``float32``, etc.).
            )pbdoc")
        .def_property_readonly("sample_format", &Image::getSampleFormat, 
            R"pbdoc(
            The sample format of the image indicating how color components are matched to channels and channels to planes.
            )pbdoc")
        .def_property_readonly("color_spec", &Image::getColorSpec,
            R"pbdoc(
            Color specification of the image indicating how the color information in image samples should be interpreted.
            )pbdoc")
        .def_property_readonly("chroma_subsampling", &Image::getChromaSubsampling,
            R"pbdoc(
            Chroma subsampling format of the image.

            For a decoded image this describes the actual plane layout of the output, after
            EXIF orientation is applied, and can differ from the requested
            ``DecodeParams.chroma_subsampling``: a 90 or 270 degree rotation transposes the
            sampling grid, so for example a ``CSS_422`` request on such a source reports
            ``CSS_440``.
            )pbdoc")
        .def_property_readonly("planes",
            [](py::object self) { return self.cast<Image&>().getPlanes(self); },
            R"pbdoc(
            Per-plane views of the image buffer, one array per plane.

            Each view carries that plane's own geometry, so for a subsampled image the chroma
            planes are smaller than the luma plane. Views share memory with the Image. Each view
            keeps the ``Image`` alive, so a view stays valid even after the ``Image`` itself goes
            out of scope. numpy arrays for host buffers, cupy for device.
            )pbdoc")
        .def_property_readonly("device_id", &Image::getDeviceId,
            R"pbdoc(
            The CUDA device id associated with this image's buffer, or ``None`` if the image is in host (CPU) memory.
            )pbdoc")
        .def_property_readonly("cuda_stream", &Image::getCudaStream,
            R"pbdoc(
            The CUDA stream associated with this image as a Python integer, or None if the image is in host (CPU) memory.
            If this handle came from a CUDA Stream Protocol object, the Image holds that object until released. A bare integer passed as ``cuda_stream`` is not retained.
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
        .def("__dlpack__", &Image::dlpack, "stream"_a = py::none(),
            "Export the image as a DLPack tensor. Raises ``RuntimeError`` for a chroma-subsampled image; use ``planes`` instead.")
        .def("__dlpack_device__", &Image::getDlpackDevice, "Get the device associated with the buffer")
        .def("to_dlpack", &Image::dlpack,
            R"pbdoc(
            Export the image with zero-copy conversion to a DLPack tensor. 
            
            Args:
                cuda_stream: Optional CUDA stream upon which the consumer should
                             synchronize with this Image's producer stream.
                             Accepts ``None`` (DLPack default: no explicit consumer
                             stream), a ``cudaStream_t`` handle as a Python integer,
                             or any object implementing the CUDA Stream Protocol
                             (``__cuda_stream__()`` -> ``(version, stream_ptr)``,
                             version 0). Unlike ``as_image`` / ``Image.cuda``, this
                             export does not pin a protocol object on the Image.

            Returns:
                DLPack tensor which is encapsulated in a PyCapsule object.

            Raises:
                RuntimeError: If the image is chroma-subsampled; use ``planes`` instead.
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
            Returns a copy of this image in device memory. If this image is already in device memory
            on the same device, no copy is performed and the original object is returned. If a
            different device_id is specified, a new image is created on the target device with the
            content copied via peer-to-peer transfer.
            
            Args:
                synchronize: If True (by default) it blocks and waits for copy to be finished, 
                             else no synchronization is executed and further synchronization needs to be done using
                             cuda stream provided by e.g. \_\_cuda_array_interface\_\_. 
                device_id: Target device id. If equals to -1 (by default) the current device will be used.
                           When the image is already on a GPU and a different device_id is specified,
                           a device-to-device copy is performed.
                cuda_stream: An optional CUDA stream upon which synchronization must take place.
                             The data transfer is enqueued on this stream.
                             Accepts ``None`` (unspecified), a ``cudaStream_t`` handle as a Python integer, or any object
                             implementing the CUDA Stream Protocol (``__cuda_stream__()`` ->
                             ``(version, stream_ptr)``, version 0), such as a CuPy or cuda-python
                             stream. This Image holds a protocol object until it is released,
                             so ``Image.cuda_stream`` and ``__cuda_array_interface__['stream']``
                             stay valid. A bare integer handle is not retained; the caller must
                             keep that stream alive while those values are used. See the
                             `CUDA Stream Protocol
                             <https://nvidia.github.io/cuda-python/cuda-core/latest/interoperability.html#cuda-stream-protocol>`_.
                             Must belong to the same device as
                             ``device_id``; passing a stream created on a different device raises
                             ``ValueError``. ``None`` or omitted is unspecified and is
                             treated as ``0``, the legacy default stream, which
                             is per-device: the copy runs under an internal device guard set to
                             ``device_id``, so the default stream of ``device_id`` is used.
                             Ignored when the image is already on the target device (no copy is
                             performed).

                             Lifetime: the resulting Image's internal buffer is detached from
                             ``cuda_stream`` before this call returns, so deallocation does not
                             use the caller's stream. Other Image methods that enqueue
                             asynchronous work (notably ``Image.cpu()``, which performs a
                             device-to-host copy on ``cuda_stream``) must still be called while
                             ``cuda_stream`` is alive. If ``synchronize=False``, the caller is
                             also responsible for synchronizing ``cuda_stream`` (or a downstream
                             consumer of it) before reading the resulting Image's contents.

            Returns:
                Image object with content in device memory or None if copy could not be done.
            )pbdoc",
            "synchronize"_a = true, "device_id"_a = NVIMGCODEC_DEVICE_CURRENT, "cuda_stream"_a = py::none());
    // clang-format on
}


} // namespace nvimgcodec
