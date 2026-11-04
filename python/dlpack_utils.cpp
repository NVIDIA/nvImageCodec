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

#include "dlpack_utils.h"
#include "type_utils.h"
#include "error_handling.h"
#include <imgproc/device_buffer.h>
#include <imgproc/pinned_buffer.h>

#include <ilogger.h>
#include <log.h>
#include <algorithm>
#include <memory>

namespace nvimgcodec {

namespace {

// Helper: Common deleter for DLManagedTensor - cleans up shape, strides, and manager_ctx
void deleteDLManagedTensor(DLManagedTensor* self) {
    if (!self) return;
    
    if (self->dl_tensor.shape != nullptr) {
        delete[] self->dl_tensor.shape;
        self->dl_tensor.shape = nullptr;
    }
    if (self->dl_tensor.strides != nullptr) {
        delete[] self->dl_tensor.strides;
        self->dl_tensor.strides = nullptr;
    }
    if (self->manager_ctx != nullptr) {
        delete static_cast<std::shared_ptr<DLPackTensorSharedState>*>(self->manager_ctx);
        self->manager_ctx = nullptr;
    }
}

// Helper: Copy shape and strides arrays from source to newly allocated arrays
// Returns pair of unique_ptrs (strides may be nullptr)
// Exception-safe: uses unique_ptr to prevent leaks if copy throws
std::pair<std::unique_ptr<int64_t[]>, std::unique_ptr<int64_t[]>> copyShapeAndStrides(const DLTensor& source) {
    std::unique_ptr<int64_t[]> shape;
    std::unique_ptr<int64_t[]> strides;
    
    if (source.ndim > 0) {
        shape.reset(new int64_t[source.ndim]);
        std::copy(source.shape, source.shape + source.ndim, shape.get());
        
        if (source.strides != nullptr) {
            strides.reset(new int64_t[source.ndim]);
            std::copy(source.strides, source.strides + source.ndim, strides.get());
        }
    }
    
    return {std::move(shape), std::move(strides)};
}

// DLPack counterpart of `is_padding_correct` (python/type_utils.h): only the row
// dimension may carry padding; all inner dimensions must be packed. Both share the
// `are_strides_row_padded_only` loop in type_utils.h.
//
// This is required because the cross-device path in Image::cuda() performs a flat
// `cudaMemcpyPeerAsync` over `height * row_stride` bytes and would silently copy
// non-image memory if any inner axis had a non-trivial stride.
bool is_dlpack_padding_correct(const DLTensor& tensor, bool is_interleaved)
{
    if (tensor.strides == nullptr) {
        return true;
    }
    // DLPack strides are in *elements* (not bytes), so the base element stride
    // is 1. Defer to the shared check in type_utils.h.
    std::vector<int64_t> shape(tensor.shape, tensor.shape + tensor.ndim);
    std::vector<int64_t> strides(tensor.strides, tensor.strides + tensor.ndim);
    return are_strides_row_padded_only(shape, strides, /*base_stride=*/1, is_interleaved);
}

// Distinguishes a negative-stride (reversed/flipped view) rejection from a genuine
// non-row-padding rejection, so the throw site can pick an accurate message instead
// of always reporting "padding" for something that is not a padding issue at all.
bool dlpack_has_negative_stride(const DLTensor& tensor)
{
    if (tensor.strides == nullptr) {
        return false;
    }
    for (int i = 0; i < tensor.ndim; ++i) {
        if (tensor.strides[i] < 0) {
            return true;
        }
    }
    return false;
}

} // anonymous namespace

bool is_cuda_accessible(DLDeviceType devType)
{
    switch (devType) {
    case kDLCUDAHost:
    case kDLCUDA:
    case kDLCUDAManaged:
        return true;
    default:
        return false;
    }
}

nvimgcodecSampleDataType_t type_from_dlpack(const DLDataType& dtype)
{
    nvimgcodecSampleDataType_t data_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UNSUPPORTED;
    switch (dtype.code) {
    case kDLBool:
    case kDLInt:
        switch (dtype.bits) {
        case 8:
            data_type = NVIMGCODEC_SAMPLE_DATA_TYPE_INT8;
            break;
        case 16:
            data_type = NVIMGCODEC_SAMPLE_DATA_TYPE_INT16;
            break;
        case 32:
            data_type = NVIMGCODEC_SAMPLE_DATA_TYPE_INT32;
            break;
        case 64:
            data_type = NVIMGCODEC_SAMPLE_DATA_TYPE_INT64;
            break;
        }
        break;
    case kDLUInt:
        switch (dtype.bits) {
        case 8:
            data_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
            break;
        case 16:
            data_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16;
            break;
        case 32:
            data_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT32;
            break;
        case 64:
            data_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT64;
            break;
        }
        break;
    case kDLFloat:
        switch (dtype.bits) {
        case 16:
            data_type = NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT16;
            break;
        case 32:
            data_type = NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32;
            break;
        case 64:
            data_type = NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT64;
            break;
        }
        break;
    //case kDLComplex:
    default:
        throw std::runtime_error("Data type code not supported, must be Int, UInt, Float or Bool");
    }

    return data_type;
}

DLDataType type_to_dlpack(nvimgcodecSampleDataType_t data_type)
{
    DLDataType dt = {};
    dt.lanes = 1;

    switch (data_type) {
    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT8:
        dt.code = kDLInt;
        dt.bits = 8;
        break;
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8:
        dt.code = kDLUInt;
        dt.bits = 8;
        break;

    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT16:
        dt.code = kDLInt;
        dt.bits = 16;
        break;
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16:
        dt.code = kDLUInt;
        dt.bits = 16;

        break;
    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT32:
        dt.code = kDLInt;
        dt.bits = 32;
        break;
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT32:
        dt.code = kDLUInt;
        dt.bits = 32;
        break;
    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT64:
        dt.code = kDLInt;
        dt.bits = 64;
        break;
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT64:
        dt.code = kDLUInt;
        dt.bits = 64;
        break;
    case NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT16:
        dt.code = kDLFloat;
        dt.bits = 16;
        break;
    case NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32:
        dt.code = kDLFloat;
        dt.bits = 32;
        break;
    case NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT64:
        dt.code = kDLFloat;
        dt.bits = 64;
        break;
        //TODO Complex type
        // case nvcv::DataKind::COMPLEX:
        //     dt.code = kDLComplex;
        //     break;
    default:
        throw std::runtime_error("Sample data type not supported, must be UNSIGNED, SIGNED, FLOAT"); //TODO or COMPLEX
    }

    return dt;
}

// Default constructor - no tensor or shared state
DLPackTensor::DLPackTensor(ILogger* logger) noexcept
    : internal_dl_managed_tensor_{}
    , shared_state_{nullptr}  // No shared state - not exportable
    , logger_{logger}
{
}

// Import constructor - wraps external DLManagedTensor (from_dlpack path)
// Creates an internal copy of the tensor metadata to enable re-export while keeping
// the external tensor alive in shared_state_
DLPackTensor::DLPackTensor(ILogger* logger, DLManagedTensor* dl_managed_tensor)
    : internal_dl_managed_tensor_{}
    , shared_state_{std::make_shared<DLPackTensorSharedState>(dl_managed_tensor, logger)}
    , logger_{logger}
{
    // Copy tensor metadata into our internal structure
    internal_dl_managed_tensor_.dl_tensor = dl_managed_tensor->dl_tensor;

    // Allocate the shape/strides buffers first; the unique_ptrs keep them
    // owned for the rest of this function and will free them on stack
    // unwinding if anything below throws.
    auto [shape, strides] = copyShapeAndStrides(dl_managed_tensor->dl_tensor);

    // Allocate manager_ctx before transferring ownership of shape/strides.
    // 'new std::shared_ptr<...>' is the only call site below that can throw
    // (std::bad_alloc); doing it first means a failure here leaves shape/
    // strides still owned by their unique_ptrs.
    internal_dl_managed_tensor_.manager_ctx = new std::shared_ptr<DLPackTensorSharedState>(shared_state_);
    internal_dl_managed_tensor_.deleter = deleteDLManagedTensor;

    // From this point on nothing throws. Hand the shape/strides buffers over
    // to the DLManagedTensor; the deleter will free them via the manager_ctx
    // we just installed.
    internal_dl_managed_tensor_.dl_tensor.shape = shape.release();
    internal_dl_managed_tensor_.dl_tensor.strides = strides.release();
}

// Export constructor - creates tensor from nvimgcodecImageInfo_t (export path).
// Creates shared_state_ to enable multiple DLPack exports and keep data alive.
//
// Coverity's PASS_BY_VALUE checker reports image_buffer is taken by value (sink pattern) 
// so callers can either copy or std::move into this constructor, and the parameter is 
// then std::move'd into shared_state_ in a single forwarding step.
// This is intentional because the std::variant is sized for its largest alternative 
// (nvimgcodecImageInfo_t, ~2168 bytes, above the 512-byte heuristic threshold), but 
// the active alternative for the shared_ptr<...> arms is small and is moved cheaply. 
// If a future maintainer wants to silence the checker structurally, change the parameter 
// to `const ImageBuffer&` and drop the std::move below; the cost trade-off is negligible 
// at current call sites.
// coverity[pass_by_value : INTENTIONAL]
DLPackTensor::DLPackTensor(ILogger* logger, const nvimgcodecImageInfo_t& image_info, 
                          ImageBuffer image_buffer)
    : internal_dl_managed_tensor_{}
    , shared_state_(std::make_shared<DLPackTensorSharedState>(std::move(image_buffer), logger))  // Shared state created - exportable
    , logger_{logger}
{
    DLTensor& tensor = internal_dl_managed_tensor_.dl_tensor;
    // will fill it up before returning from the constructor
    internal_dl_managed_tensor_.manager_ctx = nullptr;
    internal_dl_managed_tensor_.deleter = nullptr;

    // Set up device
    if (image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
        tensor.device.device_type = kDLCUDA;
        if (image_info.buffer == nullptr) {
            throw std::runtime_error("NULL CUDA buffer not accepted");
        }

        cudaPointerAttributes attrs = {};
        cudaError_t err = cudaPointerGetAttributes(&attrs, image_info.buffer);
        cudaGetLastError(); // reset the cuda error (if any)
        if (err != cudaSuccess || attrs.type == cudaMemoryTypeUnregistered) {
            throw std::runtime_error("Buffer is not CUDA-accessible");
        }
        tensor.device.device_id = attrs.device;
    } else if (image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST) {
        tensor.device.device_type = kDLCPU;
        if (image_info.buffer == nullptr) {
            throw std::runtime_error("NULL host buffer not accepted");
        }
    } else {
        throw std::runtime_error("Unsupported buffer type. Buffer type must be CUDA or CPU");
    }

    // Set up ndim
    tensor.ndim = 3; //TODO For now only IRGB

    // Set up data
    tensor.data = image_info.buffer;
    tensor.byte_offset = 0;

    // Set up dtype
    tensor.dtype = type_to_dlpack(image_info.plane_info[0].sample_type);

    // Use sample_format as the sole authority on layout, matching
    // Image::shape / Image::strides / array_interface so all interop
    // protocols report the same shape for a given Image.
    bool is_interleaved = is_sample_format_interleaved(image_info.sample_format);
    int bytes_per_element = sample_type_to_bytes_per_element(image_info.plane_info[0].sample_type);

    // Set up shape (always required) - use unique_ptr for exception safety
    std::unique_ptr<int64_t[]> shape(new int64_t[tensor.ndim]);
    std::unique_ptr<int64_t[]> strides;

    if (is_interleaved) {
        shape[0] = image_info.plane_info[0].height;
        shape[1] = image_info.plane_info[0].width;
        shape[2] = image_info.plane_info[0].num_channels;
        
        // Check if compact (no padding in row stride) - strides can be NULL per DLPack spec
        size_t expected_compact_stride = image_info.plane_info[0].width * 
                                        image_info.plane_info[0].num_channels * 
                                        bytes_per_element;
        if (image_info.plane_info[0].row_stride != expected_compact_stride) {
            // Non-compact - need explicit strides
            strides.reset(new int64_t[tensor.ndim]);
            //dlpack strides of the tensor are in number of elements, not bytes so need to divide by bytes_per_element
            strides[0] = image_info.plane_info[0].row_stride / bytes_per_element;
            strides[1] = image_info.plane_info[0].num_channels;
            strides[2] = 1;
        }
    } else {
        shape[0] = image_info.num_planes;
        shape[1] = image_info.plane_info[0].height;
        shape[2] = image_info.plane_info[0].width;
        
        // Check if compact (no padding in row stride)
        size_t expected_compact_stride = image_info.plane_info[0].width * bytes_per_element;
        if (image_info.plane_info[0].row_stride != expected_compact_stride) {
            // Non-compact - need explicit strides
            strides.reset(new int64_t[tensor.ndim]);
            // dlpack strides of the tensor are in number of elements, not bytes so need to divide by bytes_per_element
            strides[0] = image_info.plane_info[0].row_stride * image_info.plane_info[0].height / bytes_per_element;
            strides[1] = image_info.plane_info[0].row_stride / bytes_per_element;
            strides[2] = 1;
        }
    }

    // Internal tensor stores shared state for lifetime management
    internal_dl_managed_tensor_.manager_ctx = new std::shared_ptr<DLPackTensorSharedState>(shared_state_);
    internal_dl_managed_tensor_.deleter = deleteDLManagedTensor;
    tensor.shape = shape.release();
    tensor.strides = strides.release();
}

DLPackTensor::~DLPackTensor()
{
    if (isInitialized() && internal_dl_managed_tensor_.deleter) {
        internal_dl_managed_tensor_.deleter(&internal_dl_managed_tensor_);
    }
}

const DLTensor* DLPackTensor::operator->() const
{
    return isInitialized() ? &internal_dl_managed_tensor_.dl_tensor : nullptr;
}

DLTensor* DLPackTensor::operator->()
{
    return isInitialized() ? &internal_dl_managed_tensor_.dl_tensor : nullptr;
}

const DLTensor& DLPackTensor::operator*() const
{
    if (!isInitialized()) {
        throw std::runtime_error("Attempted to dereference empty DLPackTensor.");
    }
    return internal_dl_managed_tensor_.dl_tensor;
}

DLTensor& DLPackTensor::operator*()
{
    if (!isInitialized()) {
        throw std::runtime_error("Attempted to dereference empty DLPackTensor.");
    }
    return internal_dl_managed_tensor_.dl_tensor;
}

void DLPackTensor::getImageInfo(nvimgcodecImageInfo_t* image_info,
    std::optional<nvimgcodecSampleFormat_t> sample_format,
    std::optional<nvimgcodecColorSpec_t> color_spec)
{
    if (!isInitialized()) {
        throw std::runtime_error("Cannot get image info from null DLPackTensor");
    }
    
    constexpr int NVIMGCODEC_MAXDIMS = 3; //The maximum number of dimensions allowed in arrays.
    const DLTensor& tensor = internal_dl_managed_tensor_.dl_tensor;
    const int ndim = tensor.ndim;
    if (ndim > NVIMGCODEC_MAXDIMS) {
        throw std::runtime_error("DLPack tensor number of dimensions is higher than the supported maxdims=3");
    }
    if (ndim < 2) {
        // 2-D (H, W) is accepted for single-channel/grayscale images, matching
        // the array-interface as_image path.
        throw std::runtime_error("DLPack tensor number of dimensions is lower than expected at least 2");
    }

    const bool is_cuda = is_cuda_accessible(tensor.device.device_type);
    if (!is_cuda && tensor.device.device_type != kDLCPU) {
        throw std::runtime_error("Unsupported device in DLTensor. Only CPU or CUDA-accessible memory buffers can be wrapped");
    }

    if (tensor.dtype.lanes != 1) {
        throw std::runtime_error("Unsupported lanes in DLTensor dtype.");
    }

    auto sample_type = type_from_dlpack(tensor.dtype);
    int bytes_per_element = sample_type_to_bytes_per_element(sample_type);

    // Layout (HWC vs CHW) and the sample_format / color_spec labels follow the
    // caller's sample_format, resolved by the shared helpers in type_utils.h
    // (the same logic the array-interface as_image path uses). 2-D (H, W) input
    // is accepted for single-channel images.
    std::vector<long> shape;
    shape.reserve(ndim);
    for (int i = 0; i < ndim; ++i)
        shape.push_back(static_cast<long>(tensor.shape[i]));
    const AsImageLayout layout = resolveAsImageLayout(shape, sample_format);
    const bool is_interleaved = layout.is_interleaved;

    if (!is_dlpack_padding_correct(tensor, is_interleaved)) {
        if (dlpack_has_negative_stride(tensor)) {
            throw std::runtime_error("Unexpected DLPack layout. Reversed (negative-stride) views are not supported.");
        }
        throw std::runtime_error("Unexpected DLPack layout. Padding is only allowed for rows. Other dimensions must have contiguous strides.");
    }
    void* buffer = (char*)tensor.data + tensor.byte_offset;
    image_info->num_planes = layout.num_planes;
    image_info->plane_info[0].height = layout.height;
    image_info->plane_info[0].width = layout.width;
    image_info->plane_info[0].num_channels = layout.num_channels;

    const int num_channels =
        is_interleaved ? static_cast<int>(layout.num_channels) : static_cast<int>(layout.num_planes);
    const AsImageLabels labels = resolveAsImageLabels(num_channels, sample_format, color_spec);
    image_info->sample_format = labels.sample_format;
    image_info->color_spec = labels.color_spec;
    image_info->chroma_subsampling = labels.chroma_subsampling;

    // Row-stride axis: HWC and 2-D (H, W) -> axis 0; 3-D CHW planar -> axis 1
    // (rows live inside each plane). DLPack strides are in elements, so convert
    // to bytes; null strides mean the tensor is compact (row-major).
    const int row_stride_axis = (ndim == 3 && !is_interleaved) ? 1 : 0;
    int pitch_in_bytes = (tensor.strides != NULL)
                             ? tensor.strides[row_stride_axis] * bytes_per_element
                             : image_info->plane_info[0].width * image_info->plane_info[0].num_channels * bytes_per_element;
    for (size_t c = 0; c < image_info->num_planes; c++) {
        image_info->plane_info[c].width = image_info->plane_info[0].width;
        image_info->plane_info[c].height = image_info->plane_info[0].height;
        image_info->plane_info[c].row_stride = pitch_in_bytes;
        image_info->plane_info[c].sample_type = sample_type;
        image_info->plane_info[c].num_channels = image_info->plane_info[0].num_channels;
    }
    image_info->buffer = buffer;
    image_info->buffer_kind = is_cuda ? NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE
                                       : NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
}

py::capsule DLPackTensor::getPyCapsule(intptr_t consumer_stream, cudaStream_t producer_stream)
{
    if (!isInitialized()) {
        throw std::runtime_error("Cannot export DLPack tensor: invalid state (default-constructed or moved-from tensor)");
    }

    // Create a new DLManagedTensor for this export to allow multiple DLPack exports
    auto exported_tensor = std::make_unique<DLManagedTensor>();
    exported_tensor->dl_tensor = internal_dl_managed_tensor_.dl_tensor;
    exported_tensor->dl_tensor.shape = nullptr;
    exported_tensor->dl_tensor.strides = nullptr;
    
    // Copy shape and strides arrays (exception-safe via helper)
    auto [shape, strides] = copyShapeAndStrides(internal_dl_managed_tensor_.dl_tensor);
    
    // Wrap manager_ctx in unique_ptr for exception-safe cleanup
    auto manager_ctx_guard = std::make_unique<std::shared_ptr<DLPackTensorSharedState>>(shared_state_);
    
    // Assign pointers to exported tensor (still owned by unique_ptrs).
    // Coverity reports WRAPPER_ESCAPE on the manager_ctx assignment because the raw pointer
    // appears to outlive the unique_ptr; in fact ownership is transferred to the capsule
    // deleter chain via the release() calls below before this function returns.
    exported_tensor->dl_tensor.shape = shape.get();
    exported_tensor->dl_tensor.strides = strides.get();
    // coverity[escape : FALSE]
    exported_tensor->manager_ctx = manager_ctx_guard.get();
    
    // Set deleter for cleanup - calls helper then deletes the DLManagedTensor itself
    exported_tensor->deleter = [](DLManagedTensor* self) {
        deleteDLManagedTensor(self);
        delete self;
    };

    // Create capsule with custom deleter that handles exceptions per DLPack spec
    py::capsule cap(exported_tensor.get(), "dltensor", [](PyObject* ptr) {
        // Check if renamed to "used_dltensor" - if so, consumer owns it and we do nothing
        if (PyCapsule_IsValid(ptr, "used_dltensor")) {
            return;
        }
        
        // Save any in-flight exception (as per DLPack spec)
        PyObject *type, *value, *traceback;
        PyErr_Fetch(&type, &value, &traceback);
        
        if (PyCapsule_IsValid(ptr, "dltensor")) {
            DLManagedTensor* dlTensor = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(ptr, "dltensor"));
            if (dlTensor == nullptr) {
                PyErr_WriteUnraisable(ptr);
            } else {
                if (dlTensor->deleter != nullptr) {
                    dlTensor->deleter(dlTensor);
                }
            }
        }
        
        // Restore exception state
        PyErr_Restore(type, value, traceback);
    });
    
    // Capsule successfully created - now release ownership from all smart pointers.
    // Coverity flags these release() calls as RESOURCE_LEAK because it cannot follow
    // the ownership chain through exported_tensor->deleter -> deleteDLManagedTensor;
    // each released pointer is in fact already aliased into exported_tensor's fields
    // (or, for exported_tensor itself, owned by the py::capsule constructed above) and
    // will be freed when the capsule's deleter runs.
    // coverity[leaked_storage : FALSE]
    shape.release();
    // coverity[leaked_storage : FALSE]
    strides.release();
    // coverity[leaked_storage : FALSE]
    manager_ctx_guard.release();
    // coverity[leaked_storage : FALSE]
    exported_tensor.release();

    // Add synchronisation
    static constexpr intptr_t kDoNotSync = -1; // if provided stream is -1, no stream order should be established;
    if (consumer_stream != kDoNotSync) {
        if (!shared_state_->dlpack_cuda_event) {
            cudaEvent_t event;
            CHECK_CUDA(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            // Capture logger with the name expected by CHECK_CUDA_LOG macro
            shared_state_->dlpack_cuda_event = std::shared_ptr<std::remove_pointer<cudaEvent_t>::type>(
                event, [logger_ = shared_state_->logger](cudaEvent_t e) { CHECK_CUDA_LOG(cudaEventDestroy(e)); });
        }
        // the consumer stream should wait for the work on Image stream
        auto cu_consumer_stream = reinterpret_cast<cudaStream_t>(consumer_stream);
        if (cu_consumer_stream != producer_stream) {
            CHECK_CUDA(cudaEventRecord(shared_state_->dlpack_cuda_event.get(), producer_stream));
            CHECK_CUDA(cudaStreamWaitEvent(cu_consumer_stream, shared_state_->dlpack_cuda_event.get()));
        }
    }

    return cap;
}

} // namespace nvimgcodec
