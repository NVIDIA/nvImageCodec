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
    
    // Copy shape and strides arrays (exception-safe via helper)
    auto [shape, strides] = copyShapeAndStrides(dl_managed_tensor->dl_tensor);
    internal_dl_managed_tensor_.dl_tensor.shape = shape.get();
    internal_dl_managed_tensor_.dl_tensor.strides = strides.get();
    
    // Set up manager_ctx and deleter for our internal tensor
    internal_dl_managed_tensor_.manager_ctx = new std::shared_ptr<DLPackTensorSharedState>(shared_state_);
    internal_dl_managed_tensor_.deleter = deleteDLManagedTensor;

    // we can now release the unique_ptrs as they are owned by the internal_dl_managed_tensor_
    shape.release();
    strides.release();
}

// Export constructor - creates tensor from nvimgcodecImageInfo_t (export path)
// Creates shared_state_ to enable multiple DLPack exports and keep data alive
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

    bool is_interleaved = is_sample_format_interleaved(image_info.sample_format) || image_info.num_planes == 1;
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

void DLPackTensor::getImageInfo(nvimgcodecImageInfo_t* image_info)
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
    if (ndim < 3) {
        throw std::runtime_error("DLPack tensor number of dimension is lower than expected at least 3");
    }

    if (!is_cuda_accessible(tensor.device.device_type)) {
        throw std::runtime_error("Unsupported device in DLTensor. Only CUDA-accessible memory buffers can be wrapped");
    }

    if (tensor.dtype.lanes != 1) {
        throw std::runtime_error("Unsupported lanes in DLTensor dtype.");
    }

    auto sample_type = type_from_dlpack(tensor.dtype);
    int bytes_per_element = sample_type_to_bytes_per_element(sample_type);

    bool is_interleaved = true; // For now always assume interleaved
    void* buffer = (char*)tensor.data + tensor.byte_offset;
    if (is_interleaved) {
        image_info->num_planes = 1;
        image_info->plane_info[0].height = tensor.shape[0];
        image_info->plane_info[0].width = tensor.shape[1];
        image_info->plane_info[0].num_channels = tensor.shape[2];
    } else {
        image_info->num_planes = tensor.shape[0];
        image_info->plane_info[0].height = tensor.shape[1];
        image_info->plane_info[0].width = tensor.shape[2];
        image_info->plane_info[0].num_channels = 1;
    }

    image_info->color_spec = NVIMGCODEC_COLORSPEC_SRGB;
    image_info->sample_format = is_interleaved ? NVIMGCODEC_SAMPLEFORMAT_I_RGB : NVIMGCODEC_SAMPLEFORMAT_P_RGB;
    image_info->chroma_subsampling = NVIMGCODEC_SAMPLING_444;

    int pitch_in_bytes = (tensor.strides != NULL)
                             ?
                             //dlpack strides of the tensor are in number of elements, not bytes so need to multiple by bytes_per_element
                             (is_interleaved ? tensor.strides[0] * bytes_per_element
                                             : tensor.strides[1] * bytes_per_element)
                             //can be NULL, indicating tensor is compact and row - majored
                             : image_info->plane_info[0].width * image_info->plane_info[0].num_channels * bytes_per_element;
    for (size_t c = 0; c < image_info->num_planes; c++) {
        image_info->plane_info[c].width = image_info->plane_info[0].width;
        image_info->plane_info[c].height = image_info->plane_info[0].height;
        image_info->plane_info[c].row_stride = pitch_in_bytes;
        image_info->plane_info[c].sample_type = sample_type;
        image_info->plane_info[c].num_channels = image_info->plane_info[0].num_channels;
    }
    image_info->buffer = buffer;
    image_info->buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
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
    
    // Assign pointers to exported tensor (still owned by unique_ptrs)
    exported_tensor->dl_tensor.shape = shape.get();
    exported_tensor->dl_tensor.strides = strides.get();
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
    
    // Capsule successfully created - now release ownership from all smart pointers
    shape.release();
    strides.release();
    manager_ctx_guard.release();
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
