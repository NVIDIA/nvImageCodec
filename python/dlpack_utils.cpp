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

namespace nvimgcodec {

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

DLPackTensor::DLPackTensor() noexcept
    : internal_dl_managed_tensor_{}
{
}


DLPackTensor::DLPackTensor(DLManagedTensor* dl_managed_tensor)
    : internal_dl_managed_tensor_{}
    , dl_managed_tensor_ptr_{dl_managed_tensor}
{
}

DLPackTensor::DLPackTensor(const nvimgcodecImageInfo_t& image_info, std::shared_ptr<unsigned char> image_buffer)
    : internal_dl_managed_tensor_{}
    , dl_managed_tensor_ptr_{&internal_dl_managed_tensor_}
    , image_buffer_(image_buffer)
{
    internal_dl_managed_tensor_.manager_ctx = this;
    internal_dl_managed_tensor_.deleter = [](DLManagedTensor* self) {
        delete[] self->dl_tensor.shape;
        delete[] self->dl_tensor.strides;
    };

    try {
        DLTensor& tensor = internal_dl_managed_tensor_.dl_tensor;

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

        // Set up shape and strides
        tensor.shape = new int64_t[tensor.ndim];
        tensor.strides = new int64_t[tensor.ndim];

        if (is_interleaved) {
            tensor.shape[0] = image_info.plane_info[0].height;
            tensor.shape[1] = image_info.plane_info[0].width;
            tensor.shape[2] = image_info.plane_info[0].num_channels;
            //dlpack strides of the tensor are in number of elements, not bytes so need to divide by bytes_per_element
            tensor.strides[0] = image_info.plane_info[0].row_stride / bytes_per_element;
            tensor.strides[1] = image_info.plane_info[0].num_channels /* * bytes_per_element*/;
            tensor.strides[2] = /*bytes_per_element*/1;
        } else {
            tensor.shape[0] = image_info.num_planes;
            tensor.shape[1] = image_info.plane_info[0].height;
            tensor.shape[2] = image_info.plane_info[0].width;
            // dlpack strides of the tensor are in number of elements, not bytes so need to divide by bytes_per_element tensor.strides[0] =
            tensor.strides[0] = image_info.plane_info[0].row_stride * image_info.plane_info[0].height / bytes_per_element;
            tensor.strides[1] = image_info.plane_info[0].row_stride / bytes_per_element;
            tensor.strides[2] = /*bytes_per_element*/ 1;
        }
    } catch (...) {
        internal_dl_managed_tensor_.deleter(&internal_dl_managed_tensor_);
        throw;
    }
}

DLPackTensor::~DLPackTensor()
{
    if (dl_managed_tensor_ptr_ && dl_managed_tensor_ptr_->deleter) {
        dl_managed_tensor_ptr_->deleter(dl_managed_tensor_ptr_);
    }
}

const DLTensor* DLPackTensor::operator->() const
{
    return &internal_dl_managed_tensor_.dl_tensor;
}

DLTensor* DLPackTensor::operator->()
{
    return &internal_dl_managed_tensor_.dl_tensor;
}

const DLTensor& DLPackTensor::operator*() const
{
    return internal_dl_managed_tensor_.dl_tensor;
}

DLTensor& DLPackTensor::operator*()
{
    return internal_dl_managed_tensor_.dl_tensor;
}

void DLPackTensor::getImageInfo(nvimgcodecImageInfo_t* image_info)
{
    constexpr int NVIMGCODEC_MAXDIMS = 3; //The maximum number of dimensions allowed in arrays.
    const int ndim = dl_managed_tensor_ptr_->dl_tensor.ndim;
    if (ndim > NVIMGCODEC_MAXDIMS) {
        throw std::runtime_error("DLPack tensor number of dimensions is higher than the supported maxdims=3");
    }
    if (ndim < 3) {
        throw std::runtime_error("DLPack tensor number of dimension is lower than expected at least 3");
    }

    if (!is_cuda_accessible(dl_managed_tensor_ptr_->dl_tensor.device.device_type)) {
        throw std::runtime_error("Unsupported device in DLTensor. Only CUDA-accessible memory buffers can be wrapped");
    }

    if (dl_managed_tensor_ptr_->dl_tensor.dtype.lanes != 1) {
        throw std::runtime_error("Unsupported lanes in DLTensor dtype.");
    }

    auto sample_type = type_from_dlpack(dl_managed_tensor_ptr_->dl_tensor.dtype);
    int bytes_per_element = sample_type_to_bytes_per_element(sample_type);

    bool is_interleaved = true; // For now always assume interleaved
    void* buffer = (char*)dl_managed_tensor_ptr_->dl_tensor.data + dl_managed_tensor_ptr_->dl_tensor.byte_offset;
    if (is_interleaved) {
        image_info->num_planes = 1;
        image_info->plane_info[0].height = dl_managed_tensor_ptr_->dl_tensor.shape[0];
        image_info->plane_info[0].width = dl_managed_tensor_ptr_->dl_tensor.shape[1];
        image_info->plane_info[0].num_channels = dl_managed_tensor_ptr_->dl_tensor.shape[2];
    } else {
        image_info->num_planes = dl_managed_tensor_ptr_->dl_tensor.shape[0];
        image_info->plane_info[0].height = dl_managed_tensor_ptr_->dl_tensor.shape[1];
        image_info->plane_info[0].width = dl_managed_tensor_ptr_->dl_tensor.shape[2];
        image_info->plane_info[0].num_channels = 1;
    }

    image_info->color_spec = NVIMGCODEC_COLORSPEC_SRGB;
    image_info->sample_format = is_interleaved ? NVIMGCODEC_SAMPLEFORMAT_I_RGB : NVIMGCODEC_SAMPLEFORMAT_P_RGB;
    image_info->chroma_subsampling = NVIMGCODEC_SAMPLING_444;

    int pitch_in_bytes = dl_managed_tensor_ptr_->dl_tensor.strides != NULL && dl_managed_tensor_ptr_->dl_tensor.strides
                             ?
                             //dlpack strides of the tensor are in number of elements, not bytes so need to multiple by bytes_per_element
                             (is_interleaved ? dl_managed_tensor_ptr_->dl_tensor.strides[0] * bytes_per_element
                                             : dl_managed_tensor_ptr_->dl_tensor.strides[1] * bytes_per_element)
                             //can be NULL, indicating tensor is compact and row - majored
                             : image_info->plane_info[0].width * image_info->plane_info[0].num_channels * bytes_per_element;
    size_t buffer_size = 0;
    for (size_t c = 0; c < image_info->num_planes; c++) {
        image_info->plane_info[c].width = image_info->plane_info[0].width;
        image_info->plane_info[c].height = image_info->plane_info[0].height;
        image_info->plane_info[c].row_stride = pitch_in_bytes;
        image_info->plane_info[c].sample_type = sample_type;
        image_info->plane_info[c].num_channels = image_info->plane_info[0].num_channels;
        buffer_size += image_info->plane_info[c].row_stride * image_info->plane_info[0].height;
    }
    image_info->buffer = buffer;
    image_info->buffer_size = buffer_size;
    image_info->buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
}

py::capsule DLPackTensor::getPyCapsule()
{
    // When ownership was already taken
    if (dl_managed_tensor_ptr_ == nullptr) {
        return py::capsule();
    }

    // Creates the python capsule with the DLManagedTensor instance we're returning.
    py::capsule cap(dl_managed_tensor_ptr_, "dltensor", [](PyObject* ptr) {
        if (PyCapsule_IsValid(ptr, "dltensor")) {
            // If consumer didn't delete the tensor,
            if (auto* dlTensor = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(ptr, "dltensor"))) {
                // Delete the tensor.
                if (dlTensor->deleter != nullptr) {
                    dlTensor->deleter(dlTensor);
                }
            }
        }
    });
    dl_managed_tensor_ptr_ = nullptr;
    return cap;
}

} // namespace nvimgcodec
