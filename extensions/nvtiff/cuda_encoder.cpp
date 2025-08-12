/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#define NOMINMAX
#include "cuda_encoder.h"

#include <library_types.h>
#include <nvimgcodec.h>
#include <nvjpeg.h>
#include <nvtiff.h>
#include <array>
#include <cstring>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <nvtx3/nvtx3.hpp>
#include <optional>
#include <set>
#include <vector>
#include "error_handling.h"
#include "imgproc/convert_kernel_gpu.h"
#include "imgproc/device_buffer.h"
#include "imgproc/pinned_buffer.h"
#include "imgproc/device_guard.h"
#include "imgproc/sample_format_utils.h"
#include "imgproc/stream_device.h"
#include "imgproc/type_utils.h"
#include "log.h"

using nvimgcodec::PinnedBuffer;

namespace nvtiff {

struct Encoder
{
    const char* plugin_id_ = nullptr;
    const nvimgcodecFrameworkDesc_t* framework_ = nullptr;
    const nvimgcodecExecutionParams_t* exec_params_ = nullptr;

    Encoder(const char* plugin_id, 
            const nvimgcodecFrameworkDesc_t* framework, 
            const nvimgcodecExecutionParams_t* exec_params,
            const char* options = nullptr);

    ~Encoder() = default;

    nvimgcodecProcessingStatus_t
    check_image_info(const nvimgcodecImageInfo_t& img_info,
                     const nvimgcodecEncodeParams_t& params,
                     const nvimgcodecImageInfo_t& out_img_info);

    static nvimgcodecStatus_t static_destroy(nvimgcodecEncoder_t encoder);

    nvimgcodecProcessingStatus_t 
    canEncode(const nvimgcodecCodeStreamDesc_t* code_stream,
              const nvimgcodecImageDesc_t* image, 
              const nvimgcodecEncodeParams_t* params, 
              int thread_idx);

    static nvimgcodecProcessingStatus_t 
    static_can_encode(nvimgcodecEncoder_t encoder, 
                      const nvimgcodecCodeStreamDesc_t* code_stream, 
                      const nvimgcodecImageDesc_t* image,
                      const nvimgcodecEncodeParams_t* params, 
                      int thread_idx);

    nvimgcodecStatus_t
    encode(const nvimgcodecCodeStreamDesc_t* code_stream, 
           const nvimgcodecImageDesc_t* image,
           const nvimgcodecEncodeParams_t* params, 
           int thread_idx);

    static nvimgcodecStatus_t 
    static_encode_sample(nvimgcodecEncoder_t encoder, 
                         const nvimgcodecCodeStreamDesc_t* code_stream, 
                         const nvimgcodecImageDesc_t* image,
                         const nvimgcodecEncodeParams_t* params, 
                         int thread_idx);


    struct PerThreadResources
    {
        const nvimgcodecFrameworkDesc_t* framework_ = {};
        const char* plugin_id_ = {};
        nvtiffEncodeParams_t encode_params_ = {};
        nvtiffEncoder_t encoder_ = {};
        nvtiffImageInfo_t tiff_info_ = {};
        std::optional<cudaStream_t> cuda_stream_ = {};
        nvtiffDeviceAllocator_t device_allocator_ = {};
        nvtiffPinnedAllocator_t pinned_allocator_ = {};
        nvtiffDeviceAllocator_t* device_allocator_ptr_ = {};
        nvtiffPinnedAllocator_t* pinned_allocator_ptr_ = {};
        PinnedBuffer pinned_buffer_;

        PerThreadResources(const nvimgcodecFrameworkDesc_t* framework, 
                           const char* plugin_id,
                           const nvimgcodecExecutionParams_t* exec_params);
        ~PerThreadResources();

        nvtiffEncoder_t& encoder(cudaStream_t cuda_stream);
    };

    std::vector<PerThreadResources> per_thread_;
};

Encoder::
PerThreadResources::PerThreadResources(const nvimgcodecFrameworkDesc_t* framework, 
                                       const char* plugin_id,
                                       const nvimgcodecExecutionParams_t* exec_params)
    : framework_(framework)
    , plugin_id_(plugin_id)
    , pinned_buffer_(exec_params) 
{
    XM_CHECK_NVTIFF(nvtiffEncodeParamsCreate(&encode_params_));

    if (exec_params->device_allocator) {
        device_allocator_ = {
            exec_params->device_allocator->device_malloc,
            exec_params->device_allocator->device_free,
            exec_params->device_allocator->device_ctx
        };
        device_allocator_ptr_ = &device_allocator_;
    }

    if (exec_params->pinned_allocator) {
        pinned_allocator_ = {
            exec_params->pinned_allocator->pinned_malloc,
            exec_params->pinned_allocator->pinned_free,
            exec_params->pinned_allocator->pinned_ctx            
        };
        pinned_allocator_ptr_ = &pinned_allocator_;
    }
}

Encoder::
PerThreadResources::~PerThreadResources()
{
    cudaStream_t stream = cuda_stream_ ? cuda_stream_.value() : nullptr;
    if (stream) {
        XM_CUDA_LOG_DESTROY(cudaStreamSynchronize(stream));
        XM_NVTIFF_LOG_DESTROY(nvtiffEncodeParamsDestroy(encode_params_, stream));
        XM_NVTIFF_LOG_DESTROY(nvtiffEncoderDestroy(encoder_, stream));
    }
}

nvtiffEncoder_t& Encoder::
PerThreadResources::encoder(cudaStream_t stream) {
    if (!cuda_stream_) {
        NVIMGCODEC_LOG_INFO(framework_, plugin_id_, "Creating  nvtiff encoder instance");
        XM_CHECK_NVTIFF(nvtiffEncoderCreate(&encoder_, device_allocator_ptr_, 
                                            pinned_allocator_ptr_, stream));
        cuda_stream_ = stream;
    }
    return encoder_;
}

NvTiffCudaEncoderPlugin::NvTiffCudaEncoderPlugin(const nvimgcodecFrameworkDesc_t* framework)
    : encoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_ENCODER_DESC, // struct_type
                    sizeof(nvimgcodecEncoderDesc_t), // struct_size
                    nullptr, // struct_next 
                    this, // instance
                    plugin_id_, // id
                    "tiff", // codec name
                    NVIMGCODEC_BACKEND_KIND_GPU_ONLY, // backend_kind
                    static_create, // create function
                    Encoder::static_destroy, // destroy function
                    Encoder::static_can_encode, // can encode function
                    Encoder::static_encode_sample } // encode function
    , framework_(framework)
{}

nvimgcodecEncoderDesc_t* 
NvTiffCudaEncoderPlugin::getEncoderDesc()
{
    return &encoder_desc_;
}

nvimgcodecStatus_t 
NvTiffCudaEncoderPlugin::create(nvimgcodecEncoder_t* encoder, 
                                const nvimgcodecExecutionParams_t* exec_params, 
                                const char* options)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvtiff_create_encoder");
        XM_CHECK_NULL(encoder);
        XM_CHECK_NULL(exec_params);
        if (exec_params->device_id == NVIMGCODEC_DEVICE_CPU_ONLY) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }

        *encoder = reinterpret_cast<nvimgcodecEncoder_t>(new Encoder(plugin_id_, framework_, exec_params, options));
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create nvtiff encoder - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t 
NvTiffCudaEncoderPlugin::static_create(void* instance, 
                                   nvimgcodecEncoder_t* encoder, 
                                   const nvimgcodecExecutionParams_t* exec_params, 
                                   const char* options)
{
    if (!instance) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }

    auto handle = reinterpret_cast<NvTiffCudaEncoderPlugin*>(instance);
    return handle->create(encoder, exec_params, options);
}

Encoder::Encoder(const char* plugin_id, 
                 const nvimgcodecFrameworkDesc_t* framework, 
                 const nvimgcodecExecutionParams_t* exec_params, 
                 const char* options)
    : plugin_id_(plugin_id)
    , framework_(framework)
    , exec_params_(exec_params)
{
    auto executor = exec_params_->executor;
    size_t num_threads = executor->getNumThreads(executor->instance);

    per_thread_.reserve(num_threads);
    while (per_thread_.size() < num_threads) {
        per_thread_.emplace_back(framework_, plugin_id_, exec_params_);
    }
}

nvimgcodecStatus_t 
Encoder::static_destroy(nvimgcodecEncoder_t encoder)
{
    try {
        XM_CHECK_NULL(encoder);
        delete reinterpret_cast<Encoder*>(encoder);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

// TODO(duong): merge this with the set_image_info function, maybe keeping img_info at class level
// and store a flag to indicate whether the info has been set
nvimgcodecProcessingStatus_t
Encoder::check_image_info(const nvimgcodecImageInfo_t& img_info,
                          const nvimgcodecEncodeParams_t& params,
                          const nvimgcodecImageInfo_t& out_img_info)
{
    nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;

    // Check codec name
    if (strcmp(out_img_info.codec_name, "tiff") != 0) {
        return NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
    }

    // Check number of planes
    if (img_info.num_planes != 1) {
        status |= NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
    }

    // Check color spec
    if (img_info.color_spec != NVIMGCODEC_COLORSPEC_UNKNOWN &&
        img_info.color_spec != NVIMGCODEC_COLORSPEC_SRGB &&
        img_info.color_spec != NVIMGCODEC_COLORSPEC_GRAY &&
        img_info.color_spec != NVIMGCODEC_COLORSPEC_SYCC) {
        status |= NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
    }

    // Check chroma subsampling and sample format
    if (img_info.color_spec == NVIMGCODEC_COLORSPEC_GRAY) {
        if (img_info.chroma_subsampling != NVIMGCODEC_SAMPLING_GRAY &&
            img_info.chroma_subsampling != NVIMGCODEC_SAMPLING_NONE) {
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
        }
        if (img_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_P_Y &&
            img_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED) {
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
        }
    } else if (img_info.color_spec == NVIMGCODEC_COLORSPEC_SRGB || 
               img_info.color_spec == NVIMGCODEC_COLORSPEC_SYCC) {
        if (img_info.chroma_subsampling != NVIMGCODEC_SAMPLING_NONE) {
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
        }
        if (img_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_I_RGB &&
            img_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_I_BGR &&
            img_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED) {
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
        }
    } else { // NVIMGCODEC_COLORSPEC_UNKNOWN
        if (img_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_RGB ||
            img_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_BGR) {
            if (img_info.chroma_subsampling != NVIMGCODEC_SAMPLING_NONE) {
                status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
            }
        } else if (img_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_Y ||
                   img_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED) {
            if (img_info.chroma_subsampling != NVIMGCODEC_SAMPLING_GRAY &&
                img_info.chroma_subsampling != NVIMGCODEC_SAMPLING_NONE) {
                status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
            }
        } else {
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
        }
    }
    if (img_info.chroma_subsampling != out_img_info.chroma_subsampling) {
        status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
    }

    // Check number of channels
    const nvimgcodecImagePlaneInfo_t& plane = img_info.plane_info[0];
    if (plane.num_channels != 1 && plane.num_channels != 3) {
        status |= NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
    }

    // Check sample type
    if (plane.sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_UNKNOWN ||
        plane.sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_UNSUPPORTED) {
        status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
    }

    // The >> 11 gives the number of bytes for the type
    if (plane.row_stride != plane.width * (plane.sample_type >> 11) * plane.num_channels) {
        NVIMGCODEC_LOG_WARNING(framework_, plugin_id_,
                               "Row stride not the same as width * sample_size * num_channels");
        status |= NVIMGCODEC_PROCESSING_STATUS_ENCODING_UNSUPPORTED;
    }

    // Check quality type and value
    if (params.quality_type != NVIMGCODEC_QUALITY_TYPE_DEFAULT &&
        params.quality_type != NVIMGCODEC_QUALITY_TYPE_LOSSLESS) {
        status |= NVIMGCODEC_PROCESSING_STATUS_QUALITY_TYPE_UNSUPPORTED;
    }
    if (params.quality_type == NVIMGCODEC_QUALITY_TYPE_LOSSLESS) {
        if (params.quality_value != 0 && params.quality_value != 1) {
            status |= NVIMGCODEC_PROCESSING_STATUS_QUALITY_VALUE_UNSUPPORTED;
        }
    }

    // NOTE: nvtiff doesn't care about precision

    return status;
}

nvimgcodecProcessingStatus_t 
Encoder::canEncode(const nvimgcodecCodeStreamDesc_t* code_stream,
                   const nvimgcodecImageDesc_t* image, 
                   const nvimgcodecEncodeParams_t* params, 
                   int thread_idx)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvtiff_can_encode");
        XM_CHECK_NULL(code_stream);
        XM_CHECK_NULL(params);
        XM_CHECK_NULL(image);

        nvimgcodecImageInfo_t img_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO,
                                       sizeof(nvimgcodecImageInfo_t), 0};
        if (NVIMGCODEC_STATUS_SUCCESS != image->getImageInfo(image->instance, &img_info)) {
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
        }

        nvimgcodecImageInfo_t out_img_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO,
                                           sizeof(nvimgcodecImageInfo_t), 0};
        if (NVIMGCODEC_STATUS_SUCCESS != code_stream->getImageInfo(code_stream->instance, &out_img_info)) {
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
        }

        return check_image_info(img_info, *params, out_img_info);
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if nvtiff can encode " << e.what());
        return NVIMGCODEC_PROCESSING_STATUS_FAIL;
    }
    return NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
}

nvimgcodecProcessingStatus_t 
Encoder::static_can_encode(nvimgcodecEncoder_t encoder, 
                           const nvimgcodecCodeStreamDesc_t* code_stream, 
                           const nvimgcodecImageDesc_t* image,
                           const nvimgcodecEncodeParams_t* params,
                           int thread_idx)
{
    try {
        XM_CHECK_NULL(encoder);
        return reinterpret_cast<Encoder*>(encoder)->canEncode(code_stream, image, params, thread_idx);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_PROCESSING_STATUS_FAIL;
    }
    return NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
}

static void 
set_tiff_image_info(const nvimgcodecImageInfo_t& img_info,
                    const nvimgcodecEncodeParams_t& params, 
                    nvtiffImageInfo_t* tiff_info)
{
    // NOTE: As of version 0.5, nvTIFF only supports single plane, interleaved format
    tiff_info->samples_per_pixel = img_info.plane_info[0].num_channels;
    tiff_info->compression = NVTIFF_COMPRESSION_LZW;
    if (tiff_info->samples_per_pixel == 1) {
        tiff_info->photometric_int = NVTIFF_PHOTOMETRIC_MINISBLACK;
    } else if (tiff_info->samples_per_pixel == 3) {
        tiff_info->photometric_int = NVTIFF_PHOTOMETRIC_RGB;
    }
    tiff_info->image_width = img_info.plane_info[0].width;
    tiff_info->image_height = img_info.plane_info[0].height;
    tiff_info->bits_per_pixel = 0;
    for (int c = 0; c < tiff_info->samples_per_pixel; ++c) {
        switch (img_info.plane_info[0].sample_type) {
        case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8:
            tiff_info->bits_per_sample[c] = 8;
            tiff_info->sample_format[c] = NVTIFF_SAMPLEFORMAT_UINT;
            break;
        case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16:
            tiff_info->bits_per_sample[c] = 16;
            tiff_info->sample_format[c] = NVTIFF_SAMPLEFORMAT_UINT;
            break;
        case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT32:
            tiff_info->bits_per_sample[c] = 32;
            tiff_info->sample_format[c] = NVTIFF_SAMPLEFORMAT_UINT;
            break;
        case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT64:
            tiff_info->bits_per_sample[c] = 64;
            tiff_info->sample_format[c] = NVTIFF_SAMPLEFORMAT_UINT;
            break;
        case NVIMGCODEC_SAMPLE_DATA_TYPE_INT8:
            tiff_info->bits_per_sample[c] = 8;
            tiff_info->sample_format[c] = NVTIFF_SAMPLEFORMAT_INT;
            break;
        case NVIMGCODEC_SAMPLE_DATA_TYPE_INT16:
            tiff_info->bits_per_sample[c] = 16;
            tiff_info->sample_format[c] = NVTIFF_SAMPLEFORMAT_INT;
            break;
        case NVIMGCODEC_SAMPLE_DATA_TYPE_INT32:
            tiff_info->bits_per_sample[c] = 32;
            tiff_info->sample_format[c] = NVTIFF_SAMPLEFORMAT_INT;
            break;
        case NVIMGCODEC_SAMPLE_DATA_TYPE_INT64:
            tiff_info->bits_per_sample[c] = 64;
            tiff_info->sample_format[c] = NVTIFF_SAMPLEFORMAT_INT;
            break;
        case NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32:
            tiff_info->bits_per_sample[c] = 32;
            tiff_info->sample_format[c] = NVTIFF_SAMPLEFORMAT_IEEEFP;
            break;
        case NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT64:
            tiff_info->bits_per_sample[c] = 64;
            tiff_info->sample_format[c] = NVTIFF_SAMPLEFORMAT_IEEEFP;
            break;
        default:
            tiff_info->bits_per_sample[c] = 8;
            tiff_info->sample_format[c] = NVTIFF_SAMPLEFORMAT_UNKNOWN;
            break;
        };
        tiff_info->bits_per_pixel += tiff_info->bits_per_sample[c];
    }

    tiff_info->planar_config = NVTIFF_PLANARCONFIG_CONTIG;

    if (params.quality_type == NVIMGCODEC_QUALITY_TYPE_DEFAULT) {
        tiff_info->compression = NVTIFF_COMPRESSION_LZW;
    } else if (params.quality_type == NVIMGCODEC_QUALITY_TYPE_LOSSLESS) {
        tiff_info->compression = params.quality_value == 0 ? NVTIFF_COMPRESSION_NONE : NVTIFF_COMPRESSION_LZW;
    } else {
        // Can't happen. We checked this in canEncode.
        throw std::runtime_error("Unsupported quality type should have been checked."); 
    }
}

nvimgcodecStatus_t
Encoder::encode(const nvimgcodecCodeStreamDesc_t* code_stream, 
                const nvimgcodecImageDesc_t* image,
                const nvimgcodecEncodeParams_t* params, 
                int thread_idx)
{
    try {
        XM_CHECK_NULL(image);
        PerThreadResources& t = per_thread_[thread_idx];
        nvimgcodecImageInfo_t img_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO,
                                       sizeof(nvimgcodecImageInfo_t), 0};
        auto ret = image->getImageInfo(image->instance, &img_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
            return ret;
        }
        set_tiff_image_info(img_info, *params, &t.tiff_info_);
        XM_CHECK_NVTIFF(nvtiffEncodeParamsSetImageInfo(t.encode_params_, &t.tiff_info_));
        XM_CHECK_NVTIFF(nvtiffEncodeParamsSetInputs(t.encode_params_, (uint8_t**)&img_info.buffer, 1));
        nvtiffEncoder_t& encoder = t.encoder(img_info.cuda_stream);
        XM_CHECK_NVTIFF(nvtiffEncode(encoder, &t.encode_params_, 1, img_info.cuda_stream));
        XM_CHECK_NVTIFF(nvtiffEncodeFinalize(encoder, &t.encode_params_, 1, img_info.cuda_stream));

        // write the bitstream to cpu buffer
        size_t metadata_size = 0, bitstream_size = 0;
        XM_CHECK_NVTIFF(nvtiffGetBitstreamSize(encoder, &t.encode_params_, 1, &metadata_size, &bitstream_size));
        size_t total_size = metadata_size + bitstream_size;
        t.pinned_buffer_.resize(total_size, img_info.cuda_stream);
        XM_CHECK_NVTIFF(nvtiffWriteTiffBuffer(encoder, &t.encode_params_, 1, (uint8_t*)t.pinned_buffer_.data, total_size, img_info.cuda_stream));
        nvimgcodecIoStreamDesc_t* io_stream = code_stream->io_stream;
        io_stream->reserve(io_stream->instance, total_size);
        io_stream->seek(io_stream->instance, 0, SEEK_SET);
        size_t output_size = 0;
        io_stream->write(io_stream->instance, &output_size, t.pinned_buffer_.data, t.pinned_buffer_.size);
        io_stream->flush(io_stream->instance);
        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXECUTION_FAILED;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t 
Encoder::static_encode_sample(nvimgcodecEncoder_t encoder, 
                     const nvimgcodecCodeStreamDesc_t* code_stream, 
                     const nvimgcodecImageDesc_t* image,
                     const nvimgcodecEncodeParams_t* params, 
                     int thread_idx)
{
    try {
        XM_CHECK_NULL(encoder);
        return reinterpret_cast<Encoder*>(encoder)->encode(code_stream, image, params, thread_idx);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXECUTION_FAILED;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

} // namespace nvjpeg
