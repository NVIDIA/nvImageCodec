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

#include <nvimgcodec.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "encoder.h"
#include "error_handling.h"
#include "log.h"
#include "imgproc/image_info_checks.h"
#include "imgproc/safe_arithmetic.h"
#include "imgproc/type_utils.h"

namespace nvpnm {

struct EncodeState
{
    struct Sample
    {
        nvimgcodecCodeStreamDesc_t* code_stream;
        nvimgcodecImageDesc_t* image;
        const nvimgcodecEncodeParams_t* params;
    };
    const char* plugin_id_;
    const nvimgcodecFrameworkDesc_t* framework_;
    std::vector<Sample> samples_;
};

struct EncoderImpl
{
    EncoderImpl(
        const char* id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params, const char* options);
    ~EncoderImpl();

    static nvimgcodecStatus_t static_destroy(nvimgcodecEncoder_t encoder);

    nvimgcodecProcessingStatus_t canEncode(const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecImageDesc_t* image,
        const nvimgcodecEncodeParams_t* params, int thread_idx);
    static nvimgcodecProcessingStatus_t static_can_encode(nvimgcodecEncoder_t encoder, const nvimgcodecCodeStreamDesc_t* code_stream,
        const nvimgcodecImageDesc_t* image, const nvimgcodecEncodeParams_t* params, int thread_idx)
    {
        try {
            XM_CHECK_NULL(encoder);
            auto handle = reinterpret_cast<EncoderImpl*>(encoder);
            return handle->canEncode(code_stream, image, params, thread_idx);
        } catch (const std::exception& e) {
            return NVIMGCODEC_PROCESSING_STATUS_FAIL;
        }
    }

    nvimgcodecStatus_t encode(
        const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecImageDesc_t* image, const nvimgcodecEncodeParams_t* params, int thread_idx);
    static nvimgcodecStatus_t static_encode_sample(nvimgcodecEncoder_t encoder, const nvimgcodecCodeStreamDesc_t* code_stream,
        const nvimgcodecImageDesc_t* image, const nvimgcodecEncodeParams_t* params, int thread_idx)
    {
        try {
            XM_CHECK_NULL(encoder);
            auto handle = reinterpret_cast<EncoderImpl*>(encoder);
            return handle->encode(code_stream, image, params, thread_idx);
        } catch (const std::exception& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

    const char* plugin_id_;
    const nvimgcodecFrameworkDesc_t* framework_;
    std::unique_ptr<EncodeState> encode_state_batch_;
    const nvimgcodecExecutionParams_t* exec_params_;
    std::string options_;
};

// The component count this encoder accepts, matching what canEncode() has always admitted.
constexpr uint32_t kNumComponents = 3;

// Plane layout the writer requires. Shared by canEncode() and encode() so the two cannot drift:
// encode() must reject on its own, since nothing forces a caller to go through canEncode() first.
//
// write_pnm() indexes every plane over plane 0's rectangle, while SafePlaneByteOffsets() derives
// each plane's byte offset from that plane's own row_stride * height. A plane smaller than plane 0,
// or a row_stride that does not span one of its rows, is therefore read past the buffer.
static nvimgcodecProcessingStatus_t check_plane_layout(const nvimgcodecImageInfo_t& image_info)
{
    const bool interleaved = image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_RGB;
    const uint32_t expected_num_planes = interleaved ? 1 : kNumComponents;
    const uint32_t expected_num_channels = interleaved ? kNumComponents : 1;

    if (image_info.num_planes != expected_num_planes) {
        return NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
    }

    nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
    const auto& plane0 = image_info.plane_info[0];
    const size_t min_row_bytes =
        static_cast<size_t>(plane0.width) * expected_num_channels * nvimgcodec::TypeSize(plane0.sample_type);
    for (uint32_t p = 0; p < image_info.num_planes; ++p) {
        const auto& plane = image_info.plane_info[p];
        if (plane.num_channels != expected_num_channels) {
            status |= NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
        }
        if (plane.width != plane0.width || plane.height != plane0.height || plane.row_stride < min_row_bytes) {
            status |= NVIMGCODEC_PROCESSING_STATUS_RESOLUTION_UNSUPPORTED;
        }
    }
    return status;
}

// Row strides come from nvimgcodecImageInfo_t and are expressed in bytes, so row bases stay byte
// pointers and the sample is copied out at a byte offset. The image contract does not promise that
// `buffer` or `row_stride` are aligned for the sample type, and forming a misaligned `D*` is
// undefined on its own, not only when it is dereferenced.
template <typename D>
static uint32_t load_sample(const unsigned char* base, size_t pitch_bytes, size_t y, size_t idx)
{
    D value;
    std::memcpy(&value, base + y * pitch_bytes + idx * sizeof(D), sizeof(D));
    return value;
}

// Writes a binary PNM: P5 for one component, P6 for three, P7 (PAM) for four. check_plane_layout()
// admits only three-component images, so the P5 and P7 branches are not reachable through the
// public API - they are kept as they were rather than folded away, since removing them would be a
// capability change rather than a fix.
template <typename D, int SAMPLE_FORMAT = NVIMGCODEC_SAMPLEFORMAT_P_RGB>
static int write_pnm(nvimgcodecIoStreamDesc_t* io_stream, const unsigned char* chanR, size_t pitchR, const unsigned char* chanG,
    size_t pitchG, const unsigned char* chanB, size_t pitchB, const unsigned char* chanA, size_t pitchA, size_t width, size_t height,
    int num_components)
{
    static_assert(std::is_same<D, uint8_t>::value || std::is_same<D, uint16_t>::value, "PNM supports 8- and 16-bit samples only");
    constexpr uint8_t precision = sizeof(D) * 8;
    size_t written_size;
    uint32_t red = 0;
    uint32_t green = 0;
    uint32_t blue = 0;
    uint32_t alpha = 0;
    std::stringstream ss{};
    if (num_components == 4) {
        ss << "P7\n";
        ss << "#nvImageCodec\n";
        ss << "WIDTH " << width << "\n";
        ss << "HEIGHT " << height << "\n";
        ss << "DEPTH " << num_components << "\n";
        ss << "MAXVAL " << (1u << precision) - 1 << "\n";
        ss << "TUPLTYPE RGB_ALPHA\n";
        ss << "ENDHDR\n";
    } else if (num_components == 1) {
        ss << "P5\n";
        ss << "#nvImageCodec\n";
        ss << width << " " << height << "\n";
        ss << (1u << precision) - 1 << "\n";
    } else {
        ss << "P6\n";
        ss << "#nvImageCodec\n";
        ss << width << " " << height << "\n";
        ss << (1u << precision) - 1 << "\n";
    }
    std::string header = ss.str();
    size_t length = header.size() + sizeof(D) * num_components * height * width;
    io_stream->reserve(io_stream->instance, length);
    io_stream->write(io_stream->instance, &written_size, static_cast<void*>(header.data()), header.size());

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            if (SAMPLE_FORMAT == NVIMGCODEC_SAMPLEFORMAT_P_RGB) {
                red = load_sample<D>(chanR, pitchR, y, x);
                if (num_components > 1) {
                    green = load_sample<D>(chanG, pitchG, y, x);
                    blue = load_sample<D>(chanB, pitchB, y, x);
                    if (num_components == 4) {
                        alpha = load_sample<D>(chanA, pitchA, y, x);
                    }
                }
            } else if (SAMPLE_FORMAT == NVIMGCODEC_SAMPLEFORMAT_I_RGB) {
                // Interleaved: consecutive channels of a pixel are adjacent, so the pixel stride is
                // the number of components, not a hardcoded 3.
                const size_t pixel = x * static_cast<size_t>(num_components);
                red = load_sample<D>(chanR, pitchR, y, pixel);
                if (num_components > 1) {
                    green = load_sample<D>(chanR, pitchR, y, pixel + 1);
                    blue = load_sample<D>(chanR, pitchR, y, pixel + 2);
                    if (num_components == 4) {
                        alpha = load_sample<D>(chanR, pitchR, y, pixel + 3);
                    }
                }
            }
            if (precision == 8) {
                io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(red));
                if (num_components > 1) {
                    io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(green));
                    io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(blue));
                    if (num_components == 4) {
                        io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(alpha));
                    }
                }
            } else {
                // PNM stores multi-byte samples most significant byte first.
                io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(red >> 8));
                io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(red & 0xFF));
                if (num_components > 1) {
                    io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(green >> 8));
                    io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(green & 0xFF));
                    io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(blue >> 8));
                    io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(blue & 0xFF));
                    if (num_components == 4) {
                        io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(alpha >> 8));
                        io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(alpha & 0xFF));
                    }
                }
            }
        }
    }
    io_stream->flush(io_stream->instance);
    return 0;
}

NvPnmEncoderPlugin::NvPnmEncoderPlugin(const nvimgcodecFrameworkDesc_t* framework)
    : encoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_ENCODER_DESC, sizeof(nvimgcodecEncoderDesc_t), NULL, this, plugin_id_, "pnm",
          NVIMGCODEC_BACKEND_KIND_CPU_ONLY, static_create, EncoderImpl::static_destroy, EncoderImpl::static_can_encode,
          EncoderImpl::static_encode_sample}
    , framework_(framework)
{
}

nvimgcodecEncoderDesc_t* NvPnmEncoderPlugin::getEncoderDesc()
{
    return &encoder_desc_;
}

nvimgcodecStatus_t NvPnmEncoderPlugin::create(
    nvimgcodecEncoder_t* encoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "pnm_create_encoder");
        XM_CHECK_NULL(encoder);
        XM_CHECK_NULL(exec_params);
        *encoder = reinterpret_cast<nvimgcodecEncoder_t>(new EncoderImpl(plugin_id_, framework_, exec_params, options));
        return NVIMGCODEC_STATUS_SUCCESS;
    } catch (std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create pnm encoder - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
}

nvimgcodecStatus_t NvPnmEncoderPlugin::static_create(
    void* instance, nvimgcodecEncoder_t* encoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    if (!instance) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }

    auto handle = reinterpret_cast<NvPnmEncoderPlugin*>(instance);
    return handle->create(encoder, exec_params, options);
}

nvimgcodecStatus_t EncoderImpl::static_destroy(nvimgcodecEncoder_t encoder)
{
    try {
        XM_CHECK_NULL(encoder);
        auto handle = reinterpret_cast<EncoderImpl*>(encoder);
        delete handle;
    } catch (const std::exception& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

EncoderImpl::EncoderImpl(
    const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params, const char* options)
    : plugin_id_(plugin_id)
    , framework_(framework)
    , exec_params_(exec_params)
    , options_(options)
{
    encode_state_batch_ = std::make_unique<EncodeState>();
    encode_state_batch_->plugin_id_ = plugin_id_;
    encode_state_batch_->framework_ = framework_;
}

EncoderImpl::~EncoderImpl()
{
    NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "pnm_destroy_encoder");
}

nvimgcodecProcessingStatus_t EncoderImpl::canEncode(const nvimgcodecCodeStreamDesc_t* code_stream,
    const nvimgcodecImageDesc_t* image, const nvimgcodecEncodeParams_t* params, int thread_idx)
{
    nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;

    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "pnm_can_encode");
        XM_CHECK_NULL(code_stream);
        XM_CHECK_NULL(params);

        XM_CHECK_NULL(image);
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        auto ret = image->getImageInfo(image->instance, &image_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
        }

        nvimgcodecImageInfo_t out_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        ret = code_stream->getImageInfo(code_stream->instance, &out_image_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
        }

        if (strcmp(out_image_info.codec_name, "pnm") != 0) {
            NVIMGCODEC_LOG_INFO(framework_, plugin_id_, "cannot encode because it is not pnm codec but " << out_image_info.codec_name);
            return NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
        }

        if (params->quality_type != NVIMGCODEC_QUALITY_TYPE_DEFAULT && params->quality_type != NVIMGCODEC_QUALITY_TYPE_LOSSLESS) {
            status |= NVIMGCODEC_PROCESSING_STATUS_QUALITY_TYPE_UNSUPPORTED;
        }

        if (image_info.color_spec != NVIMGCODEC_COLORSPEC_SRGB) {
            status |= NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
        }
        if (image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_NONE) {
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
        }
        if (out_image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_NONE) {
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
        }
        if ((image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_P_RGB) && (image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_I_RGB)) {
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
        } else {
            // Only meaningful once the sample format is one this encoder writes. For any other
            // format the expected plane layout is undefined, and reporting a layout status next to
            // SAMPLE_FORMAT_UNSUPPORTED would name a reason that is not the reason.
            status |= check_plane_layout(image_info);
        }

        if (!nvimgcodec::check_planes_consistency(framework_, plugin_id_, image_info)) {
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
        }
        if (!nvimgcodec::check_orientation_supported(framework_, plugin_id_, image_info, out_image_info)) {
            status |= NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED;
        }

        for (uint32_t p = 0; p < image_info.num_planes; ++p) {
            if (image_info.plane_info[p].sample_type != NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8 &&
                (image_info.plane_info[p].sample_type != NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16)) {
                status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
            }
        }

        // Check for tile geometry - PNM encoder does not support tiling
        nvimgcodecTileGeometryInfo_t* tile_geometry = static_cast<nvimgcodecTileGeometryInfo_t*>(params->struct_next);
        while (tile_geometry && tile_geometry->struct_type != NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO)
            tile_geometry = static_cast<nvimgcodecTileGeometryInfo_t*>(tile_geometry->struct_next);
        if (tile_geometry) {
            if (tile_geometry->num_tiles_x != 0 || tile_geometry->num_tiles_y != 0 ||
                tile_geometry->tile_width != 0 || tile_geometry->tile_height != 0) {
                NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "Tiling is not supported with PNM encoder.");
                status |= NVIMGCODEC_PROCESSING_STATUS_TILING_UNSUPPORTED;
            }
        }
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if pnm can encode - " << e.what());
        return NVIMGCODEC_PROCESSING_STATUS_FAIL;
    }
    return status;
}

nvimgcodecStatus_t EncoderImpl::encode(const nvimgcodecCodeStreamDesc_t* code_stream,
    const nvimgcodecImageDesc_t* image, const nvimgcodecEncodeParams_t* params, int thread_idx)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "pnm_encode");
        XM_CHECK_NULL(image);
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        auto ret = image->getImageInfo(image->instance, &image_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
            return ret;
        }

        nvimgcodec::warn_if_custom_precision_unsupported(framework_, plugin_id_, image_info);

        // A single write_pnm instantiation covers the whole image, so every plane has to agree on the
        // sample type. canEncode() already enforces that, but encode() must not depend on the caller
        // having gone through it.
        const auto sample_type = image_info.plane_info[0].sample_type;
        if ((sample_type != NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8 && sample_type != NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16) ||
            !nvimgcodec::check_planes_consistency(framework_, plugin_id_, image_info)) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED | NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }

        if (image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_I_RGB && image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_P_RGB) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED | NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }

        const nvimgcodecProcessingStatus_t layout_status = check_plane_layout(image_info);
        if (layout_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unsupported plane layout for pnm encoding.");
            image->imageReady(image->instance, layout_status | NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }

        unsigned char* host_buffer = reinterpret_cast<unsigned char*>(image_info.buffer);

        if (NVIMGCODEC_SAMPLEFORMAT_I_RGB == image_info.sample_format) {
            // The sample type selects the write_pnm instantiation: instantiating it as uint8_t for a
            // 16-bit image would index a byte-strided row as bytes and emit garbage.
            if (sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8) {
                write_pnm<uint8_t, NVIMGCODEC_SAMPLEFORMAT_I_RGB>(code_stream->io_stream, host_buffer,
                    image_info.plane_info[0].row_stride, NULL, 0, NULL, 0, NULL, 0, image_info.plane_info[0].width,
                    image_info.plane_info[0].height, image_info.plane_info[0].num_channels);
            } else {
                write_pnm<uint16_t, NVIMGCODEC_SAMPLEFORMAT_I_RGB>(code_stream->io_stream, host_buffer,
                    image_info.plane_info[0].row_stride, NULL, 0, NULL, 0, NULL, 0, image_info.plane_info[0].width,
                    image_info.plane_info[0].height, image_info.plane_info[0].num_channels);
            }
        } else {
            // Plane offsets are the running sum of row_stride * height per the
            // C image_info contract; this keeps offsets correct when planes
            // carry their own height (e.g. chroma-subsampled YUV), with
            // overflow checked.
            size_t plane_offsets[NVIMGCODEC_MAX_NUM_PLANES];
            if (!nvimgcodec::SafePlaneByteOffsets(image_info, plane_offsets)) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Plane byte offsets overflow.");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }
            if (sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8) {
                write_pnm<uint8_t>(code_stream->io_stream, host_buffer, image_info.plane_info[0].row_stride,
                    host_buffer + plane_offsets[1], image_info.plane_info[1].row_stride,
                    host_buffer + plane_offsets[2], image_info.plane_info[2].row_stride,
                    NULL, 0, image_info.plane_info[0].width, image_info.plane_info[0].height, image_info.num_planes);
            } else {
                write_pnm<uint16_t>(code_stream->io_stream, host_buffer, image_info.plane_info[0].row_stride,
                    host_buffer + plane_offsets[1], image_info.plane_info[1].row_stride,
                    host_buffer + plane_offsets[2], image_info.plane_info[2].row_stride,
                    NULL, 0, image_info.plane_info[0].width, image_info.plane_info[0].height, image_info.num_planes);
            }
        }
        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
        return NVIMGCODEC_STATUS_SUCCESS;
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not encode pnm code stream - " << e.what());
        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
        return NVIMGCODEC_STATUS_EXECUTION_FAILED;
    }
}

} // namespace nvpnm
