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

#pragma once

#include <cstring>
#include <vector>

#include <nvimgcodec.h>
#include "imgproc/type_utils.h"

#include <gtest/gtest.h>

namespace nvimgcodec { namespace test {

class ExtensionTestBase
{
  public:
    virtual ~ExtensionTestBase() = default;
    virtual void SetUp()
    {
        nvimgcodecInstanceCreateInfo_t create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};
        create_info.create_debug_messenger = 1;
        create_info.message_severity = NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEFAULT;
        create_info.message_category = NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_ALL;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance_, &create_info));

        images_.clear();
        streams_.clear();
    }

    virtual void TearDown() { ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance_)); }

    virtual void TearDownCodecResources()
    {
        if (future_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureDestroy(future_));
        }
        if (in_image_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageDestroy(in_image_));
        }
        if (out_image_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageDestroy(out_image_));
        }
        if (in_code_stream_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(in_code_stream_));
        }
        if (out_code_stream_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(out_code_stream_));
        }
    }

    void PrepareImageForPlanarFormat(int num_planes = 3)
    {
        image_info_.num_planes = num_planes;
        for (uint32_t p = 0; p < image_info_.num_planes; p++) {
            image_info_.plane_info[p].height = image_info_.plane_info[0].height;
            image_info_.plane_info[p].width = image_info_.plane_info[0].width;
            image_info_.plane_info[p].num_channels = 1;
            image_info_.plane_info[p].sample_type = image_info_.plane_info[0].sample_type;
            image_info_.plane_info[p].precision = image_info_.plane_info[0].precision;
            image_info_.plane_info[p].row_stride = (
                static_cast<size_t>(image_info_.plane_info[p].width) *
                TypeSize(image_info_.plane_info[p].sample_type)
            );
        }
        image_buffer_.resize(GetBufferSize(image_info_));
        image_info_.buffer = image_buffer_.data();
        image_info_.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
    }

    void PrepareImageForInterleavedFormat(int num_channels = 3)
    {
        image_info_.num_planes = 1;
        image_info_.plane_info[0].num_channels = num_channels;
        image_info_.plane_info[0].row_stride = (
            static_cast<size_t>(image_info_.plane_info[0].width) *
            image_info_.plane_info[0].num_channels *
            TypeSize(image_info_.plane_info[0].sample_type)
        );
        image_buffer_.resize(GetBufferSize(image_info_));
        image_info_.buffer = image_buffer_.data();
        image_info_.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
    }

    void PrepareImageForFormat(bool add_alpha = false)
    {
        image_info_.color_spec = color_spec_;
        image_info_.sample_format = sample_format_;
        image_info_.chroma_subsampling = chroma_subsampling_;

        switch (sample_format_) {
        case NVIMGCODEC_SAMPLEFORMAT_P_YUV:
        case NVIMGCODEC_SAMPLEFORMAT_P_BGR:
        case NVIMGCODEC_SAMPLEFORMAT_P_RGB: {
            PrepareImageForPlanarFormat(3 + static_cast<int>(add_alpha));
            break;
        }
        case NVIMGCODEC_SAMPLEFORMAT_P_Y:
        case NVIMGCODEC_SAMPLEFORMAT_I_Y: {
            PrepareImageForPlanarFormat(1);
            break;
        }
        case NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED: {
            PrepareImageForInterleavedFormat(3 + static_cast<int>(add_alpha));
            break;
        }
        case NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED: {
            PrepareImageForPlanarFormat(3 + static_cast<int>(add_alpha));
            break;
        }
        case NVIMGCODEC_SAMPLEFORMAT_I_BGR:
        case NVIMGCODEC_SAMPLEFORMAT_I_RGB: {
            PrepareImageForInterleavedFormat(3 + static_cast<int>(add_alpha));
            break;
        }
        default: {
            std::cerr << "Unsupported sample format: " << sample_format_ << "\n";
            assert(!"TODO");
        }
        }
    }

    // The interleave/deinterleave helpers below move whole samples, sized by the plane's sample
    // type, rather than single bytes. Byte-wise copies silently truncate anything wider than
    // uint8 - the same class of bug that made the PNM encoder emit garbage for 16-bit images.
    size_t SampleSize() const { return TypeSize(image_info_.plane_info[0].sample_type); }

    // Byte offset of sample (y, x, c) in an interleaved buffer with the given row stride.
    size_t InterleavedOffset(uint32_t y, uint32_t x, uint32_t c) const
    {
        const auto& plane = image_info_.plane_info[0];
        return y * plane.row_stride + (static_cast<size_t>(x) * plane.num_channels + c) * SampleSize();
    }

    // Byte offset of sample (y, x, c) in a tightly packed planar buffer.
    size_t PlanarOffset(uint32_t y, uint32_t x, uint32_t c) const
    {
        const auto& plane = image_info_.plane_info[0];
        return ((static_cast<size_t>(c) * plane.height + y) * plane.width + x) * SampleSize();
    }

    // Writes through out_buffer rather than image_info_.buffer: out_buffer is the vector that
    // image_info_.buffer points into at every call site, so reading the cached image_info_.buffer
    // after the resize would use a stale pointer if the resize ever reallocated.
    void Convert_P_RGB_to_I_RGB(std::vector<uint8_t>& out_buffer, const std::vector<uint8_t>& in_buffer)
    {
        out_buffer.resize(in_buffer.size());
        auto* dst = out_buffer.data();
        for (uint32_t y = 0; y < image_info_.plane_info[0].height; y++) {
            for (uint32_t x = 0; x < image_info_.plane_info[0].width; x++) {
                for (uint32_t c = 0; c < image_info_.plane_info[0].num_channels; ++c) {
                    memcpy(dst + InterleavedOffset(y, x, c), in_buffer.data() + PlanarOffset(y, x, c), SampleSize());
                }
            }
        }
    }

    void Convert_I_RGB_to_P_RGB()
    {
        planar_out_buffer_.resize(image_buffer_.size());
        const auto* src = static_cast<const uint8_t*>(image_info_.buffer);
        for (uint32_t y = 0; y < image_info_.plane_info[0].height; y++) {
            for (uint32_t x = 0; x < image_info_.plane_info[0].width; x++) {
                for (uint32_t c = 0; c < image_info_.plane_info[0].num_channels; ++c) {
                    memcpy(planar_out_buffer_.data() + PlanarOffset(y, x, c), src + InterleavedOffset(y, x, c), SampleSize());
                }
            }
        }
    }

    void Convert_P_BGR_to_P_RGB()
    {
        planar_out_buffer_.resize(image_buffer_.size());
        auto plane_size = image_info_.plane_info[0].height * image_info_.plane_info[0].row_stride;
        memcpy(planar_out_buffer_.data(), static_cast<char*>(image_info_.buffer) + 2 * plane_size, plane_size);
        memcpy(planar_out_buffer_.data() + plane_size, static_cast<char*>(image_info_.buffer) + plane_size, plane_size);
        memcpy(planar_out_buffer_.data() + 2 * plane_size, static_cast<char*>(image_info_.buffer), plane_size);
    }

    void Convert_I_BGR_to_P_RGB()
    {
        planar_out_buffer_.resize(image_buffer_.size());
        const auto* src = static_cast<const uint8_t*>(image_info_.buffer);
        const uint32_t num_channels = image_info_.plane_info[0].num_channels;
        for (uint32_t y = 0; y < image_info_.plane_info[0].height; y++) {
            for (uint32_t x = 0; x < image_info_.plane_info[0].width; x++) {
                for (uint32_t c = 0; c < num_channels; ++c) {
                    memcpy(planar_out_buffer_.data() + PlanarOffset(y, x, num_channels - c - 1),
                        src + InterleavedOffset(y, x, c), SampleSize());
                }
            }
        }
    }

    void ConvertToPlanar()
    {
        switch (sample_format_) {
        case NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED:
        case NVIMGCODEC_SAMPLEFORMAT_P_Y:
        case NVIMGCODEC_SAMPLEFORMAT_P_YUV:
        case NVIMGCODEC_SAMPLEFORMAT_P_RGB: {
            planar_out_buffer_.resize(image_buffer_.size());
            memcpy(planar_out_buffer_.data(), image_buffer_.data(), image_buffer_.size());
            break;
        }
        case NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED:
        case NVIMGCODEC_SAMPLEFORMAT_I_RGB: {
            Convert_I_RGB_to_P_RGB();
            break;
        }
        case NVIMGCODEC_SAMPLEFORMAT_P_BGR: {
            Convert_P_BGR_to_P_RGB();
            break;
        }
        case NVIMGCODEC_SAMPLEFORMAT_I_BGR: {
            Convert_I_BGR_to_P_RGB();
            break;
        }
        default: {
            assert(!"TODO");
        }
        }
    }

    unsigned char* ResizeBuffer(size_t bytes)
    {
        code_stream_buffer_.resize(bytes);
        return code_stream_buffer_.data();
    }

    template <class T>
    static unsigned char* ResizeBufferStatic(void* ctx, size_t bytes)
    {
        auto handle = reinterpret_cast<T*>(ctx);
        return handle->ResizeBuffer(bytes);
    }

    nvimgcodecInstance_t instance_;
    std::string image_file_;
    nvimgcodecCodeStream_t in_code_stream_ = nullptr;
    nvimgcodecCodeStream_t out_code_stream_ = nullptr;
    nvimgcodecImage_t in_image_ = nullptr;
    nvimgcodecImage_t out_image_ = nullptr;
    std::vector<nvimgcodecImage_t> images_;
    std::vector<nvimgcodecCodeStream_t> streams_;
    nvimgcodecFuture_t future_ = nullptr;

    nvimgcodecImageInfo_t image_info_;
    nvimgcodecSampleFormat_t reference_output_format_;
    std::vector<unsigned char> planar_out_buffer_;
    nvimgcodecColorSpec_t color_spec_;
    nvimgcodecSampleFormat_t sample_format_;
    nvimgcodecChromaSubsampling_t chroma_subsampling_;
    std::vector<unsigned char> image_buffer_;
    std::vector<unsigned char> code_stream_buffer_;
};
}} // namespace nvimgcodec::test
