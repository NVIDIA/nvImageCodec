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

#include <gtest/gtest.h>
#include <nvimgcodec.h>
#include <parsers/parser_test_utils.h>
#include <test_utils.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include "nvimgcodec_tests.h"

#define DEBUG_DUMP_DECODE_OUTPUT 0

namespace nvimgcodec { namespace test {

class CommonExtDecoderTest
{
  public:
    CommonExtDecoderTest() {}


    void SetUp()
    {
        nvimgcodecInstanceCreateInfo_t create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};
        create_info.message_severity = NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEFAULT;
        create_info.message_category = NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_ALL;

        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance_, &create_info));

        image_info_ = {NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        images_.clear();
        streams_.clear();
        nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
        exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;
        exec_params.max_num_cpu_threads = 1;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCreate(instance_, &decoder_, &exec_params, nullptr));
        params_ = {NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), 0};
        params_.apply_exif_orientation= 1;
    }

    void TearDown()
    {
        if (decoder_)
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDestroy(decoder_));
        if (future_)
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureDestroy(future_));
        if (image_)
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageDestroy(image_));
        if (in_code_stream_)
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(in_code_stream_));
        for (auto& ext : extensions_)
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(ext));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance_));
    }

    void TestSingleImage(const std::string& rel_path, nvimgcodecSampleFormat_t sample_format,
        nvimgcodecRegion_t region = {NVIMGCODEC_STRUCTURE_TYPE_REGION, sizeof(nvimgcodecRegion_t), nullptr, 0})
    {
        std::string filename = resources_dir + "/" + rel_path;
        std::string reference_filename = std::filesystem::path(resources_dir + "/ref/" + rel_path).replace_extension(".ppm").string();
        int num_channels = sample_format == NVIMGCODEC_SAMPLEFORMAT_P_Y ? 1 : 3;
        auto cv_type = num_channels == 1 ? CV_8UC1 : CV_8UC3;
        int cv_flags = num_channels == 1 ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;
        cv::Mat ref;
        if (region.ndim == 0) {
            ref = cv::imread(reference_filename, cv_flags);
        } else {
            int start_x = region.start[1];
            int start_y = region.start[0];
            int crop_w = region.end[1] - region.start[1];
            int crop_h = region.end[0] - region.start[0];
            cv::Mat tmp = cv::imread(reference_filename, cv_flags);
            cv::Rect roi(start_x, start_y, crop_w, crop_h);
            tmp(roi).copyTo(ref);
        }

        bool planar = sample_format == NVIMGCODEC_SAMPLEFORMAT_P_RGB || sample_format == NVIMGCODEC_SAMPLEFORMAT_P_BGR ||
                      sample_format == NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED;
        bool bgr = sample_format == NVIMGCODEC_SAMPLEFORMAT_P_BGR || sample_format == NVIMGCODEC_SAMPLEFORMAT_I_BGR;
        if (!bgr && num_channels >= 3)
            ref = bgr2rgb(ref);

        LoadImageFromFilename(instance_, in_code_stream_, filename);
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(in_code_stream_, &image_info_));
        image_info_.sample_format = sample_format;
        image_info_.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        image_info_.num_planes = 1;
        uint32_t& width = image_info_.plane_info[0].width;
        uint32_t& height = image_info_.plane_info[0].height;
        bool swap_xy = params_.apply_exif_orientation && image_info_.orientation.rotated % 180 == 90;
        if (swap_xy) {
            std::swap(width, height);
        }
        if (region.ndim == 2) {
            width = region.end[1] - region.start[1];
            height = region.end[0] - region.start[0];
        }

        image_info_.region = region;
        image_info_.num_planes = planar ? num_channels : 1;
        int plane_nchannels = planar ? 1 : num_channels;
        for (int p = 0; p < image_info_.num_planes; p++) {
            image_info_.plane_info[p].width = width;
            image_info_.plane_info[p].height = height;
            image_info_.plane_info[p].row_stride = width * plane_nchannels;
            image_info_.plane_info[p].num_channels = plane_nchannels;
            image_info_.plane_info[p].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        }
        image_info_.buffer_size = height * width * num_channels;
        out_buffer_.resize(image_info_.buffer_size);
        image_info_.buffer = out_buffer_.data();
        image_info_.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &image_, &image_info_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDecode(decoder_, &in_code_stream_, &image_, 1, &params_, &future_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future_));

        nvimgcodecProcessingStatus_t status;
        size_t status_size;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future_, &status, &status_size));
        ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, status);

        ASSERT_EQ(ref.size[0], height);
        ASSERT_EQ(ref.size[1], width);
        ASSERT_EQ(ref.type(), cv_type);
#if DEBUG_DUMP_DECODE_OUTPUT
        cv::Mat decoded_image(height, width, cv_type, static_cast<void*>(out_buffer_.data()));
        cv::imwrite("./decode_out.ppm", rgb2bgr(decoded_image));
        cv::imwrite("./ref.ppm", rgb2bgr(ref));
#endif

        uint8_t eps = 1;
        if (rel_path.find("exif") != std::string::npos) {
            eps = 4;
        }
        else if (rel_path.find("cmyk") != std::string::npos) {
            eps = 2;
        }
        if (planar) {
            size_t out_pos = 0;
            for (size_t c = 0; c < num_channels; c++) {
                for (size_t i = 0; i < height; i++) {
                    for (size_t j = 0; j < width; j++, out_pos++) {
                        auto out_val = out_buffer_[out_pos];
                        size_t ref_pos = i * width * num_channels + j * num_channels + c;
                        auto ref_val = ref.data[ref_pos];
                        ASSERT_NEAR(out_val, ref_val, eps)
                            << "@" << i << "x" << j << "x" << c << " : " << (int)out_val << " != " << (int)ref_val << "\n";
                    }
                }
            }
        } else {
            size_t pos = 0;
            for (size_t i = 0; i < height; i++) {
                for (size_t j = 0; j < width; j++) {
                    for (size_t c = 0; c < num_channels; c++, pos++) {
                        ASSERT_NEAR(out_buffer_.data()[pos], ref.data[pos], eps)
                            << "@" << i << "x" << j << "x" << c << " : " << (int)out_buffer_.data()[pos] << " != " << (int)ref.data[pos]
                            << "\n";
                    }
                }
            }
        }
    }

    void TestNotSupported(const std::string& rel_path, nvimgcodecSampleFormat_t sample_format, nvimgcodecSampleDataType_t sample_type,
        nvimgcodecProcessingStatus_t expected_status)
    {
        std::string filename = resources_dir + "/" + rel_path;

        int num_channels = sample_format == NVIMGCODEC_SAMPLEFORMAT_P_Y ? 1 : 3;
        LoadImageFromFilename(instance_, in_code_stream_, filename);
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(in_code_stream_, &image_info_));
        image_info_.sample_format = sample_format;
        image_info_.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        image_info_.num_planes = 1;
        image_info_.plane_info[0].row_stride = image_info_.plane_info[0].width * num_channels;
        image_info_.plane_info[0].num_channels = num_channels;
        image_info_.plane_info[0].sample_type = sample_type;
        image_info_.buffer_size =
            image_info_.plane_info[0].height * image_info_.plane_info[0].width * image_info_.plane_info[0].num_channels;
        out_buffer_.resize(image_info_.buffer_size);
        image_info_.buffer = out_buffer_.data();
        image_info_.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &image_, &image_info_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDecode(decoder_, &in_code_stream_, &image_, 1, &params_, &future_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future_));

        nvimgcodecProcessingStatus_t status;
        size_t status_size;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future_, &status, &status_size));
        ASSERT_EQ(expected_status, status);
    }

    nvimgcodecInstance_t instance_;
    nvimgcodecDecoder_t decoder_;
    nvimgcodecDecodeParams_t params_;
    nvimgcodecImageInfo_t image_info_;
    nvimgcodecCodeStream_t in_code_stream_ = nullptr;
    nvimgcodecImage_t image_ = nullptr;
    std::vector<nvimgcodecImage_t> images_;
    std::vector<nvimgcodecCodeStream_t> streams_;
    nvimgcodecFuture_t future_ = nullptr;
    std::vector<uint8_t> out_buffer_;
    std::vector<nvimgcodecExtension_t> extensions_;
};

}} // namespace nvimgcodec::test
