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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nvimgcodec.h>
#include <memory>
#include "nvimgcodec_tests.h"

namespace nvimgcodec { namespace test {

class ObjectReuseTest : public ::testing::Test
{
  public:
    ObjectReuseTest() {}

    void SetUp() override { 
        create_info_.load_builtin_modules = 1;
        create_info_.load_extension_modules = 1;
        create_info_.create_debug_messenger = 1;
        create_info_.message_severity = NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEFAULT;
        create_info_.message_category = NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_ALL;
    
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance_, &create_info_));
        exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;
        exec_params.max_num_cpu_threads = 1;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCreate(instance_, &decoder_, &exec_params, nullptr));
    }

    void TearDown() override {
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDestroy(decoder_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance_));
    }

    nvimgcodecInstanceCreateInfo_t create_info_{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};
    nvimgcodecInstance_t instance_;
    nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
    nvimgcodecDecoder_t decoder_;
    nvimgcodecDecodeParams_t dec_params_{NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), 0};
};

void ExpectEqual(const nvimgcodecImageInfo_t& image_info1, const nvimgcodecImageInfo_t& image_info2)
{
    EXPECT_EQ(image_info1.struct_type, image_info2.struct_type);
    EXPECT_EQ(image_info1.struct_size, image_info2.struct_size);
    EXPECT_EQ(image_info1.struct_next, image_info2.struct_next);
    EXPECT_STREQ(image_info1.codec_name, image_info2.codec_name);
    EXPECT_EQ(image_info1.color_spec, image_info2.color_spec);
    EXPECT_EQ(image_info1.chroma_subsampling, image_info2.chroma_subsampling);
    EXPECT_EQ(image_info1.sample_format, image_info2.sample_format);
    EXPECT_EQ(image_info1.orientation.struct_type, image_info2.orientation.struct_type);
    EXPECT_EQ(image_info1.orientation.struct_size, image_info2.orientation.struct_size);
    EXPECT_EQ(image_info1.orientation.struct_next, image_info2.orientation.struct_next);
    EXPECT_EQ(image_info1.orientation.rotated, image_info2.orientation.rotated);
    EXPECT_EQ(image_info1.orientation.flip_x, image_info2.orientation.flip_x);
    EXPECT_EQ(image_info1.orientation.flip_y, image_info2.orientation.flip_y);
    EXPECT_EQ(image_info1.num_planes, image_info2.num_planes);
    for (uint32_t i = 0; i < image_info1.num_planes; ++i) {
        const auto& p1 = image_info1.plane_info[i];
        const auto& p2 = image_info2.plane_info[i];
        EXPECT_EQ(p1.struct_type, p2.struct_type);
        EXPECT_EQ(p1.struct_size, p2.struct_size);
        EXPECT_EQ(p1.struct_next, p2.struct_next);
        EXPECT_EQ(p1.width, p2.width) << "Plane " << i;
        EXPECT_EQ(p1.height, p2.height) << "Plane " << i;
        EXPECT_EQ(p1.row_stride, p2.row_stride) << "Plane " << i;
        EXPECT_EQ(p1.num_channels, p2.num_channels) << "Plane " << i;
        EXPECT_EQ(p1.sample_type, p2.sample_type) << "Plane " << i;
        EXPECT_EQ(p1.precision, p2.precision) << "Plane " << i;
    }
    EXPECT_EQ(image_info1.buffer, image_info2.buffer);
    EXPECT_EQ(image_info1.buffer_kind, image_info2.buffer_kind);
    EXPECT_EQ(image_info1.cuda_stream, image_info2.cuda_stream);
}


TEST_F(ObjectReuseTest, code_stream_reuse)
{
    std::string path0 = resources_dir + "/jpeg/padlock-406986_640_410.jpg";
    nvimgcodecImageInfo_t image_info0{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    std::string path1 = resources_dir + "/jpeg2k/cat-1046544_640.jp2";
    nvimgcodecImageInfo_t image_info1{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    nvimgcodecImageInfo_t image_info2{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};

    nvimgcodecCodeStream_t stream_handle = nullptr;
    // 1. destroying an recreating
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamCreateFromFile(instance_, &stream_handle, path0.c_str()));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle, &image_info0));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(stream_handle));
    stream_handle = nullptr;  // need to reset the handle
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamCreateFromFile(instance_, &stream_handle, path1.c_str()));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle, &image_info1));
    // 2. reuse without destroying
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamCreateFromFile(instance_, &stream_handle, path0.c_str()));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle, &image_info2));
    ExpectEqual(image_info0, image_info2);

    // Destroy.
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(stream_handle));
}

TEST_F(ObjectReuseTest, image_reuse)
{
    nvimgcodecImage_t image = nullptr;
    nvimgcodecCodeStream_t stream_handle = nullptr;

    std::vector<uint8_t> buffer0, buffer1;

    nvimgcodecImageInfo_t image_info0{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    std::string path0 = resources_dir + "/jpeg/padlock-406986_640_410.jpg";
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamCreateFromFile(instance_, &stream_handle, path0.c_str()));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,nvimgcodecCodeStreamGetImageInfo(stream_handle, &image_info0));
    image_info0.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
    buffer0.resize(image_info0.plane_info[0].height * image_info0.plane_info[0].width * 3);
    image_info0.buffer = buffer0.data();
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &image, &image_info0));
    nvimgcodecFuture_t fut0 = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDecode(decoder_, &stream_handle, &image, 1, &dec_params_, &fut0));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(fut0));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureDestroy(fut0));

    nvimgcodecImageInfo_t image_info1{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    std::string path1 = resources_dir + "/jpeg2k/cat-1046544_640.jp2";
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamCreateFromFile(instance_, &stream_handle, path1.c_str()));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,nvimgcodecCodeStreamGetImageInfo(stream_handle, &image_info1));
    image_info1.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
    buffer1.resize(image_info1.plane_info[0].height * image_info1.plane_info[0].width * 3);
    image_info1.buffer = buffer1.data();
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &image, &image_info1));
    nvimgcodecFuture_t fut1 = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDecode(decoder_, &stream_handle, &image, 1, &dec_params_, &fut1));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(fut1));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureDestroy(fut1));

    // Destroy only once.
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageDestroy(image));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(stream_handle));
}

}} // namespace nvimgcodec::test
