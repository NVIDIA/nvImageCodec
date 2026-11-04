/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>
#include <nvimgcodec.h>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace nvimgcodec { namespace test {

class FileExtensionMappingApiTest : public ::testing::Test
{
  public:
    void SetUp() override
    {
        create_info_.struct_type = NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info_.struct_size = sizeof(nvimgcodecInstanceCreateInfo_t);
        create_info_.struct_next = nullptr;
        create_info_.load_builtin_modules = 1;
        create_info_.load_extension_modules = 0;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance_, &create_info_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecGetFrameworkDesc(instance_, &framework_));
        ASSERT_NE(nullptr, framework_);
    }

    void TearDown() override { ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance_)); }

    nvimgcodecInstanceCreateInfo_t create_info_{};
    nvimgcodecInstance_t instance_ = nullptr;
    const nvimgcodecFrameworkDesc_t* framework_ = nullptr;
};

TEST_F(FileExtensionMappingApiTest, jpeg_extension_default_and_list)
{
    int count = 0;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        framework_->getCodecNames(framework_->instance, nullptr, &count, ".jpg"));
    ASSERT_GE(count, 1);

    std::vector<char> names(static_cast<size_t>(count) * NVIMGCODEC_MAX_CODEC_NAME_SIZE, '\0');
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        framework_->getCodecNames(framework_->instance, names.data(), &count, ".jpg"));
    EXPECT_STREQ("jpeg", names.data());

    char default_only[NVIMGCODEC_MAX_CODEC_NAME_SIZE] = {};
    int one = 1;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        framework_->getCodecNames(framework_->instance, default_only, &one, "JPG"));
    EXPECT_EQ(1, one);
    EXPECT_STREQ("jpeg", default_only);

    count = 0;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        framework_->getFileExtensions(framework_->instance, nullptr, &count, "jpeg"));
    ASSERT_EQ(2, count);
    std::vector<char> exts(static_cast<size_t>(count) * NVIMGCODEC_MAX_FILE_EXTENSION_SIZE, '\0');
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        framework_->getFileExtensions(framework_->instance, exts.data(), &count, "jpeg"));
    EXPECT_STREQ(".jpg", exts.data());
    EXPECT_STREQ(".jpeg", exts.data() + NVIMGCODEC_MAX_FILE_EXTENSION_SIZE);

    std::vector<char> truncated_exts(2 * NVIMGCODEC_MAX_FILE_EXTENSION_SIZE, 'x');
    int truncated_count = 1;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        framework_->getFileExtensions(framework_->instance, truncated_exts.data(), &truncated_count, "jpeg"));
    EXPECT_EQ(1, truncated_count);
    EXPECT_STREQ(".jpg", truncated_exts.data());
    EXPECT_EQ('x', truncated_exts[NVIMGCODEC_MAX_FILE_EXTENSION_SIZE]);

    int oversized_count = 3;
    std::vector<char> oversized_exts(3 * NVIMGCODEC_MAX_FILE_EXTENSION_SIZE, 'x');
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        framework_->getFileExtensions(framework_->instance, oversized_exts.data(), &oversized_count, "jpeg"));
    EXPECT_EQ(3, oversized_count);
    EXPECT_EQ('x', oversized_exts[0]);
}

TEST_F(FileExtensionMappingApiTest, unknown_extension_and_codec)
{
    int count = -1;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        framework_->getCodecNames(framework_->instance, nullptr, &count, ".unknown"));
    EXPECT_EQ(0, count);

    count = 1;
    char name[NVIMGCODEC_MAX_CODEC_NAME_SIZE] = {'x'};
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        framework_->getCodecNames(framework_->instance, name, &count, ".unknown"));
    EXPECT_EQ(1, count);
    EXPECT_EQ('x', name[0]);

    count = -1;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        framework_->getFileExtensions(framework_->instance, nullptr, &count, "no_such_codec"));
    EXPECT_EQ(0, count);
}

TEST_F(FileExtensionMappingApiTest, all_file_extensions_are_unique)
{
    int count = 0;
    ASSERT_EQ(
        NVIMGCODEC_STATUS_SUCCESS, framework_->getFileExtensions(framework_->instance, nullptr, &count, nullptr));
    ASSERT_GT(count, 0);

    std::vector<char> extensions(static_cast<size_t>(count) * NVIMGCODEC_MAX_FILE_EXTENSION_SIZE, '\0');
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        framework_->getFileExtensions(framework_->instance, extensions.data(), &count, nullptr));

    std::set<std::string> unique_extensions;
    for (int i = 0; i < count; ++i) {
        unique_extensions.emplace(extensions.data() + static_cast<size_t>(i) * NVIMGCODEC_MAX_FILE_EXTENSION_SIZE);
    }
    EXPECT_EQ(static_cast<size_t>(count), unique_extensions.size());
    EXPECT_EQ(1, unique_extensions.count(".jpg"));
}

TEST_F(FileExtensionMappingApiTest, jpeg2k_and_pnm_extensions)
{
    char codec[NVIMGCODEC_MAX_CODEC_NAME_SIZE] = {};
    int count = 1;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        framework_->getCodecNames(framework_->instance, codec, &count, ".jp2"));
    EXPECT_STREQ("jpeg2k", codec);

    count = 0;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        framework_->getFileExtensions(framework_->instance, nullptr, &count, "pnm"));
    EXPECT_EQ(4, count);
}

}} // namespace nvimgcodec::test
