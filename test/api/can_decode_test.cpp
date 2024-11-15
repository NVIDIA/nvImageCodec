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

#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nvimgcodec.h>

#include "can_de_en_code_common.h"
#include "parsers/bmp.h"
#include "parsers/parser_test_utils.h"

using ::testing::Bool;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

namespace nvimgcodec { namespace test {

namespace {
static unsigned char small_bmp[] = {0x42, 0x4D, 0x1E, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1A, 0x00, 0x00, 0x00, 0x0C, 0x00, 0x00,
    0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x18, 0x00, 0x00, 0x00, 0xFF, 0x00};

using test_case_tuple_t =
    std::tuple<const std::vector<std::vector<nvimgcodecProcessingStatus_t>>*, bool, const std::vector<nvimgcodecProcessingStatus_t>*>;

class MockDecoderPlugin
{
  public:
    explicit MockDecoderPlugin(const nvimgcodecFrameworkDesc_t* framework, const std::vector<nvimgcodecProcessingStatus_t>& return_status)
        : return_status_(return_status)
        , decoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC, sizeof(nvimgcodecDecoderDesc_t), NULL,
              this,                // instance
              "mock_test_decoder", // id
              "bmp",               // codec_type
              NVIMGCODEC_BACKEND_KIND_CPU_ONLY,
              static_create, static_destroy, static_can_decode, static_decode_sample, nullptr, nullptr}
        , i_(0)
    {
    }
    nvimgcodecDecoderDesc_t* getDecoderDesc() { return &decoder_desc_; }

  private:
    static nvimgcodecStatus_t static_create(
        void* instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
    {
        *decoder = static_cast<nvimgcodecDecoder_t>(instance);
        return NVIMGCODEC_STATUS_SUCCESS;
    }
    static nvimgcodecStatus_t static_destroy(nvimgcodecDecoder_t decoder) { return NVIMGCODEC_STATUS_SUCCESS; }

    static nvimgcodecProcessingStatus_t static_can_decode(nvimgcodecDecoder_t decoder, const nvimgcodecImageDesc_t* image,
        const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx)
    {
        auto handle = reinterpret_cast<MockDecoderPlugin*>(decoder);
        return handle->return_status_[handle->i_++];  // TODO(janton): this only works if we don't parallelize canDecode!
    }
    
    static nvimgcodecStatus_t static_decode_sample(nvimgcodecDecoder_t decoder, const nvimgcodecImageDesc_t* image,
        const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx)
    {
        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    const std::vector<nvimgcodecProcessingStatus_t>& return_status_;
    nvimgcodecDecoderDesc_t decoder_desc_;
    int i_;
};

struct MockCodecExtensionFactory
{
  public:
    explicit MockCodecExtensionFactory(const std::vector<std::vector<nvimgcodecProcessingStatus_t>>* statuses)
        : desc_{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), nullptr, this, "test_extension", NVIMGCODEC_VER, NVIMGCODEC_EXT_API_VER, static_extension_create,
              static_extension_destroy}
        , statuses_(statuses)

    {
    }

    nvimgcodecExtensionDesc_t* getExtensionDesc() { return &desc_; };

    struct Extension
    {
        explicit Extension(const nvimgcodecFrameworkDesc_t* framework, const std::vector<std::vector<nvimgcodecProcessingStatus_t>>* statuses)
            : framework_(framework)
            , statuses_(statuses)
        {
            decoders_.reserve(statuses_->size());
            for (auto& item : *statuses_)
            {
                decoders_.emplace_back(framework, item);
                framework->registerDecoder(framework->instance, decoders_.back().getDecoderDesc(), NVIMGCODEC_PRIORITY_NORMAL);
            }
        }
        ~Extension()
        {
            for (auto& item : decoders_) {
                framework_->unregisterDecoder(framework_->instance, item.getDecoderDesc());
            }
        }

        const nvimgcodecFrameworkDesc_t* framework_;
        std::vector<MockDecoderPlugin> decoders_;
        const std::vector<std::vector<nvimgcodecProcessingStatus_t>>* statuses_;
    };

    static nvimgcodecStatus_t static_extension_create(
        void* instance, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework)
    {
        auto handle = reinterpret_cast<MockCodecExtensionFactory*>(instance);
        *extension = reinterpret_cast<nvimgcodecExtension_t>(new Extension(framework, handle->statuses_));
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static nvimgcodecStatus_t static_extension_destroy(nvimgcodecExtension_t extension)
    {
        auto ext_handle = reinterpret_cast<Extension*>(extension);
        delete ext_handle;
        return NVIMGCODEC_STATUS_SUCCESS;
    }

  private:
    nvimgcodecExtensionDesc_t desc_;
    const std::vector<std::vector<nvimgcodecProcessingStatus_t>>* statuses_;
};

} // namespace

class NvImageCodecsCanDecodeApiTest : public TestWithParam < std::tuple<test_case_tuple_t, bool>>
{
  public:
    NvImageCodecsCanDecodeApiTest() {}
    virtual ~NvImageCodecsCanDecodeApiTest() = default;

  protected:
    void SetUp() override
    {
        test_case_tuple_t test_case = std::get<0>(GetParam());
        mock_extension_ = std::make_unique<MockCodecExtensionFactory>(std::get<0>(test_case));
        force_format_ = std::get<1>(test_case);
        expected_statuses_ = std::get<2>(test_case);
        register_extension_ =  std::get<1>(GetParam());

        nvimgcodecInstanceCreateInfo_t create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};
        create_info.load_builtin_modules= 1;

        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance_, &create_info));

        if (register_extension_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extension_, mock_extension_->getExtensionDesc()));
        }

        nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
        exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;
        exec_params.max_num_cpu_threads = 1;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCreate(instance_, &decoder_, &exec_params, nullptr));
        params_ = {NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), 0};
        image_info_ = {NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        image_info_.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        out_buffer_.resize(1);
        image_info_.buffer = out_buffer_.data();
        image_info_.buffer_size = 1;

        images_.clear();
        streams_.clear();

        for (size_t i = 0; i < expected_statuses_->size(); ++i) {
            nvimgcodecCodeStream_t code_stream = nullptr;
            LoadImageFromHostMemory(instance_, code_stream, small_bmp, sizeof(small_bmp));
            streams_.push_back(code_stream);
            nvimgcodecImage_t image;
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &image, &image_info_));
            images_.push_back(image);
        }
    }

    void TearDown() override
    {
        for (auto im : images_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageDestroy(im));
        }
        for (auto cs : streams_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(cs));
        }
        if (decoder_)
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDestroy(decoder_));
        if (extension_)
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(extension_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance_));
        mock_extension_.reset();
    }

    nvimgcodecInstance_t instance_;
    nvimgcodecExtension_t extension_ = nullptr;
    std::unique_ptr<MockCodecExtensionFactory> mock_extension_;
    std::vector<unsigned char> out_buffer_;
    nvimgcodecImageInfo_t image_info_;
    nvimgcodecDecoder_t decoder_;
    nvimgcodecDecodeParams_t params_;
    std::vector<nvimgcodecImage_t> images_;
    std::vector<nvimgcodecCodeStream_t> streams_;
    bool force_format_ = true;
    bool register_extension_ = true;
    const std::vector<nvimgcodecProcessingStatus_t>* expected_statuses_;
};

TEST_P(NvImageCodecsCanDecodeApiTest, CanDecode)
{
    std::vector<nvimgcodecProcessingStatus_t> processing_statuses(expected_statuses_->size());
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCanDecode(
        decoder_, streams_.data(), images_.data(), streams_.size(), &params_, processing_statuses.data(), force_format_));
    for (size_t i = 0; i < streams_.size(); ++i) {
        if (register_extension_) {
            EXPECT_EQ((*expected_statuses_)[i], processing_statuses[i]);
        } else {
            EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED, processing_statuses[i]);
        }
    }
}
// clang-format off
test_case_tuple_t can_decode_test_cases[] = {
    {&statuses_to_return_case1_with_force_format_true, true, &statuses_to_expect_for_case1_with_force_format_true},
    {&statuses_to_return_case1_with_force_format_false, false, &statuses_to_expect_for_case1_with_force_format_false},
    {&statuses_to_return_case2_with_force_format_true, true, &statuses_to_expect_for_case2_with_force_format_true},
    {&statuses_to_return_case2_with_force_format_false, false, &statuses_to_expect_for_case2_with_force_format_false},
    {&statuses_to_return_case3_with_force_format_true, true, &statuses_to_expect_for_case3_with_force_format_true},
    {&statuses_to_return_case3_with_force_format_false, false, &statuses_to_expect_for_case3_with_force_format_false},
    {&statuses_to_return_case4_with_force_format_true, true, &statuses_to_expect_for_case4_with_force_format_true},
    {&statuses_to_return_case4_with_force_format_false, false, &statuses_to_expect_for_case4_with_force_format_false}};
// clang-format on

INSTANTIATE_TEST_SUITE_P(API_CAN_DECODE, NvImageCodecsCanDecodeApiTest, Combine(::testing::ValuesIn(can_decode_test_cases), ::testing::Values(true, false)));

}} // namespace nvimgcodec::test
