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

using ::testing::Bool;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

namespace nvimgcodec { namespace test {

namespace {

using test_case_tuple_t =
    std::tuple<const std::vector<std::vector<nvimgcodecProcessingStatus_t>>*, bool, const std::vector<nvimgcodecProcessingStatus_t>*>;

class MockEncoderPlugin
{
  public:
    explicit MockEncoderPlugin(const nvimgcodecFrameworkDesc_t* framework, const std::vector<nvimgcodecProcessingStatus_t>& return_status)
        : return_status_(return_status)
        , encoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_ENCODER_DESC, sizeof(nvimgcodecEncoderDesc_t), NULL,
              this,                // instance
              "mock_test_encoder", // id
              "bmp",               // codec_type
              NVIMGCODEC_BACKEND_KIND_CPU_ONLY, static_create, static_destroy, static_can_encode, static_encode_sample}
        , i_(0)
    {
    }
    nvimgcodecEncoderDesc_t* getEncoderDesc() { return &encoder_desc_; }

  private:
    static nvimgcodecStatus_t static_create(
        void* instance, nvimgcodecEncoder_t* encoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
    {
        *encoder = static_cast<nvimgcodecEncoder_t>(instance);
        return NVIMGCODEC_STATUS_SUCCESS;
    }
    static nvimgcodecStatus_t static_destroy(nvimgcodecEncoder_t encoder) { return NVIMGCODEC_STATUS_SUCCESS; }

    static nvimgcodecProcessingStatus_t static_can_encode(nvimgcodecEncoder_t encoder, const nvimgcodecCodeStreamDesc_t* code_stream,
        const nvimgcodecImageDesc_t* image, const nvimgcodecEncodeParams_t* params, int thread_idx)
    {
        auto handle = reinterpret_cast<MockEncoderPlugin*>(encoder);
        return handle->return_status_[handle->i_++]; // TODO(janton): this only works if we don't parallelize canEncode!
    }

    static nvimgcodecStatus_t static_encode_sample(nvimgcodecEncoder_t encoder, const nvimgcodecCodeStreamDesc_t* code_stream,
        const nvimgcodecImageDesc_t* image, const nvimgcodecEncodeParams_t* params, int thread_idx)
    {
        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    const std::vector<nvimgcodecProcessingStatus_t>& return_status_;
    nvimgcodecEncoderDesc_t encoder_desc_;
    int i_;
};

struct MockCodecExtensionFactory
{
  public:
    explicit MockCodecExtensionFactory(const std::vector<std::vector<nvimgcodecProcessingStatus_t>>* statuses)
        : desc_{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), nullptr, this, "test_extension", NVIMGCODEC_VER, NVIMGCODEC_VER, static_extension_create,
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
            encoders_.reserve(statuses_->size());
            for (auto& item : *statuses_) {
                encoders_.emplace_back(framework, item);
                framework->registerEncoder(framework->instance, encoders_.back().getEncoderDesc(), NVIMGCODEC_PRIORITY_NORMAL);
            }
        }
        ~Extension()
        {
            for (auto& item : encoders_) {
                framework_->unregisterEncoder(framework_->instance, item.getEncoderDesc());
            }
        }
        const nvimgcodecFrameworkDesc_t* framework_;
        std::vector<MockEncoderPlugin> encoders_;
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

class NvImageCodecsCanEncodeApiTest : public TestWithParam<std::tuple<test_case_tuple_t, bool> >
{
  public:
    NvImageCodecsCanEncodeApiTest() {}
    virtual ~NvImageCodecsCanEncodeApiTest() = default;

  protected:
    void SetUp() override
    {
        test_case_tuple_t test_case = std::get<0>(GetParam());
        mock_extension_ = std::make_unique<MockCodecExtensionFactory>(std::get<0>(test_case));
        force_format_ = std::get<1>(test_case);
        expected_statuses_ = std::get<2>(test_case);
        register_extension_ = std::get<1>(GetParam());

        nvimgcodecInstanceCreateInfo_t create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};

        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance_, &create_info));
        if (register_extension_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extension_, mock_extension_->getExtensionDesc()));
        }
        const char* options = nullptr;
        nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
        exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;
        exec_params.max_num_cpu_threads = 1;

        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderCreate(instance_, &encoder_, &exec_params, options));
        params_ = {NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(nvimgcodecEncodeParams_t), 0};
        image_info_ = {NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        image_info_.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        out_buffer_.resize(1);
        image_info_.buffer = out_buffer_.data();

        images_.clear();
        streams_.clear();

        for (size_t i = 0; i < expected_statuses_->size(); ++i) {
            nvimgcodecImageInfo_t out_image_info(image_info_);
            strcpy(out_image_info.codec_name,"bmp");
            nvimgcodecCodeStream_t code_stream = nullptr;
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamCreateToHostMem(instance_, &code_stream, (void*)this,
                                                    &NvImageCodecsCanEncodeApiTest::ResizeOutputBufferStatic, &out_image_info));
            streams_.push_back(code_stream);
            nvimgcodecImage_t image = nullptr;
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &image, &image_info_));
            images_.push_back(image);
        }
    }

    virtual void TearDown()
    {
        for (auto im : images_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageDestroy(im));
        }
        for (auto cs : streams_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(cs));
        }
        if (encoder_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderDestroy(encoder_));
        }
        if (extension_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(extension_));
        }
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance_));
        mock_extension_.reset();
    }

    unsigned char* ResizeOutputBuffer(size_t bytes) {
        out_buffer_.resize(bytes);
        return out_buffer_.data();
    }

    static unsigned char* ResizeOutputBufferStatic(void* ctx, size_t bytes) {
        auto handle = reinterpret_cast<NvImageCodecsCanEncodeApiTest*>(ctx);
        return handle->ResizeOutputBuffer(bytes);
    }

    nvimgcodecInstance_t instance_;
    nvimgcodecExtension_t extension_ = nullptr;
    std::unique_ptr<MockCodecExtensionFactory> mock_extension_;
    std::vector<unsigned char> out_buffer_;
    nvimgcodecImageInfo_t image_info_;
    nvimgcodecEncoder_t encoder_ = nullptr;
    nvimgcodecEncodeParams_t params_;
    std::vector<nvimgcodecImage_t> images_;
    std::vector<nvimgcodecCodeStream_t> streams_;
    bool force_format_ = true;
    bool register_extension_ = true;
    const std::vector<nvimgcodecProcessingStatus_t>* expected_statuses_;
};

TEST_P(NvImageCodecsCanEncodeApiTest, CanEncode)
{
    std::vector<nvimgcodecProcessingStatus_t> processing_statuses(expected_statuses_->size());
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderCanEncode(encoder_, images_.data(), streams_.data(), streams_.size(), &params_,
                                            processing_statuses.data(), force_format_));
    for (size_t i = 0; i < streams_.size(); ++i) {
        if (register_extension_) {
            EXPECT_EQ((*expected_statuses_)[i], processing_statuses[i]);
        } else {
            EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED, processing_statuses[i]);
        }
    }
}

test_case_tuple_t can_encode_test_cases[] = {
    {&statuses_to_return_case1_with_force_format_true, true, &statuses_to_expect_for_case1_with_force_format_true},
    {&statuses_to_return_case1_with_force_format_false, false, &statuses_to_expect_for_case1_with_force_format_false},
    {&statuses_to_return_case2_with_force_format_true, true, &statuses_to_expect_for_case2_with_force_format_true},
    {&statuses_to_return_case2_with_force_format_false, false, &statuses_to_expect_for_case2_with_force_format_false},
    {&statuses_to_return_case3_with_force_format_true, true, &statuses_to_expect_for_case3_with_force_format_true},
    {&statuses_to_return_case3_with_force_format_false, false, &statuses_to_expect_for_case3_with_force_format_false},
    {&statuses_to_return_case4_with_force_format_true, true, &statuses_to_expect_for_case4_with_force_format_true},
    {&statuses_to_return_case4_with_force_format_false, false, &statuses_to_expect_for_case4_with_force_format_false}};
// clang-format on

INSTANTIATE_TEST_SUITE_P(
    API_CAN_ENCODE, NvImageCodecsCanEncodeApiTest, Combine(::testing::ValuesIn(can_encode_test_cases), ::testing::Values(true, false)));

}} // namespace nvimgcodec::test
