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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>

#include "../src/decoder_worker.h"
#include "mock_codec.h"
#include "mock_image_decoder.h"
#include "mock_image_decoder_factory.h"
#include "mock_logger.h"

namespace nvimgcodec { namespace test {

using ::testing::_;
using ::testing::ByMove;
using ::testing::Return;
using ::testing::TestWithParam;
using ::testing::Values;

using test_case_tuple_t = std::tuple<std::vector<nvimgcodecBackendKind_t>, std::vector<nvimgcodecBackend_t>, int, int>;

class DecoderWorkerTest : public TestWithParam<test_case_tuple_t>
{
  public:
    virtual ~DecoderWorkerTest() = default;

  protected:
    void SetUp() override
    {
        auto given_backend_kinds = std::get<0>(GetParam());
        allowed_backends_ = std::get<1>(GetParam());
        int start_index = std::get<2>(GetParam());
        expected_return_index_ = std::get<3>(GetParam());
        codec_ = std::make_unique<MockCodec>();
        EXPECT_CALL(*codec_.get(), getDecodersNum()).WillRepeatedly(Return(given_backend_kinds.size()));
        image_dec_factories_.resize(given_backend_kinds.size());
        image_decs_.resize(given_backend_kinds.size());
        image_dec_ptrs_.resize(given_backend_kinds.size());
        for (int i = 0; i < given_backend_kinds.size(); ++i) {
            auto backend_kind = given_backend_kinds[i];
            auto image_dec = std::make_unique<MockImageDecoder>();
            image_dec_ptrs_[i] = image_dec.get();
            image_dec_factories_[i] = new MockImageDecoderFactory();
            MockImageDecoderFactory* image_dec_factory(image_dec_factories_[i]);
            ON_CALL(*image_dec_factory, getDecoderId()).WillByDefault(Return("decoder_id"));
            EXPECT_CALL(*image_dec_factory, getBackendKind()).WillRepeatedly(Return(backend_kind));
            EXPECT_CALL(*image_dec_factory, createDecoder(_, _)).WillRepeatedly(Return(ByMove(std::move(image_dec))));
            EXPECT_CALL(*codec_.get(), getDecoderFactory(i)).WillRepeatedly(Return(image_dec_factory));
        }
        exec_params_.struct_type = NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS;
        exec_params_.struct_size = sizeof(nvimgcodecExecutionParams_t);
        exec_params_.struct_next = nullptr;
        exec_params_.device_id = NVIMGCODEC_DEVICE_CURRENT;
        exec_params_.num_backends = allowed_backends_.size();
        exec_params_.backends = allowed_backends_.data();
        decoder_worker_ = std::make_unique<DecoderWorker>(&logger_, nullptr, &exec_params_, "", codec_.get(), start_index);
    }

    void TearDown() override
    {
        decoder_worker_.reset();
        image_dec_ptrs_.clear();
        for (auto f : image_dec_factories_) {
            delete f;
        }
        image_dec_factories_.clear();
        codec_.reset();
        allowed_backends_.clear();
    }

    MockLogger logger_;
    std::unique_ptr<MockCodec> codec_;
    std::vector<std::unique_ptr<MockImageDecoder>> image_decs_;
    std::vector<IImageDecoder*> image_dec_ptrs_;
    std::vector<MockImageDecoderFactory*> image_dec_factories_;
    nvimgcodecExecutionParams_t exec_params_;
    std::unique_ptr<DecoderWorker> decoder_worker_;
    std::vector<nvimgcodecBackend_t> allowed_backends_;
    int expected_return_index_;
};

TEST_P(DecoderWorkerTest, for_given_backend_kinds_and_allowed_backends_get_decoder_returns_correct_decoder)
{
    IImageDecoder* decoder = decoder_worker_->getDecoder();
    if (expected_return_index_ == -1) {
        EXPECT_EQ(decoder, nullptr);
    } else {
        EXPECT_EQ(decoder, image_dec_ptrs_[expected_return_index_]);
    }
}

// clang-format off

namespace {
test_case_tuple_t test_cases[] = {
    //For given one CPU backend and allowed CPU backend, return index 0
    {{NVIMGCODEC_BACKEND_KIND_CPU_ONLY}, 
     {{NVIMGCODEC_STRUCTURE_TYPE_BACKEND, sizeof(nvimgcodecBackend_t), 0, NVIMGCODEC_BACKEND_KIND_CPU_ONLY, {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t), 0, 50}}},
     0, 0
    },
    //For given one CPU backend and allowed all, return index 0
    {{NVIMGCODEC_BACKEND_KIND_CPU_ONLY}, 
     {},//all backends allowed
     0, 0
    },
    //For given 3 backends in order HW, GPU, CPU and allowed all, return index 0
    {{NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY, NVIMGCODEC_BACKEND_KIND_GPU_ONLY, NVIMGCODEC_BACKEND_KIND_CPU_ONLY}, 
    {},//all backends allowed
     0, 0
    },
    ///For given 3 backends in order HW, GPU, CPU, and allowed only HW, return index 0
    {{NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY, NVIMGCODEC_BACKEND_KIND_GPU_ONLY, NVIMGCODEC_BACKEND_KIND_CPU_ONLY}, 
    {{NVIMGCODEC_STRUCTURE_TYPE_BACKEND, sizeof(nvimgcodecBackend_t), 0, NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY, {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t), 0, 50}}},
     0, 0
    },
    //For given 3 backends in order HW, GPU, CPU, and allowed only GPU, return index 1
    {{NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY, NVIMGCODEC_BACKEND_KIND_GPU_ONLY, NVIMGCODEC_BACKEND_KIND_CPU_ONLY}, 
    {{NVIMGCODEC_STRUCTURE_TYPE_BACKEND, sizeof(nvimgcodecBackend_t), 0, NVIMGCODEC_BACKEND_KIND_GPU_ONLY, {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t), 0, 50}}},
     0, 1
    },
    //For given 3 backends in order HW,GPU,CPU and allowed CPU only, return decoder with index 2
    {{NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY, NVIMGCODEC_BACKEND_KIND_GPU_ONLY, NVIMGCODEC_BACKEND_KIND_CPU_ONLY}, 
    {{NVIMGCODEC_STRUCTURE_TYPE_BACKEND, sizeof(nvimgcodecBackend_t), 0, NVIMGCODEC_BACKEND_KIND_CPU_ONLY, {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t), 0, 50}}},
     0, 2
    },
    //For given 3 backends in order HW,GPU,CPU, and allowed only HYBRID (which is not present), return nullptr
    {{NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY, NVIMGCODEC_BACKEND_KIND_GPU_ONLY, NVIMGCODEC_BACKEND_KIND_CPU_ONLY}, 
    {{NVIMGCODEC_STRUCTURE_TYPE_BACKEND, sizeof(nvimgcodecBackend_t), 0, NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU, {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t), 0, 50}}},
     0, -1 //nullptr 
    },
    //For given 3 backends in order HW,GPU,CPU, and all allowed backends and start index 1 (first fallback), return decoder with index 1
    {{NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY, NVIMGCODEC_BACKEND_KIND_GPU_ONLY, NVIMGCODEC_BACKEND_KIND_CPU_ONLY}, 
    {},
     1, 1
    },
    //For given 3 backends in order HW,GPU,CPU, and allowed GPU and CPU, return decoder with index 1 (GPU)
    {{NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY, NVIMGCODEC_BACKEND_KIND_GPU_ONLY, NVIMGCODEC_BACKEND_KIND_CPU_ONLY}, 
    {{NVIMGCODEC_STRUCTURE_TYPE_BACKEND, sizeof(nvimgcodecBackend_t), 0, NVIMGCODEC_BACKEND_KIND_CPU_ONLY, {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t), 0, 50}},
     {NVIMGCODEC_STRUCTURE_TYPE_BACKEND, sizeof(nvimgcodecBackend_t), 0, NVIMGCODEC_BACKEND_KIND_GPU_ONLY, {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t), 0, 50}}
    },
     0, 1
    },
    //For given 3 backends in order HW,CPU, GPU, and allowed GPU and CPU, return decoder with index 1 (CPU)
    {{NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY, NVIMGCODEC_BACKEND_KIND_CPU_ONLY, NVIMGCODEC_BACKEND_KIND_GPU_ONLY}, 
    {{NVIMGCODEC_STRUCTURE_TYPE_BACKEND, sizeof(nvimgcodecBackend_t), 0, NVIMGCODEC_BACKEND_KIND_CPU_ONLY, {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t), 0, 50}},
     {NVIMGCODEC_STRUCTURE_TYPE_BACKEND, sizeof(nvimgcodecBackend_t), 0, NVIMGCODEC_BACKEND_KIND_GPU_ONLY, {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t), 0, 50}}
    },
     0, 1
    },

};
// clang-format on
} // namespace

INSTANTIATE_TEST_SUITE_P(DECODER_WORKER_GET_DECODER_TEST, DecoderWorkerTest, ::testing::ValuesIn(test_cases));

}} // namespace nvimgcodec::test