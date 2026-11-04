/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace nvimgcodec { namespace test {

TEST(api_mismatch, wrong_instance_info_size)
{
    size_t wrong_size = sizeof(nvimgcodecInstanceCreateInfo_t) - 1;
    nvimgcodecInstance_t instance;
    nvimgcodecInstanceCreateInfo_t create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, wrong_size, 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_INTERNAL_ERROR, nvimgcodecInstanceCreate(&instance, &create_info));
}


struct LegacyExecutionParams
{
    nvimgcodecStructureType_t struct_type;
    size_t struct_size;
    void* struct_next;
    nvimgcodecDeviceAllocator_t* device_allocator;
    nvimgcodecPinnedAllocator_t* pinned_allocator;
    int max_num_cpu_threads;
    void* executor;
    int device_id;
    int pre_init;
    int num_backends;
    const void* backends;
};

static_assert(sizeof(LegacyExecutionParams) != sizeof(nvimgcodecExecutionParams_t), "Struct sizes are the same!");

TEST(api_mismatch, legacy_execution_params_are_rejected)
{
    nvimgcodecInstance_t instance;
    nvimgcodecInstanceCreateInfo_t create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance, &create_info));

    LegacyExecutionParams legacy_params{
        NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(LegacyExecutionParams), nullptr};
    legacy_params.device_id = NVIMGCODEC_DEVICE_CPU_ONLY;
    const auto* params = reinterpret_cast<const nvimgcodecExecutionParams_t*>(&legacy_params);

    nvimgcodecDecoder_t decoder = nullptr;
    nvimgcodecEncoder_t encoder = nullptr;
    EXPECT_EQ(NVIMGCODEC_STATUS_INTERNAL_ERROR, nvimgcodecDecoderCreate(instance, &decoder, params, nullptr));
    EXPECT_EQ(NVIMGCODEC_STATUS_INTERNAL_ERROR, nvimgcodecEncoderCreate(instance, &encoder, params, nullptr));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance));
}

}} // namespace nvimgcodec::test
