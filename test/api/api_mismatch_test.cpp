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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <nvimgcodec.h>

namespace nvimgcodec { namespace test {

using ::testing::ByMove;
using ::testing::Return;
using ::testing::ReturnPointee;
using ::testing::ReturnRef;
using ::testing::Eq;

TEST(api_mismatch, wrong_instance_info_size)
{
    size_t wrong_size = sizeof(nvimgcodecInstanceCreateInfo_t) - 1;
    nvimgcodecInstance_t instance;
    nvimgcodecInstanceCreateInfo_t create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, wrong_size, 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_INTERNAL_ERROR, nvimgcodecInstanceCreate(&instance, &create_info));
}


namespace v0_3_0_api {

typedef struct
{
    nvimgcodecStructureType_t struct_type;
    size_t struct_size;
    const void* struct_next;
    void* instance;
    nvimgcodecStatus_t (*launch)(void* instance, int device_id, int sample_idx, void* task_context,
        void (*task)(int thread_id, int sample_idx, void* task_context));
    int (*getNumThreads)(void* instance);
} nvimgcodecExecutorDesc_t;

typedef struct
{
    nvimgcodecStructureType_t struct_type;
    size_t struct_size;
    void* struct_next;
    float load_hint;
} nvimgcodecBackendParams_t;

typedef struct
{
    nvimgcodecStructureType_t struct_type;
    size_t struct_size;
    void* struct_next;
    nvimgcodecBackendKind_t kind;
    nvimgcodecBackendParams_t params;
} nvimgcodecBackend_t;

typedef struct
{
    nvimgcodecStructureType_t struct_type;
    size_t struct_size;
    void* struct_next;
    nvimgcodecDeviceAllocator_t* device_allocator;
    nvimgcodecPinnedAllocator_t* pinned_allocator;
    int max_num_cpu_threads;
    nvimgcodecExecutorDesc_t* executor;
    int device_id;
    int pre_init;
    int num_backends;
    const nvimgcodecBackend_t* backends;
} nvimgcodecExecutionParams_t;

} // namespace v0_3_0_api

static_assert(sizeof(v0_3_0_api::nvimgcodecExecutorDesc_t) != sizeof(nvimgcodecExecutorDesc_t), "Struct sizes are the same!");

TEST(api_mismatch, old_api_doesnt_fail)
{
    nvimgcodecInstance_t instance;
    nvimgcodecInstanceCreateInfo_t create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance, &create_info));

    v0_3_0_api::nvimgcodecExecutionParams_t old_api_params{
        NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(v0_3_0_api::nvimgcodecExecutionParams_t), 0};

    v0_3_0_api::nvimgcodecExecutorDesc_t old_api_executor{
        NVIMGCODEC_STRUCTURE_TYPE_EXECUTOR_DESC, sizeof(v0_3_0_api::nvimgcodecExecutorDesc_t), 0};
    old_api_params.executor = &old_api_executor;

    v0_3_0_api::nvimgcodecBackend_t backends[3];
    backends[0] = v0_3_0_api::nvimgcodecBackend_t{NVIMGCODEC_STRUCTURE_TYPE_BACKEND, sizeof(v0_3_0_api::nvimgcodecBackend_t), nullptr,
        NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY,
        v0_3_0_api::nvimgcodecBackendParams_t{
            NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(v0_3_0_api::nvimgcodecBackendParams_t), 0, 0.5f}};
    backends[1] = v0_3_0_api::nvimgcodecBackend_t{NVIMGCODEC_STRUCTURE_TYPE_BACKEND, sizeof(v0_3_0_api::nvimgcodecBackend_t), nullptr,
        NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU,
        v0_3_0_api::nvimgcodecBackendParams_t{
            NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(v0_3_0_api::nvimgcodecBackendParams_t), 0, 1.0f}};
    backends[2] = v0_3_0_api::nvimgcodecBackend_t{NVIMGCODEC_STRUCTURE_TYPE_BACKEND, sizeof(v0_3_0_api::nvimgcodecBackend_t), nullptr,
        NVIMGCODEC_BACKEND_KIND_CPU_ONLY,
        v0_3_0_api::nvimgcodecBackendParams_t{
            NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(v0_3_0_api::nvimgcodecBackendParams_t), 0, 1.0f}};
    old_api_params.num_backends = 3;
    old_api_params.backends = backends;

    nvimgcodecExecutionParams_t* params = reinterpret_cast<nvimgcodecExecutionParams_t*>(&old_api_params);

    nvimgcodecDecoder_t decoder = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCreate(instance, &decoder, params, nullptr));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDestroy(decoder));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance));
}

}} // namespace nvimgcodec::test
