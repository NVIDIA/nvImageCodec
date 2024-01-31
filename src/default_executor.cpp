/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "default_executor.h"
#include <cassert>
#include <thread>
#include "exception.h"
#include "log.h"

namespace nvimgcodec {

DefaultExecutor::DefaultExecutor(ILogger* logger, int num_threads)
    : logger_(logger)
    , desc_{NVIMGCODEC_STRUCTURE_TYPE_EXECUTOR_DESC, sizeof(nvimgcodecExecutorDesc_t), nullptr, this, &static_launch, &static_get_num_threads}
    , num_threads_(num_threads)
{
    if (num_threads_ == 0) {
        const auto cpu_cores_count = std::thread::hardware_concurrency();
        num_threads_ = cpu_cores_count ? cpu_cores_count : 1;
    }
}

DefaultExecutor::~DefaultExecutor()
{
}

nvimgcodecExecutorDesc_t* DefaultExecutor::getExecutorDesc()
{
    return &desc_;
}

nvimgcodecStatus_t DefaultExecutor::launch(int device_id, int sample_idx, void* task_context,
    void (*task)(int thread_id, int sample_idx, void* task_context))
{
    try {
        std::stringstream ss;
        ss << "Executor-" << device_id;
        auto it =
            device_id2thread_pool_.try_emplace(device_id, num_threads_, device_id, false, ss.str());

        auto& thread_pool = it.first->second;
        auto task_wrapper = [task_context, sample_idx, task](int thread_id) {
            task(thread_id, sample_idx, task_context); 
        };
        thread_pool.addWork(task_wrapper, 0, true);
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(logger_, e.what());
        return NVIMGCODEC_STATUS_INTERNAL_ERROR;
    }

    return NVIMGCODEC_STATUS_SUCCESS;
}

int DefaultExecutor::get_num_threads() const
{
    return num_threads_;
}

nvimgcodecStatus_t DefaultExecutor::static_launch(void* instance, int device_id, int sample_idx, void* task_context,
    void (*task)(int thread_id, int sample_idx, void* task_context))
{
    DefaultExecutor* handle = reinterpret_cast<DefaultExecutor*>(instance);
    return handle->launch(device_id, sample_idx, task_context, task);
}

int DefaultExecutor::static_get_num_threads(void* instance)
{
    DefaultExecutor* handle = reinterpret_cast<DefaultExecutor*>(instance);
    return handle->get_num_threads();
}

} // namespace nvimgcodec