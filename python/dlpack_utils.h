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

#include <variant>
#include <memory>
#include <nvimgcodec.h>

#include <dlpack/dlpack.h>

#include <pybind11/buffer_info.h>
#include <pybind11/pybind11.h>

namespace nvimgcodec {

namespace py = pybind11;

class ILogger;
struct DeviceBuffer;
struct PinnedBuffer;

// Forward declare ImageBuffer type alias
// nvimgcodecImageInfo_t is used when buffer is externally managed and will contain information about originally passed data
using ImageBuffer = std::variant<std::shared_ptr<DeviceBuffer>, std::shared_ptr<PinnedBuffer>, nvimgcodecImageInfo_t>;

class DLPackTensor final
{
  public:
    DLPackTensor(ILogger* logger) noexcept;
    DLPackTensor(DLPackTensor&& that) noexcept;

    explicit DLPackTensor(ILogger* logger, DLManagedTensor* dl_managed_tensor);
    explicit DLPackTensor(ILogger* logger, const nvimgcodecImageInfo_t& image_info, 
                         ImageBuffer image_buffer = ImageBuffer{});

    ~DLPackTensor();

    const DLTensor* operator->() const;
    DLTensor* operator->();

    const DLTensor& operator*() const;
    DLTensor& operator*();

    void getImageInfo(nvimgcodecImageInfo_t* image_info);
    py::capsule getPyCapsule(intptr_t consumer_stream, cudaStream_t producer_stream);

  private:
    DLManagedTensor internal_dl_managed_tensor_;
    DLManagedTensor* dl_managed_tensor_ptr_;
    ImageBuffer image_buffer_;
    std::shared_ptr<std::remove_pointer<cudaEvent_t>::type> dlpack_cuda_event_;
    ILogger* logger_;
};

bool is_cuda_accessible(DLDeviceType devType);

} // namespace nvimgcodec
