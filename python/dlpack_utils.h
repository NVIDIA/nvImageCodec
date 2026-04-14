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

// Shared state that keeps tensor data alive across multiple DLPack exports.
// This struct serves as the single ownership root for image buffers and CUDA events,
// enabling "long-lived owner + shared_ptr fan-out" pattern. When multiple DLPack
// capsules are created from the same image, they all share a reference to this state,
// ensuring the underlying data remains valid as long as any capsule exists.
//
// For imported tensors, this struct also stores the external DLManagedTensor* to keep
// it alive, while DLPackTensor creates its own internal metadata copy for re-export.
struct DLPackTensorSharedState
{
    ImageBuffer image_buffer;
    std::shared_ptr<std::remove_pointer<cudaEvent_t>::type> dlpack_cuda_event;
    ILogger* logger;
    DLManagedTensor* external_tensor;  // Non-null for imported tensors, null for owned tensors

    // Constructor for owned tensors (export path)
    explicit DLPackTensorSharedState(ImageBuffer buffer, ILogger* log)
        : image_buffer(std::move(buffer)), logger(log), external_tensor(nullptr) {}
    
    // Constructor for imported tensors (keeps external tensor alive)
    explicit DLPackTensorSharedState(DLManagedTensor* ext_tensor, ILogger* log)
        : logger(log), external_tensor(ext_tensor) {}
    
    // Destructor calls external deleter if we're holding an external tensor
    ~DLPackTensorSharedState() {
        if (external_tensor && external_tensor->deleter) {
            external_tensor->deleter(external_tensor);
        }
    }
};

class DLPackTensor final
{
  public:
    DLPackTensor(ILogger* logger) noexcept;

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
    bool isInitialized() const {
      return shared_state_ != nullptr;
    }

    DLManagedTensor internal_dl_managed_tensor_;
    
    // Shared state for all exportable tensors.
    // - For owned tensors (export path): holds ImageBuffer and enables multiple exports
    // - For imported tensors (import path): holds external_tensor to keep it alive during re-export
    // - Only null for default-constructed or moved-from tensors
    std::shared_ptr<DLPackTensorSharedState> shared_state_;
    
    ILogger* logger_;
};

bool is_cuda_accessible(DLDeviceType devType);

} // namespace nvimgcodec
