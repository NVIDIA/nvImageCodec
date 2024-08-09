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

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

namespace nvimgcodec { namespace {

bool cuInitChecked()
{
    static CUresult res = cuInit(0);
    return res == CUDA_SUCCESS;
}

// /**
//  * Simple RAII device handling:
//  * Switch to new device on construction, back to old
//  * device on destruction
//  */
class DeviceGuard
{
  public:
    /// @brief Saves current device id and restores it upon object destruction
    DeviceGuard()
        : old_context_(NULL)
    {
        if (!cuInitChecked()) {
            throw std::runtime_error(
                "Failed to load libcuda.so. "
                "Check your library paths and if the driver is installed correctly.");
        }

        if (cuCtxGetCurrent(&old_context_) != CUDA_SUCCESS) {
            throw std::runtime_error("Unable to get current CUDA context");
        }
    }

    /// @brief Saves current device id, sets a new one and switches back
    ///        to the original device upon object destruction.
    //         for device id < 0 it is no-op
    explicit DeviceGuard(int new_device)
        : old_context_(NULL)
    {
        if (new_device >= 0) {
            if (!cuInitChecked()) {
                throw std::runtime_error(
                    "Failed to load libcuda.so. "
                    "Check your library paths and if the driver is installed correctly.");
            }
            if (cuCtxGetCurrent(&old_context_) != CUDA_SUCCESS) {
                throw std::runtime_error("Unable to get current CUDA context");
            }
            if (cudaSetDevice(new_device) != cudaSuccess) {
                throw std::runtime_error("Unable to set device");
            }
        }
    }

    ~DeviceGuard()
    {
        if (old_context_ != NULL) {
            CUresult err = cuCtxSetCurrent(old_context_);
            if (err != CUDA_SUCCESS) {
                std::cerr << "Failed to recover from DeviceGuard: " << err << std::endl;
            }
        }
    }

  private:
    CUcontext old_context_;
};

} } // namespace nvimgcodec
