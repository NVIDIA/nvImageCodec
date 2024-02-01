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

#include "device_guard.h"
#include "exception.h"

namespace nvimgcodec {

bool cuInitChecked()
{
    static CUresult res = cuInit(0);
    return res == CUDA_SUCCESS;
}

DeviceGuard::DeviceGuard() :
  old_context_(NULL) {
 if (!cuInitChecked()){
     throw Exception(INTERNAL_ERROR,
         "Failed to load libcuda.so. "
         "Check your library paths and if the driver is installed correctly.");
 }
  CHECK_CU(cuCtxGetCurrent(&old_context_));
}

DeviceGuard::DeviceGuard(int new_device) :
  old_context_(NULL) {
  if (new_device >= 0) {
     if (!cuInitChecked()) {
         throw Exception(INTERNAL_ERROR,
             "Failed to load libcuda.so. "
             "Check your library paths and if the driver is installed correctly.");
     }
     CHECK_CU(cuCtxGetCurrent(&old_context_));
     CHECK_CUDA(cudaSetDevice(new_device));
  }
}

DeviceGuard::~DeviceGuard() {
  if (old_context_ != NULL) {
    CUresult err = cuCtxSetCurrent(old_context_);
    if (err != CUDA_SUCCESS) {
         std::cerr << "Failed to recover from DeviceGuard: " << err << std::endl;
         std::terminate();
    }
  }
}

} // namespace nvimgcodec
