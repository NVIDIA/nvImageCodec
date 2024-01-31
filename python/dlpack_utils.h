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

#include <nvimgcodec.h>

#include <dlpack/dlpack.h>

#include <pybind11/buffer_info.h>
#include <pybind11/pybind11.h>

namespace nvimgcodec {

namespace py = pybind11;

class DLPackTensor final
{
  public:
    DLPackTensor() noexcept;
    DLPackTensor(DLPackTensor&& that) noexcept;

    explicit DLPackTensor(DLManagedTensor* dl_managed_tensor);
    explicit DLPackTensor(const nvimgcodecImageInfo_t& image_info, std::shared_ptr<unsigned char> image_buffer);

    ~DLPackTensor();

    const DLTensor* operator->() const;
    DLTensor* operator->();

    const DLTensor& operator*() const;
    DLTensor& operator*();

    void getImageInfo(nvimgcodecImageInfo_t* image_info);
    py::capsule getPyCapsule();

  private:
    DLManagedTensor internal_dl_managed_tensor_;
    DLManagedTensor* dl_managed_tensor_ptr_;
    std::shared_ptr<unsigned char> image_buffer_;
};

bool is_cuda_accessible(DLDeviceType devType);

} // namespace nvimgcodec
