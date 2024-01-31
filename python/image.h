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

#include <string>
#include <vector>
#include <memory>

#include <nvimgcodec.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dlpack_utils.h"

namespace nvimgcodec {

namespace py = pybind11;
using namespace py::literals;

class Image
{
  public:
    Image(nvimgcodecInstance_t instance, nvimgcodecImageInfo_t* image_info);
    Image(nvimgcodecInstance_t instance, PyObject* o, intptr_t cuda_stream);

    int getWidth() const;
    int getHeight() const;
    int getNdim() const;
    nvimgcodecImageBufferKind_t getBufferKind() const;

    py::dict array_interface() const;
    py::dict cuda_interface() const;

    py::object shape() const;
    py::object dtype() const;
    int precision() const;

    py::capsule dlpack(py::object stream) const;
    const py::tuple getDlpackDevice() const;

    py::object cpu();
    py::object cuda(bool synchronize);

    nvimgcodecImage_t getNvImgCdcsImage() const;
    static void exportToPython(py::module& m);

  private:
    void initImageInfoFromInterfaceDict(const py::dict& d, nvimgcodecImageInfo_t* image_info);
    void initInterfaceDictFromImageInfo(const nvimgcodecImageInfo_t& image_info, py::dict* d);
    void initArrayInterface(const nvimgcodecImageInfo_t& image_info);
    void initCudaArrayInterface(const nvimgcodecImageInfo_t& image_info);
    void initCudaEventForDLPack();
    void initDLPack(nvimgcodecImageInfo_t* image_info, py::capsule cap);
    void initBuffer(nvimgcodecImageInfo_t* image_info);
    void initDeviceBuffer(nvimgcodecImageInfo_t* image_info);
    void initHostBuffer(nvimgcodecImageInfo_t* image_info);

    nvimgcodecInstance_t instance_;
    std::shared_ptr<unsigned char> img_host_buffer_;
    std::shared_ptr<unsigned char> img_buffer_;
    std::shared_ptr<std::remove_pointer<nvimgcodecImage_t>::type> image_;
    py::dict array_interface_;
    py::dict cuda_array_interface_;
    std::shared_ptr<DLPackTensor> dlpack_tensor_;
    std::shared_ptr<std::remove_pointer<cudaEvent_t>::type> dlpack_cuda_event_;
};

} // namespace nvimgcodec
