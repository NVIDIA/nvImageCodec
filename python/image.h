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
#include <variant>
#include <cstddef>
#include <optional>

#include <nvimgcodec.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dlpack_utils.h"
#include "cuda_stream_utils.h"

namespace nvimgcodec {

namespace py = pybind11;
using namespace py::literals;

class ILogger;
struct DeviceBuffer;
struct PinnedBuffer;
struct PageableHostBuffer;

class Image
{
  public:
    Image(nvimgcodecInstance_t instance, ILogger* logger, nvimgcodecImageInfo_t* image_info, int device_id = NVIMGCODEC_DEVICE_CURRENT);
    Image(nvimgcodecInstance_t instance, ILogger* logger, PyObject* o, CudaStreamArg cuda_stream = {},
          std::optional<nvimgcodecSampleFormat_t> sample_format = std::nullopt,
          std::optional<nvimgcodecColorSpec_t> color_spec = std::nullopt,
          std::optional<int> precision = std::nullopt);

    int getWidth() const;
    int getHeight() const;
    int getNdim() const;
    int num_channels() const;

    nvimgcodecImageBufferKind_t getBufferKind() const;
    size_t size() const; // size of the data in the buffer
    size_t capacity() const; // size of the buffer

    py::dict array_interface() const;
    py::dict cuda_interface() const;
    py::tuple getPlanes(py::object self) const;
    py::object toNumpyArray(py::object dtype_obj, py::object copy_obj) const;

    py::tuple shape() const;
    py::tuple strides() const;
    py::object dtype() const;
    int precision() const;
    nvimgcodecSampleFormat_t getSampleFormat() const;
    nvimgcodecColorSpec_t getColorSpec() const;
    nvimgcodecChromaSubsampling_t getChromaSubsampling() const;
    std::optional<int> getDeviceId() const;
    std::optional<intptr_t> getCudaStream() const;

    // Pin a Python stream-protocol object whose lifetime should outlive this Image.
    // No-op for bare integer handles (owner is none). Does not affect detachFromUserStream.
    void setCudaStreamOwner(py::object owner);

    py::capsule dlpack(py::object stream) const;
    const py::tuple getDlpackDevice() const;

    py::object cpu();
    py::object cuda(bool synchronize, int device_id = NVIMGCODEC_DEVICE_CURRENT, CudaStreamArg cuda_stream = {});

    void reuse(nvimgcodecImageInfo_t* image_info);

    // Rebind any internal buffer's deleter so it no longer references the
    // CUDA stream it was allocated on. After this call, the caller may
    // destroy the originating stream at any time without affecting correct
    // cleanup of this image. Safe no-op for externally-managed buffers.
    void detachFromUserStream() noexcept;

    nvimgcodecImage_t getNvImgCdcsImage() const;
    static void exportToPython(py::module& m);

  private:
    void initImageInfoFromDLPack(nvimgcodecImageInfo_t* image_info, py::capsule cap,
        std::optional<nvimgcodecSampleFormat_t> sample_format = std::nullopt,
        std::optional<nvimgcodecColorSpec_t> color_spec = std::nullopt);
    void initImageInfoFromInterfaceDict(const py::dict& d, nvimgcodecImageInfo_t* image_info,
        std::optional<nvimgcodecSampleFormat_t> sample_format = std::nullopt,
        std::optional<nvimgcodecColorSpec_t> color_spec = std::nullopt);
    void initInterfaceDictFromImageInfo(py::dict* d, bool is_cuda) const;

    void initBuffer(nvimgcodecImageInfo_t* image_info);
    void initDeviceBuffer(nvimgcodecImageInfo_t* image_info);
    void initHostBuffer(nvimgcodecImageInfo_t* image_info);
    
    bool hasInternallyManagedBuffer() const;

    nvimgcodecInstance_t instance_;
    ILogger* logger_;
    int device_id_;
    ImageBuffer img_buffer_;
    std::shared_ptr<std::remove_pointer<nvimgcodecImage_t>::type> image_;
    std::shared_ptr<DLPackTensor> dlpack_tensor_;
    // Optional Python object that owns the CUDA stream associated with this
    // Image (via __cuda_stream__). Kept so the foreign stream cannot be GC'd
    // while CAI / cuda_stream still report that handle. Bare int handles leave
    // this empty (caller owns lifetime).
    py::object cuda_stream_owner_;
};

} // namespace nvimgcodec
