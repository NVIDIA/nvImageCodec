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

#include <optional>
#include <string>
#include <vector>

#include <nvimgcodec.h>
#include <pybind11/pybind11.h>

#include "backend.h"
#include "decode_params.h"
#include "encode_params.h"
#include "image.h"

namespace nvimgcodec {

namespace py = pybind11;
using namespace py::literals;

class ILogger;
class CodeStream;

class Encoder
{
  public:
    Encoder(nvimgcodecInstance_t instance, ILogger* logger, int device_id, int max_num_cpu_threads, std::optional<std::vector<Backend>> backends,
        const std::string& options);
    ~Encoder();

    py::object encode(const py::object& image_s, const std::string& codec, std::optional<py::object> code_stream_s = std::nullopt,
        const std::optional<EncodeParams>& params = std::nullopt, intptr_t cuda_stream = 0);

    py::object write(const std::string& file_name, const py::object& image, const std::string& codec, const std::optional<EncodeParams>& params, intptr_t cuda_stream);
    std::vector<py::object> write(const std::vector<std::string>& file_names, const std::vector<py::object>& images, const std::string& codec,
        const std::optional<EncodeParams>& params, intptr_t cuda_stream);

    py::object enter();
    void exit(const std::optional<pybind11::type>& exc_type, const std::optional<pybind11::object>& exc_value,
        const std::optional<pybind11::object>& traceback);

    static void exportToPython(py::module& m, nvimgcodecInstance_t instance, ILogger* logger);

  private:
    void encode_batch_impl(const std::vector<const Image*>& images, const std::optional<EncodeParams>& params, intptr_t cuda_stream,
        std::function<void(size_t i, nvimgcodecImageInfo_t& out_image_info, nvimgcodecCodeStream_t* code_stream)> create_code_stream,
        std::function<void(size_t i, bool skip_item)> post_encode_call_back);
    py::object encode_image(const Image* image, const std::string& codec, std::optional<CodeStream*> code_stream,
        const std::optional<EncodeParams>& params = std::nullopt, intptr_t cuda_stream = 0);
    std::vector<py::object> encode_batch(const std::vector<const Image*>& images, const std::string& codec, std::optional<std::vector<CodeStream*>> code_streams,
        const std::optional<EncodeParams>& params = std::nullopt, intptr_t cuda_stream = 0);
    py::object write_image(const std::string& file_name, const Image* image, const std::string& codec, const std::optional<EncodeParams>& params, intptr_t cuda_stream);
    std::vector<py::object> write_batch(const std::vector<std::string>& file_names, const std::vector<const Image*>& images, const std::string& codec,
        const std::optional<EncodeParams>& params, intptr_t cuda_stream);

    // Helper function to convert Python objects to Image pointers with exception handling
    std::vector<const Image*> convertPyObjectsToImages(const std::vector<py::object>& py_images, intptr_t cuda_stream, std::vector<Image>& image_raii);

    std::shared_ptr<std::remove_pointer<nvimgcodecEncoder_t>::type> encoder_;
    nvimgcodecInstance_t instance_;
    ILogger* logger_;
};

} // namespace nvimgcodec
