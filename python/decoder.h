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
#include <optional>

#include <nvimgcodec.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "image.h"
#include "decode_params.h"
#include "backend.h"

namespace nvimgcodec {

namespace py = pybind11;
using namespace py::literals;

class ILogger;

class Decoder
{
  public:
    Decoder(nvimgcodecInstance_t instance, ILogger* logger, int device_id, int max_num_cpu_threads, std::optional<std::vector<Backend>> backends,
        const std::string& options);
    Decoder(nvimgcodecInstance_t instance, ILogger* logger, int device_id, int max_num_cpu_threads,
        std::optional<std::vector<nvimgcodecBackendKind_t>> backend_kinds, const std::string& options);
    ~Decoder();

    py::object decode(const std::string& file_name, std::optional<DecodeParams> params, intptr_t cuda_stream);
    py::object decode(py::array_t<uint8_t> data, std::optional<DecodeParams> params, intptr_t cuda_stream);
    py::object decode(py::bytes data, std::optional<DecodeParams> params, intptr_t cuda_stream);

    std::vector<py::object> decode(const std::vector<std::string>& file_names, std::optional<DecodeParams> params, intptr_t cuda_stream);
    std::vector<py::object> decode(const std::vector<py::array_t<uint8_t>>& data_list, std::optional<DecodeParams> params, intptr_t cuda_stream);
    std::vector<py::object> decode(const std::vector<py::bytes>& data_list, std::optional<DecodeParams> params, intptr_t cuda_stream);

    py::object enter();
    void exit(const std::optional<pybind11::type>& exc_type, const std::optional<pybind11::object>& exc_value,
        const std::optional<pybind11::object>& traceback);

    static void exportToPython(py::module& m, nvimgcodecInstance_t instance, ILogger* logger);

  private:
    std::vector<py::object> decode(std::vector<nvimgcodecCodeStream_t>& code_streams, std::optional<DecodeParams> params, intptr_t cuda_stream);

    std::shared_ptr<std::remove_pointer<nvimgcodecDecoder_t>::type> decoder_;
    nvimgcodecInstance_t instance_;
    ILogger* logger_;
};

} // namespace nvimgcodec
