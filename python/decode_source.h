/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <sstream>
#include "code_stream.h"
#include "region.h"

namespace nvimgcodec {

namespace py = pybind11;
using namespace py::literals;

class DecodeSource
{
  public:
    DecodeSource(std::unique_ptr<CodeStream> code_stream,
                 std::optional<Region> region = {});
    DecodeSource(const CodeStream* code_stream_ptr,
                 std::optional<Region> region = {});
    ~DecodeSource();

    DecodeSource(DecodeSource&&) = default;
    DecodeSource& operator=(DecodeSource&&) = default;

    DecodeSource(const DecodeSource&) = delete;
    DecodeSource& operator=(DecodeSource const&) = delete;

    const CodeStream* code_stream() const;
    std::optional<Region> region() const;

    static void exportToPython(py::module& m, nvimgcodecInstance_t instance);

  private:
    std::unique_ptr<CodeStream> code_stream_;  // owned by this instance
    const CodeStream* code_stream_ptr_ = nullptr;  // externally provided
    std::optional<Region> region_;
};


std::ostream& operator<<(std::ostream& os, const DecodeSource& ds);

} // namespace nvimgcodec
