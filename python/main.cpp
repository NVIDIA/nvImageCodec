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
#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <nvimgcodec.h>
#include <nvimgcodec_version.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "backend.h"
#include "backend_kind.h"
#include "load_hint_policy.h"
#include "backend_params.h"
#include "chroma_subsampling.h"
#include "code_stream.h"
#include "color_spec.h"
#include "decode_params.h"
#include "decode_source.h"
#include "decoder.h"
#include "encode_params.h"
#include "encoder.h"
#include "image.h"
#include "image_buffer_kind.h"
#include "jpeg2k_bitstream_type.h"
#include "jpeg2k_encode_params.h"
#include "jpeg2k_prog_order.h"
#include "jpeg_encode_params.h"
#include "module.h"
#include "region.h"

#include <iostream>

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(nvimgcodec_impl, m, py::mod_gil_not_used())
{
    using namespace nvimgcodec;

    static Module module;

    m.doc() = R"pbdoc(

        nvImageCodec Python API reference

        This is the Python API reference for the NVIDIA® nvImageCodec library.
    )pbdoc";

    nvimgcodecProperties_t properties{NVIMGCODEC_STRUCTURE_TYPE_PROPERTIES, sizeof(nvimgcodecProperties_t), 0};
    nvimgcodecGetProperties(&properties);
    std::stringstream ver_ss{};
    ver_ss << NVIMGCODEC_STREAM_VER(properties.version);
    m.attr("__version__") = ver_ss.str();
    m.attr("__cuda_version__") = properties.cudart_version;

    BackendKind::exportToPython(m);
    LoadHintPolicy::exportToPython(m);
    BackendParams::exportToPython(m);
    Backend::exportToPython(m);
    ColorSpec::exportToPython(m);
    ChromaSubsampling::exportToPython(m);
    ImageBufferKind::exportToPython(m);
    Jpeg2kBitstreamType::exportToPython(m);
    Jpeg2kProgOrder::exportToPython(m);
    DecodeParams::exportToPython(m);
    JpegEncodeParams::exportToPython(m);
    Jpeg2kEncodeParams::exportToPython(m);
    EncodeParams::exportToPython(m);
    CodeStream::exportToPython(m, module.instance_);
    Region::exportToPython(m);
    DecodeSource::exportToPython(m, module.instance_);
    Image::exportToPython(m);
    Decoder::exportToPython(m, module.instance_, module.logger_.get());
    Encoder::exportToPython(m, module.instance_, module.logger_.get());
    Module::exportToPython(m, module.instance_);
}
