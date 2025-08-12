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
#include <cstdlib>
#include <iostream>
#include <string>

#include <pybind11/stl_bind.h>

#include <ilogger.h>
#include <log.h>

#include <nvimgcodec.h>
#include "image.h"
#include "module.h"

namespace nvimgcodec {

uint32_t verbosity2severity(int verbose)
{
    uint32_t result = 0;
    if (verbose >= 1)
        result |= NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_FATAL | NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_ERROR;
    if (verbose >= 2)
        result |= NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_WARNING;
    if (verbose >= 3)
        result |= NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_INFO;
    if (verbose >= 4)
        result |= NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEBUG;
    if (verbose >= 5)
        result |= NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_TRACE;

    return result;
}

Module::Module()
    : dbg_messenger_handle_(nullptr)
{
    int verbosity = 1;
    std::string verbosity_warning;
    char* v = std::getenv("PYNVIMGCODEC_VERBOSITY");
    try {
        if (v) {
            verbosity = std::stoi(v);
        }
    } catch (std::invalid_argument const& ex) {
        verbosity_warning = "PYNVIMGCODEC_VERBOSITY has wrong value";
    } catch (std::out_of_range const& ex) {
        verbosity_warning = "PYNVIMGCODEC_VERBOSITY has out of range value";
    }

    if (verbosity > 0) {
        dbg_messenger_ = std::make_unique<DefaultDebugMessenger>(verbosity2severity(verbosity), NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_ALL);
        logger_ = std::make_unique<Logger>("pynvimgcodec", dbg_messenger_.get());

        if (!verbosity_warning.empty()) {
            NVIMGCODEC_LOG_WARNING(logger_.get(), verbosity_warning);
        }
    } else {
        logger_ = std::make_unique<Logger>("pynvimgcodec");
    }

    nvimgcodecInstanceCreateInfo_t instance_create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};
    instance_create_info.load_builtin_modules = 1;
    instance_create_info.load_extension_modules = 1;
    instance_create_info.create_debug_messenger = verbosity > 0 ? 1 : 0;
    instance_create_info.debug_messenger_desc = verbosity > 0 ? dbg_messenger_->getDesc() : nullptr;

    nvimgcodecInstanceCreate(&instance_, &instance_create_info);

}

Module ::~Module()
{
    nvimgcodecInstanceDestroy(instance_);
}

void Module::exportToPython(py::module& m, nvimgcodecInstance_t instance, ILogger* logger)
{
    m.def(
         "as_image", [instance, logger](py::handle source, intptr_t cuda_stream) -> Image { return Image(instance, logger, source.ptr(), cuda_stream); },
         R"pbdoc(
        Wraps an external buffer as an image and ties the buffer lifetime to the image.
        
        At present, the image must always have a three-dimensional shape in the HWC layout (height, width, channels), 
        which is also known as the interleaved format, and be stored as a contiguous array in C-style, but rows can have additional padding.

        Args:
            source: Input DLPack tensor which is encapsulated in a PyCapsule object or other object 
                    with __cuda_array_interface__, __array_interface__ or __dlpack__ and __dlpack_device__ methods.
            
            cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place in the created Image.

        Returns:
            nvimgcodec.Image

        )pbdoc",
         "source"_a, "cuda_stream"_a = 0, py::keep_alive<0, 1>())
        .def(
            "as_images",
            [instance, logger](const std::vector<py::handle>& sources, intptr_t cuda_stream) -> std::vector<py::object> {
                std::vector<py::object> py_images;
                py_images.reserve(sources.size());
                for (auto& source : sources) {
                    Image img(instance, logger, source.ptr(), cuda_stream);
                    py::object py_img = py::cast(img);
                    py_images.push_back(py_img);
                    py::detail::keep_alive_impl(py_img, source);
                }
                return py_images;
            },
            R"pbdoc(
            Wraps all an external buffers as an images and ties the buffers lifetime to the images.
            
            At present, the image must always have a three-dimensional shape in the HWC layout (height, width, channels), 
            which is also known as the interleaved format, and be stored as a contiguous array in C-style, but rows can have additional padding.

            Args:
                sources: List of input DLPack tensors which is encapsulated in a PyCapsule objects or other objects 
                         with __cuda_array_interface__, __array_interface__ or __dlpack__ and __dlpack_device__ methods.
                
                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place in created Image.

            Returns:
                List of nvimgcodec.Image's
            )pbdoc",
            "sources"_a, "cuda_stream"_a = 0)
        .def(
            "from_dlpack",
            [instance, logger](py::handle source, intptr_t cuda_stream) -> Image { return Image(instance, logger, source.ptr(), cuda_stream); },
            R"pbdoc(
            Zero-copy conversion from a DLPack tensor to a image. 

            Args:
                source: Input DLPack tensor which is encapsulated in a PyCapsule object or other (array) object 
                        with __dlpack__  and __dlpack_device__ methods.
                
                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place in created Image.
            
            Returns:
                nvimgcodec.Image

            )pbdoc",
            "source"_a, "cuda_stream"_a = 0, py::keep_alive<0, 1>());
}

} // namespace nvimgcodec
