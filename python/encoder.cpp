/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "encoder.h"
#include "code_stream.h"

#include <string.h>
#include <filesystem>
#include <iostream>
#include <optional>
#include <vector>
#include <memory>

#include <ilogger.h>
#include <log.h>

#include "../src/file_ext_codec.h"
#include "backend.h"
#include "error_handling.h"
#include "imgproc/exception.h"

namespace fs = std::filesystem;

namespace nvimgcodec {

Encoder::Encoder(nvimgcodecInstance_t instance, ILogger* logger, int device_id, int max_num_cpu_threads,
    std::optional<std::vector<Backend>> backends, const std::string& options)
    : encoder_(nullptr)
    , instance_(instance)
    , logger_(logger)
{
    std::vector<nvimgcodecBackend_t> nvimgcds_backends(backends.has_value() ? backends->size() : 0);
    if (backends.has_value()) {
        for (size_t i = 0; i < backends->size(); ++i) {
            nvimgcds_backends[i] = (*backends)[i].backend_;
        }
    }

    auto backends_ptr = nvimgcds_backends.size() ? nvimgcds_backends.data() : nullptr;
    nvimgcodecEncoder_t encoder;
    nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
    exec_params.device_id = device_id;
    exec_params.max_num_cpu_threads = max_num_cpu_threads;
    exec_params.num_backends = nvimgcds_backends.size();
    exec_params.backends = backends_ptr;

    nvimgcodecStatus_t status = nvimgcodecEncoderCreate(instance, &encoder, &exec_params, options.c_str());
    if (status != NVIMGCODEC_STATUS_SUCCESS)
        throw Exception(ARCH_MISMATCH, "Could not create encoder. The requested backends are not supported.");

    encoder_ = std::shared_ptr<std::remove_pointer<nvimgcodecEncoder_t>::type>(
        encoder, [](nvimgcodecEncoder_t encoder) { nvimgcodecEncoderDestroy(encoder); });
}

Encoder::~Encoder()
{
}

void Encoder::encode_batch_impl(const std::vector<const Image*>& images, const std::optional<EncodeParams>& params_opt, intptr_t cuda_stream,
    std::function<void(size_t i, nvimgcodecImageInfo_t& out_image_info, nvimgcodecCodeStream_t* code_stream)> create_code_stream,
    std::function<void(size_t i, bool skip_item)> post_encode_call_back)
{
    size_t orig_batch_size = images.size();
    std::vector<nvimgcodecImage_t> valid_images;
    valid_images.reserve(orig_batch_size);
    std::vector<nvimgcodecCodeStream_t> code_streams;
    code_streams.reserve(orig_batch_size);
    std::vector<int> valid_image_idx(orig_batch_size); //for original index retuns position in valid_images or -1 for excluded images

    EncodeParams params = params_opt.has_value() ? params_opt.value() : EncodeParams();

    for (size_t i = 0; i < orig_batch_size; i++) {
        if (images[i]) {
            auto img = images[i]->getNvImgCdcsImage();
            valid_images.push_back(img); 
            valid_image_idx[i] = valid_images.size() - 1;

            nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            nvimgcodecImageGetImageInfo(valid_images.back(), &image_info);

            nvimgcodecImageInfo_t out_image_info(image_info);
            
            // Set chroma_subsampling: use user-specified value if provided,
            // otherwise default to GRAY for single-channel images, 444 for multi-channel
            if (params.chroma_subsampling_.has_value()) {
                out_image_info.chroma_subsampling = params.chroma_subsampling_.value();
            } else {
                uint32_t num_channels = image_info.plane_info[0].num_channels;
                out_image_info.chroma_subsampling = (num_channels < 3) 
                    ? NVIMGCODEC_SAMPLING_GRAY 
                    : NVIMGCODEC_SAMPLING_444;
            }
            
            if (params.color_spec_ != NVIMGCODEC_COLORSPEC_UNCHANGED) {
                out_image_info.color_spec = params.color_spec_;
            }
            out_image_info.struct_next = (void*)(&params.jpeg_encode_params_.nvimgcodec_jpeg_image_info_);

            code_streams.emplace_back();
            create_code_stream(i, out_image_info, &code_streams.back());
        } else {
            valid_image_idx[i] = -1; // negative number for excluded imagaes
        }
    }

    std::vector<nvimgcodecProcessingStatus_t> encode_status(valid_images.size());
    if (!valid_images.empty())
    {
        py::gil_scoped_release release;
        nvimgcodecFuture_t encode_future;
        CHECK_NVIMGCODEC(nvimgcodecEncoderEncode(
            encoder_.get(), valid_images.data(), code_streams.data(), valid_images.size(), params.handle(), &encode_future));
        nvimgcodecFutureWaitForAll(encode_future);

        size_t status_size;
        CHECK_NVIMGCODEC(nvimgcodecFutureGetProcessingStatus(encode_future, encode_status.data(), &status_size));
        assert(status_size == encode_status.size());
        CHECK_NVIMGCODEC(nvimgcodecFutureDestroy(encode_future));
    }

    for (size_t i = 0; i < orig_batch_size; ++i) {
        bool skip_image =  (valid_image_idx[i] < 0) || (encode_status[valid_image_idx[i]] != NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
        post_encode_call_back(i, skip_image);
    }

}


std::vector<py::object> Encoder::encode_batch(const std::vector<const Image*>& images, const std::string& codec,
        std::optional<std::vector<CodeStream*>> code_streams, const std::optional<EncodeParams>& params, intptr_t cuda_stream)
{
    if (codec.empty()) {
        throw std::invalid_argument("Unspecified codec.");
    }
    std::string codec_name = codec[0] == '.' ? file_ext_to_codec(codec) : codec;
    if (codec_name.empty()) {
        throw std::invalid_argument("Unsupported codec.");
    }
    size_t orig_batch_size = images.size();
    std::vector<py::object> data_list(orig_batch_size, py::none());

    std::vector<std::unique_ptr<CodeStream>> new_code_streams(orig_batch_size);

    auto create_code_stream = [&](size_t i, nvimgcodecImageInfo_t& out_image_info, nvimgcodecCodeStream_t* code_stream) -> void {
        strcpy(out_image_info.codec_name, codec_name.c_str());
        if (code_streams.has_value()) {
            code_streams.value()[i]->reuse(out_image_info); 
            *code_stream = code_streams.value()[i]->handle(); 
        } else {
            new_code_streams[i] = std::make_unique<CodeStream>(instance_, logger_, out_image_info);
            *code_stream = new_code_streams[i]->handle();
        }
    };

    auto post_encode_callback = [&](size_t i, bool skip_item) -> void {
        if (skip_item) {
            NVIMGCODEC_LOG_WARNING(logger_, "Something went wrong during encoding image #" << i << " there will be None on corresponding output position");
            //we have None on corresponding output position
        } else {
            if (code_streams.has_value()) {
                data_list[i] = py::cast(code_streams.value()[i], py::return_value_policy::move);
            } else {
                data_list[i] = py::cast(new_code_streams[i].release(), py::return_value_policy::take_ownership);
            }
        }
    };

    encode_batch_impl(images, params, cuda_stream, create_code_stream, post_encode_callback);

    return data_list;
}

py::object Encoder::encode_image(const Image* image, const std::string& codec, std::optional<CodeStream*> code_stream, const std::optional<EncodeParams>& params, intptr_t cuda_stream)
{
    if(!image) {
        return py::none();
    }
    std::optional<std::vector<CodeStream*>> code_streams;
    if (code_stream.has_value()) {
        code_streams = std::vector<CodeStream*>{code_stream.value()};
    }
    std::vector<const Image*> images{image};
    std::vector<py::object> code_streams_output = encode_batch(images, codec, code_streams, params, cuda_stream);
    return code_streams_output.size() == 1 ? code_streams_output[0] : py::none();
}

// Python object overloads with exception handling
py::object Encoder::encode(const py::object& image_s, const std::string& codec, std::optional<py::object> code_stream_s, const std::optional<EncodeParams>& params, intptr_t cuda_stream)
{
    if (image_s.is_none()) {
        return py::none();
    }

    // Check if it's a Python list
    if (py::isinstance<py::list>(image_s)) {
        py::list lst = image_s.cast<py::list>();
        std::vector<py::object> images;
        for (size_t i = 0; i < lst.size(); ++i) {
            images.push_back(lst[i]);
        }
        
        // Convert list of code_streams if provided
        std::optional<std::vector<CodeStream*>> code_streams;
        if (code_stream_s.has_value()) {
            // Check if code_stream is actually a list of CodeStreams
            if (py::isinstance<py::list>(*code_stream_s)) {
                py::list code_stream_list = code_stream_s->cast<py::list>();
                if (code_stream_list.size() != images.size()) {
                    throw std::invalid_argument("if multiple images are provided, code_stream_s must be a list of CodeStreams the same size as the number of images");
                }
                std::vector<CodeStream*> streams;
                for (size_t i = 0; i < code_stream_list.size(); ++i) {
                    streams.push_back(code_stream_list[i].cast<CodeStream*>());
                }
                code_streams = streams;
            } else {
                // If single code_stream provided raise error
                throw std::invalid_argument("if multiple images are provided, code_stream_s must be a list of CodeStreams the same size as the number of images");
            }
        }
        
        std::vector<Image> image_raii;
        std::vector<const Image*> image_ptrs = convertPyObjectsToImages(images, cuda_stream, image_raii);
        std::vector<py::object> result = encode_batch(image_ptrs, codec, code_streams, params, cuda_stream);
        
        // Return the list as a Python object
        return py::cast(result);
    } else {
        // Single object processing
        std::vector<py::object> single_image{image_s};
        std::vector<Image> image_raii;
        std::vector<const Image*> image_ptrs = convertPyObjectsToImages(single_image, cuda_stream, image_raii);
        
        if (image_ptrs[0] == nullptr) {
            return py::none();
        }

        // Convert single code_stream for single image
        std::optional<CodeStream*> single_code_stream;
        if (code_stream_s.has_value()) {
            if (py::isinstance<CodeStream>(*code_stream_s)) {
                single_code_stream = code_stream_s->cast<CodeStream*>();
            } else {
                throw std::invalid_argument("code_stream_s must be a single CodeStream if single image is provided");
            }
        }
        return encode_image(image_ptrs[0], codec, single_code_stream, params, cuda_stream);
    }
}

std::vector<py::object> Encoder::write_batch(const std::vector<std::string>& file_names, const std::vector<const Image*>& images, const std::string& codec,
    const std::optional<EncodeParams>& params, intptr_t cuda_stream)
{
    size_t orig_batch_size = images.size();
    if (file_names.size() != orig_batch_size) {
        throw std::invalid_argument("Size mismatch - filenames list has " + std::to_string(file_names.size()) + 
                            " items, but images list has " + std::to_string(images.size()) + " items.");
    }
    std::vector<py::object> encoded_files(file_names.size(), py::none());

    std::vector<CodeStream> code_streams;
    code_streams.reserve(images.size());

    auto create_code_stream = [&](size_t i, nvimgcodecImageInfo_t& out_image_info, nvimgcodecCodeStream_t* code_stream) -> void {
        std::string codec_name{};

        if (codec.empty()) {
            auto file_extension = fs::path(file_names[i]).extension();
            codec_name = file_ext_to_codec(file_extension.string().c_str());
            if (codec_name.empty()) {
                NVIMGCODEC_LOG_WARNING(logger_, "File '" << file_names[i] << "' without extension. As default choosing jpeg codec");
                codec_name = "jpeg";
            }
        } else {
            codec_name = codec[0] == '.' ? file_ext_to_codec(codec) : codec;
            if (codec_name.empty()) {
                NVIMGCODEC_LOG_WARNING(logger_, "Unsupported codec.  As default choosing jpeg codec");
                codec_name = "jpeg";
            }
        }
        strcpy(out_image_info.codec_name, codec_name.c_str());
        code_streams.emplace_back(instance_, logger_, file_names[i], out_image_info);
        *code_stream = code_streams.back().handle();
    };
    auto post_encode_callback = [&](size_t i, bool skip_item) -> void {
        if (skip_item) {
            NVIMGCODEC_LOG_WARNING(logger_, "Something went wrong during encoding image #" << i << " there will be None on corresponding output position");
            //we have None on corresponding output position
        } else {
            encoded_files[i] = py::str(file_names[i]);
        }
    };
    encode_batch_impl(images, params, cuda_stream, create_code_stream, post_encode_callback);
    
    return encoded_files;
}

py::object Encoder::write_image(
    const std::string& file_name, const Image* image, const std::string& codec, const std::optional<EncodeParams>& params, intptr_t cuda_stream)
{
    std::vector<const Image*> images{image};
    std::vector<std::string> file_names{file_name};

    std::vector<py::object> encoded_files = write_batch(file_names, images, codec, params, cuda_stream);
    if (encoded_files.size() == 1)
        return py::cast<py::object>(encoded_files[0]);
    else
        return py::none();
}

py::object Encoder::write(const std::string& file_name, const py::object& image, const std::string& codec, const std::optional<EncodeParams>& params, intptr_t cuda_stream)
{
    if (image.is_none()) {
        return py::none();
    }

    std::vector<py::object> single_image{image};
    std::vector<Image> image_raii;
    std::vector<const Image*> image_ptrs = convertPyObjectsToImages(single_image, cuda_stream, image_raii);
    
    if (image_ptrs[0] == nullptr) {
        return py::none();
    }

    return write_image(file_name, image_ptrs[0], codec, params, cuda_stream);
}

std::vector<py::object> Encoder::write(const std::vector<std::string>& file_names, const std::vector<py::object>& images, const std::string& codec,
    const std::optional<EncodeParams>& params, intptr_t cuda_stream)
{
    std::vector<Image> image_raii;
    std::vector<const Image*> image_ptrs = convertPyObjectsToImages(images, cuda_stream, image_raii);
    return write_batch(file_names, image_ptrs, codec, params, cuda_stream);
}

// Helper function to convert Python objects to Image pointers with exception handling
std::vector<const Image*> Encoder::convertPyObjectsToImages(const std::vector<py::object>& py_images, intptr_t cuda_stream, std::vector<Image>& image_raii)
{
    size_t orig_batch_size = py_images.size();
    std::vector<const Image*> image_ptrs;
    image_ptrs.reserve(orig_batch_size);
    image_raii.clear();
    image_raii.reserve(orig_batch_size);

    // Convert Python objects to Image pointers with exception handling
    for (size_t i = 0; i < orig_batch_size; ++i) {
        auto& image = py_images[i];
        if (image.is_none()) {
            image_ptrs.push_back(nullptr);
            continue;
        }

        try {
            const Image* image_ptr = image.cast<const Image*>();
            image_ptrs.push_back(image_ptr);
        } catch (...) {
            try {
                // Try to create Image from Python object
                image_raii.emplace_back(instance_, logger_, image.ptr(), cuda_stream);
                image_ptrs.push_back(&image_raii.back());
            } catch (const std::exception& e) {
                NVIMGCODEC_LOG_WARNING(logger_, "Failed to convert Python object #" << i << " to Image: " << e.what());
                image_ptrs.push_back(nullptr);
            }
        }
    }

    return image_ptrs;
}

py::object Encoder::enter()
{
    return py::cast(*this);
}

void Encoder::exit(const std::optional<pybind11::type>& exc_type, const std::optional<pybind11::object>& exc_value,
    const std::optional<pybind11::object>& traceback)
{
    encoder_.reset();
}

void Encoder::exportToPython(py::module& m, nvimgcodecInstance_t instance, ILogger* logger)
{
    // clang-format off
    py::class_<Encoder>(m, "Encoder", "Encoder for image encoding operations. "
        "It allows converting images to various compressed formats or save them to files. "
        "The encoding process can be customized with different parameters and options.")
        .def(py::init<>(
                 [instance, logger](int device_id, int max_num_cpu_threads, std::optional<std::vector<Backend>> backends,
                     const std::string& options) { return new Encoder(instance, logger, device_id, max_num_cpu_threads, backends, options); }),
            R"pbdoc(
            Initialize encoder.

            Args:
                device_id: Device id to execute encoding on.

                max_num_cpu_threads: Max number of CPU threads in default executor (0 means default value equal to number of cpu cores)
                
                backends: List of allowed backends. If empty, all backends are allowed with default parameters.
                
                options: Encoder specific options.

            )pbdoc",
            "device_id"_a = NVIMGCODEC_DEVICE_CURRENT, "max_num_cpu_threads"_a = 0, "backends"_a = py::none(), "options"_a = "")
        .def("encode", py::overload_cast<const py::object&, const std::string&, std::optional<py::object>, const std::optional<EncodeParams>&, intptr_t>(&Encoder::encode),
            R"pbdoc(
            Encode image(s) to CodeStream(s).

            Args:
                image_s: Image to encode or list of images to encode (can be any Python object that can be converted to Image, or a list of such objects)

                codec: String that defines the output format e.g.'jpeg2k'. When it is file extension it must include a leading period e.g. '.jp2'.
                
                code_stream_s: CodeStream or list of CodeStreams to encode image to. If None, a new CodeStream(s) will be created.
                
                params: Encode parameters.
                
                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

            Returns:
                For single image: CodeStream with encoded image, or None if the input cannot be converted to Image or if encoding failed.
                For list of images: List of CodeStreams with encoded images. If an input cannot be converted to Image or if encoding failed, the corresponding position will contain None.
            )pbdoc",
            "image_s"_a, "codec"_a, py::kw_only(), "code_stream_s"_a = py::none(), "params"_a = py::none(), "cuda_stream"_a = 0)
        .def("write",
            py::overload_cast<const std::string&, const py::object&, const std::string&, const std::optional<EncodeParams>&, intptr_t>(&Encoder::write),
            R"pbdoc(
            Encode image to file.

            Args:
                file_name: File name to save encoded code stream.

                image: Image to encode (can be any Python object that can be converted to Image)

                codec: String that defines the output format e.g.'jpeg2k'. When it is file extension it must include a 
                leading period e.g. '.jp2'. If codec is not specified, it is deducted based on file extension. 
                If there is no extension by default 'jpeg' is choosen. 

                params: Encode parameters.

                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

            Returns: 
                Encoded file name, or None if the input image could not be encoded for any reason. 
            )pbdoc",
            "file_name"_a, "image"_a, "codec"_a = "", "params"_a = py::none(), "cuda_stream"_a = 0)
        .def("write",
            py::overload_cast<const std::vector<std::string>&, const std::vector<py::object>&, const std::string&,
                const std::optional<EncodeParams>&, intptr_t>(&Encoder::write),
            R"pbdoc(
            Encode batch of images to files.

            Args:
                file_names: List of file names to save encoded code streams.

                images: List of images to encode (can contain any Python objects that can be converted to Image)

                codec: String that defines the output format e.g.'jpeg2k'. When it is file extension it must include a 
                leading period e.g. '.jp2'. If codec is not specified, it is deducted based on file extension. 
                If there is no extension by default 'jpeg' is choosen. (optional)
                
                params: Encode parameters.
                
                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

            Returns:
                List of encoded file names. If an image could not be encoded for any reason, the corresponding position in the list will contain None.
            )pbdoc",
            "file_names"_a, "images"_a, "codec"_a = "", "params"_a = py::none(), "cuda_stream"_a = 0)
        .def("__enter__", &Encoder::enter, "Enter the runtime context related to this encoder.")
        .def("__exit__", &Encoder::exit,
            "Exit the runtime context related to this encoder and releases allocated resources.",
            "exc_type"_a = py::none(),
            "exc_value"_a = py::none(), "traceback"_a = py::none());
    // clang-format on
}


} // namespace nvimgcodec
