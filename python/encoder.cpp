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

#include "encoder.h"

#include <string.h>
#include <filesystem>
#include <iostream>

#include <ilogger.h>
#include <log.h>

#include "../src/file_ext_codec.h"
#include "backend.h"
#include "error_handling.h"

namespace fs = std::filesystem;

namespace nvimgcodec {

Encoder::Encoder(nvimgcodecInstance_t instance, ILogger* logger, int device_id, int max_num_cpu_threads,
    std::optional<std::vector<Backend>> backends, const std::string& options)
    : encoder_(nullptr)
    , instance_(instance)
    , logger_(logger)
{
    std::vector<nvimgcodecBackend_t> nvimgcds_backends(backends.has_value() ? backends.value().size() : 0);
    if (backends.has_value()) {
        for (size_t i = 0; i < backends.value().size(); ++i) {
            nvimgcds_backends[i] = backends.value()[i].backend_;
        }
    }

    auto backends_ptr = nvimgcds_backends.size() ? nvimgcds_backends.data() : nullptr;
    nvimgcodecEncoder_t encoder;
    nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
    exec_params.device_id = device_id;
    exec_params.max_num_cpu_threads = max_num_cpu_threads;
    exec_params.num_backends = nvimgcds_backends.size();
    exec_params.backends = backends_ptr;

    nvimgcodecEncoderCreate(instance, &encoder, &exec_params, options.c_str());
    encoder_ = std::shared_ptr<std::remove_pointer<nvimgcodecEncoder_t>::type>(
        encoder, [](nvimgcodecEncoder_t encoder) { nvimgcodecEncoderDestroy(encoder); });
}

Encoder::Encoder(nvimgcodecInstance_t instance, ILogger* logger, int device_id, int max_num_cpu_threads,
    std::optional<std::vector<nvimgcodecBackendKind_t>> backend_kinds, const std::string& options)
    : encoder_(nullptr)
    , instance_(instance)
    , logger_(logger)
{
    std::vector<nvimgcodecBackend_t> nvimgcds_backends(backend_kinds.has_value() ? backend_kinds.value().size() : 0);
    if (backend_kinds.has_value()) {
        for (size_t i = 0; i < backend_kinds.value().size(); ++i) {
            nvimgcds_backends[i].kind = backend_kinds.value()[i];
            nvimgcds_backends[i].params = {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t), nullptr, 1.0f};
        }
    }
    auto backends_ptr = nvimgcds_backends.size() ? nvimgcds_backends.data() : nullptr;
    nvimgcodecEncoder_t encoder;
    nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
    exec_params.device_id = device_id;
    exec_params.max_num_cpu_threads = max_num_cpu_threads;
    exec_params.num_backends = nvimgcds_backends.size();
    exec_params.backends = backends_ptr;
    nvimgcodecEncoderCreate(instance, &encoder, &exec_params, options.c_str());
    encoder_ = std::shared_ptr<std::remove_pointer<nvimgcodecEncoder_t>::type>(
        encoder, [](nvimgcodecEncoder_t encoder) { nvimgcodecEncoderDestroy(encoder); });
}

Encoder::~Encoder()
{
}

void Encoder::convertPyImagesToImages(const std::vector<py::handle>& py_images, std::vector<Image*>* images, intptr_t cuda_stream)
{
    images->reserve(py_images.size());
    int i = 0;
    for (auto& pi : py_images) {
        Image* image = nullptr;
        try {
            image = pi.cast<Image*>();
        } catch (...) {
            image = new Image(instance_, pi.ptr(), cuda_stream);
        }
        if (image) {
            images->push_back(image);
        } else {
            NVIMGCODEC_LOG_WARNING(logger_, "Input object #" << i << " cannot be interpreted as Image.  It will not be included in output");
        }
        i++;
    }
}

py::object Encoder::encode(py::handle image_source, const std::string& codec, std::optional<EncodeParams> params, intptr_t cuda_stream)
{
    if (py::isinstance<py::list>(image_source)) {
        auto images = image_source.cast<std::vector<py::handle>>();
        std::vector<py::bytes> data_list = encode(images, codec, params, cuda_stream);
        return py::cast(data_list);
    } else {
        std::vector<py::handle> images{image_source};

        std::vector<py::bytes> data_list = encode(images, codec, params, cuda_stream);
        if (data_list.size() == 1)
            return py::cast<py::object>(data_list[0]);
        else
            return py::none();
    }
}

void Encoder::encode(
    const std::string& file_name, py::handle image, const std::string& codec, std::optional<EncodeParams> params, intptr_t cuda_stream)
{
    std::vector<py::handle> images{image};
    std::vector<std::string> file_names{file_name};

    encode(file_names, images, codec, params, cuda_stream);
}

void Encoder::encode(const std::vector<Image*>& images, std::optional<EncodeParams> params_opt, intptr_t cuda_stream,
    std::function<void(size_t i, nvimgcodecImageInfo_t& out_image_info, nvimgcodecCodeStream_t* code_stream)> create_code_stream,
    std::function<void(size_t i, bool skip_item, nvimgcodecCodeStream_t code_stream)> post_encode_call_back)
{
    std::vector<nvimgcodecCodeStream_t> code_streams(images.size());
    std::vector<nvimgcodecImage_t> int_images(images.size());
    EncodeParams params = params_opt.has_value() ? params_opt.value() : EncodeParams();

    params.jpeg2k_encode_params_.nvimgcodec_jpeg2k_encode_params_.struct_next = nullptr;
    params.jpeg_encode_params_.nvimgcodec_jpeg_encode_params_.struct_next = &params.jpeg2k_encode_params_.nvimgcodec_jpeg2k_encode_params_;
    params.encode_params_.struct_next = &params.jpeg_encode_params_.nvimgcodec_jpeg_encode_params_;

    for (size_t i = 0; i < images.size(); i++) {
        int_images[i] = images[i]->getNvImgCdcsImage();

        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        nvimgcodecImageGetImageInfo(int_images[i], &image_info);

        nvimgcodecImageInfo_t out_image_info(image_info);
        out_image_info.chroma_subsampling = params.chroma_subsampling_;
        out_image_info.color_spec = params.color_spec_;
        out_image_info.struct_next = (void*)(&params.jpeg_encode_params_.nvimgcodec_jpeg_image_info_);

        create_code_stream(i, out_image_info, &code_streams[i]);
    }
    nvimgcodecFuture_t encode_future;
    CHECK_NVIMGCODEC(nvimgcodecEncoderEncode(
        encoder_.get(), int_images.data(), code_streams.data(), images.size(), &params.encode_params_, &encode_future));
    nvimgcodecFutureWaitForAll(encode_future);
    size_t status_size;
    nvimgcodecFutureGetProcessingStatus(encode_future, nullptr, &status_size);
    std::vector<nvimgcodecProcessingStatus_t> encode_status(status_size);
    nvimgcodecFutureGetProcessingStatus(encode_future, &encode_status[0], &status_size);
    for (size_t i = 0; i < encode_status.size(); ++i) {
        if (encode_status[i] != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
             NVIMGCODEC_LOG_WARNING(logger_,"Something went wrong during encoding image #" << i << " it will not be included in output");
        }
        post_encode_call_back(i, encode_status[i] != NVIMGCODEC_PROCESSING_STATUS_SUCCESS, code_streams[i]);
    }
    nvimgcodecFutureDestroy(encode_future);
    for (auto& cs : code_streams) {
        nvimgcodecCodeStreamDestroy(cs);
    }
}

std::vector<py::bytes> Encoder::encode(
    const std::vector<py::handle>& py_images, const std::string& codec, std::optional<EncodeParams> params, intptr_t cuda_stream)
{
    std::vector<Image*> images;
    convertPyImagesToImages(py_images, &images, cuda_stream);
    return encode(images, codec, params, cuda_stream);
}

void Encoder::encode(const std::vector<std::string>& file_names, const std::vector<py::handle>& py_images, const std::string& codec,
    std::optional<EncodeParams> params, intptr_t cuda_stream)
{
    std::vector<Image*> images;
    convertPyImagesToImages(py_images, &images, cuda_stream);
    return encode(file_names, images, codec, params, cuda_stream);
}

std::vector<py::bytes> Encoder::encode(
    const std::vector<Image*>& images, const std::string& codec, std::optional<EncodeParams> params, intptr_t cuda_stream)
{
    std::vector<py::bytes> data_list;
    if (codec.empty()) {
        NVIMGCODEC_LOG_ERROR(logger_, "Unspecified codec.");
        return data_list;
    }
    std::string codec_name = codec[0] == '.' ? file_ext_to_codec(codec) : codec;
    if (codec_name.empty()) {
        NVIMGCODEC_LOG_ERROR(logger_, "Unsupported codec.");
        return data_list;
    }

    struct PyObjectWrap
    {
        unsigned char* getBuffer(size_t bytes)
        {
            ptr_ = PyBytes_FromStringAndSize(nullptr, bytes);
            return (unsigned char*)PyBytes_AsString(ptr_);
        }

        static unsigned char* resize_buffer_static(void* ctx, size_t bytes)
        {
            auto handle = reinterpret_cast<PyObjectWrap*>(ctx);
            return handle->getBuffer(bytes);
        }

        PyObject* ptr_;
    };

    std::vector<PyObjectWrap> py_objects(images.size());

    auto create_code_stream = [&](size_t i, nvimgcodecImageInfo_t& out_image_info, nvimgcodecCodeStream_t* code_stream) -> void {
        strcpy(out_image_info.codec_name, codec_name.c_str());
        CHECK_NVIMGCODEC(nvimgcodecCodeStreamCreateToHostMem(
            instance_, code_stream, (void*)&py_objects[i], &PyObjectWrap::resize_buffer_static, &out_image_info));
    };

    data_list.reserve(images.size());
    auto post_encode_callback = [&](size_t i, bool skip_item, nvimgcodecCodeStream_t code_stream) -> void {
        if (skip_item && py_objects[i].ptr_) {
            Py_DECREF(py_objects[i].ptr_);
        } else {
            data_list.push_back(py::reinterpret_steal<py::object>(py_objects[i].ptr_));
        }
    };

    encode(images, params, cuda_stream, create_code_stream, post_encode_callback);

    return data_list;
}

void Encoder::encode(const std::vector<std::string>& file_names, const std::vector<Image*>& images, const std::string& codec,
    std::optional<EncodeParams> params, intptr_t cuda_stream)
{
    std::vector<nvimgcodecCodeStream_t> code_streams(images.size());
    auto create_code_stream = [&](size_t i, nvimgcodecImageInfo_t& out_image_info, nvimgcodecCodeStream_t* code_stream) -> void {
        std::string codec_name{};

        if (codec.empty()) {
            auto file_extension = fs::path(file_names[i]).extension();
            codec_name = file_ext_to_codec(file_extension);
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
        CHECK_NVIMGCODEC(nvimgcodecCodeStreamCreateToFile(instance_, code_stream, file_names[i].c_str(), &out_image_info));
    };
    auto post_encode_callback = [&](size_t i, bool skip_item, nvimgcodecCodeStream_t code_stream) -> void {};
    encode(images, params, cuda_stream, create_code_stream, post_encode_callback);
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
    py::class_<Encoder>(m, "Encoder")
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
        .def(py::init<>(
                 [instance, logger](int device_id, int max_num_cpu_threads, std::optional<std::vector<nvimgcodecBackendKind_t>> backend_kinds,
                     const std::string& options) { return new Encoder(instance, logger, device_id, max_num_cpu_threads, backend_kinds, options); }),
            R"pbdoc(
            Initialize encoder.

            Args:
                device_id: Device id to execute encoding on.

                max_num_cpu_threads: Max number of CPU threads in default executor (0 means default value equal to number of cpu cores)
                
                backend_kinds: List of allowed backend kinds. If empty or None, all backends are allowed with default parameters.
                
                options: Encoder specific options.

            )pbdoc",
            "device_id"_a = NVIMGCODEC_DEVICE_CURRENT, "max_num_cpu_threads"_a = 0, "backend_kinds"_a = py::none(),
            "options"_a = ":fancy_upsampling=0")
        .def("encode", py::overload_cast<py::handle, const std::string&, std::optional<EncodeParams>, intptr_t>(&Encoder::encode),
            R"pbdoc(
            Encode image(s) to buffer(s).

            Args:
                image_s: Image or list of images to encode
                
                codec: String that defines the output format e.g.'jpeg2k'. When it is file extension it must include a leading period e.g. '.jp2'.
                
                params: Encode parameters.
                
                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

            Returns:
                Buffer or list of buffers with compressed code stream(s). None if the image(s) cannot be encoded because of any reason.
            )pbdoc",
            "image_s"_a, "codec"_a, "params"_a = py::none(), "cuda_stream"_a = 0)
        .def("write",
            py::overload_cast<const std::string&, py::handle, const std::string&, std::optional<EncodeParams>, intptr_t>(&Encoder::encode),
            R"pbdoc(
            Encode image to file.

            Args:
                file_name: File name to save encoded code stream.

                image: Image to encode

                codec: String that defines the output format e.g.'jpeg2k'. When it is file extension it must include a 
                leading period e.g. '.jp2'. If codec is not specified, it is deducted based on file extension. 
                If there is no extension by default 'jpeg' is choosen. 

                params: Encode parameters.

                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

            Returns: 
                None
            )pbdoc",
            "file_name"_a, "image"_a, "codec"_a = "", "params"_a = py::none(), "cuda_stream"_a = 0)
        .def("write",
            py::overload_cast<const std::vector<std::string>&, const std::vector<py::handle>&, const std::string&,
                std::optional<EncodeParams>, intptr_t>(&Encoder::encode),
            R"pbdoc(
            Encode batch of images to files.

            Args:
                file_names: List of file names to save encoded code streams.

                images: List of images to encode.

                codec: String that defines the output format e.g.'jpeg2k'. When it is file extension it must include a 
                leading period e.g. '.jp2'. If codec is not specified, it is deducted based on file extension. 
                If there is no extension by default 'jpeg' is choosen. (optional)
                
                params: Encode parameters.
                
                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

            Returns:
                List of buffers with compressed code streams.
            )pbdoc",
            "file_names"_a, "images"_a, "codec"_a = "", "params"_a = py::none(), "cuda_stream"_a = 0)
        .def("__enter__", &Encoder::enter, "Enter the runtime context related to this encoder.")
        .def("__exit__", &Encoder::exit,
            "Exit the runtime context related to this encoder and releases allocated resources."
            "exc_type"_a = py::none(),
            "exc_value"_a = py::none(), "traceback"_a = py::none());
}

} // namespace nvimgcodec
