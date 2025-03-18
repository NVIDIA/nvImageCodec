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
            nvimgcds_backends[i].params = {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t), nullptr, 1.0f, NVIMGCODEC_LOAD_HINT_POLICY_FIXED};
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

std::vector<std::unique_ptr<Image>> Encoder::convertPyImagesToImages(
    const std::vector<py::handle>& py_images, std::vector<Image*>& images, intptr_t cuda_stream)
{
    images.clear();
    images.reserve(py_images.size());
    std::vector<std::unique_ptr<Image>> images_raii;
    images_raii.reserve(py_images.size());
    int i = 0;
    for (auto& pi : py_images) {
        Image* image = nullptr;
        try {
            image = pi.cast<Image*>();
        } catch (...) {
            try {
                auto image_uptr = std::make_unique<Image>(instance_, pi.ptr(), cuda_stream);
                images_raii.push_back(std::move(image_uptr));
                image = images_raii.back().get();
            } catch (const std::runtime_error& e) {
                image = nullptr;
                NVIMGCODEC_LOG_WARNING(logger_, "Input object #" << i << " cannot be converted to Image. " << e.what());
            }
        }

        images.push_back(image);
        i++;
    }
    return images_raii;
}

void Encoder::encode(const std::vector<Image*>& images, std::optional<EncodeParams> params_opt, intptr_t cuda_stream,
    std::function<void(size_t i, nvimgcodecImageInfo_t& out_image_info, nvimgcodecCodeStream_t* code_stream)> create_code_stream,
    std::function<void(size_t i, bool skip_item, nvimgcodecCodeStream_t code_stream)> post_encode_call_back)
{
    size_t orig_batch_size = images.size();
    std::vector<nvimgcodecImage_t> valid_images;
    valid_images.reserve(orig_batch_size);
    std::vector<nvimgcodecCodeStream_t> code_streams;
    code_streams.reserve(orig_batch_size);
    std::vector<int> valid_image_idx(orig_batch_size);

    EncodeParams params = params_opt.has_value() ? params_opt.value() : EncodeParams();

    params.jpeg2k_encode_params_.nvimgcodec_jpeg2k_encode_params_.struct_next = nullptr;
    params.jpeg_encode_params_.nvimgcodec_jpeg_encode_params_.struct_next = &params.jpeg2k_encode_params_.nvimgcodec_jpeg2k_encode_params_;
    params.encode_params_.struct_next = &params.jpeg_encode_params_.nvimgcodec_jpeg_encode_params_;

    for (size_t i = 0; i < orig_batch_size; i++) {
        if (images[i]) {
            auto img = images[i]->getNvImgCdcsImage();
            valid_images.push_back(img); 
            valid_image_idx[i] = valid_images.size() - 1;

            nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            nvimgcodecImageGetImageInfo(valid_images.back(), &image_info);

            nvimgcodecImageInfo_t out_image_info(image_info);
            out_image_info.chroma_subsampling = params.chroma_subsampling_;
            out_image_info.color_spec = params.color_spec_;
            out_image_info.struct_next = (void*)(&params.jpeg_encode_params_.nvimgcodec_jpeg_image_info_);

            code_streams.emplace_back();
            create_code_stream(i, out_image_info, &code_streams.back());
        } else {
            valid_image_idx[i] = -1; // negative number for excluded imagaes
        }
    }

    std::vector<nvimgcodecProcessingStatus_t> encode_status(valid_images.size());
    {
        py::gil_scoped_release release;
        nvimgcodecFuture_t encode_future;
        CHECK_NVIMGCODEC(nvimgcodecEncoderEncode(
            encoder_.get(), valid_images.data(), code_streams.data(), valid_images.size(), &params.encode_params_, &encode_future));
        nvimgcodecFutureWaitForAll(encode_future);

        size_t status_size;
        nvimgcodecFutureGetProcessingStatus(encode_future, encode_status.data(), &status_size);
        assert(status_size == encode_status.size());
        nvimgcodecFutureDestroy(encode_future);
    }

    for (size_t i = 0; i < orig_batch_size; ++i) {
        bool skip_image =  (valid_image_idx[i] < 0) || (encode_status[valid_image_idx[i]] != NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
        nvimgcodecCodeStream_t code_stream =  valid_image_idx[i] < 0 ? nullptr : code_streams[valid_image_idx[i]];
        post_encode_call_back(valid_image_idx[i], skip_image, code_stream);
    }

    {
        py::gil_scoped_release release;
        for (auto& cs : code_streams)
            nvimgcodecCodeStreamDestroy(cs);
    }
}

std::vector<py::object> Encoder::encode(
    const std::vector<Image*>& images, const std::string& codec, std::optional<EncodeParams> params, intptr_t cuda_stream)
{
    size_t orig_batch_size = images.size();
    std::vector<py::object> data_list;
    if (codec.empty()) {
        throw std::invalid_argument("Unspecified codec.");
    }
    std::string codec_name = codec[0] == '.' ? file_ext_to_codec(codec) : codec;
    if (codec_name.empty()) {
        throw std::invalid_argument("Unsupported codec.");
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
            py::gil_scoped_acquire acquire;
            auto handle = reinterpret_cast<PyObjectWrap*>(ctx);
            return handle->getBuffer(bytes);
        }

        PyObject* ptr_;
    };

    std::vector<PyObjectWrap> py_objects;
    py_objects.reserve(images.size());
    std::vector<int> valid_batch_idx;
    valid_batch_idx.reserve(orig_batch_size);

    auto create_code_stream = [&](size_t i, nvimgcodecImageInfo_t& out_image_info, nvimgcodecCodeStream_t* code_stream) -> void {
        strcpy(out_image_info.codec_name, codec_name.c_str());
        py_objects.emplace_back();
        CHECK_NVIMGCODEC(nvimgcodecCodeStreamCreateToHostMem(
            instance_, code_stream, (void*)&py_objects.back(), &PyObjectWrap::resize_buffer_static, &out_image_info));
        valid_batch_idx.push_back(i);
    };

    data_list.reserve(images.size());
    auto post_encode_callback = [&](size_t i, bool skip_item, nvimgcodecCodeStream_t code_stream) -> void {
        if (skip_item) {
            NVIMGCODEC_LOG_WARNING(logger_, "Something went wrong during encoding image #" << valid_batch_idx[i] << " there will be None on corresponding output position");
            data_list.push_back(py::none());

            // When skipping batch item because image could not be created, there is not need to decrement reference of output object
            // as for such items neither code stream nor output py_object were created.  
            if (code_stream && py_objects[i].ptr_) { 
                Py_DECREF(py_objects[i].ptr_);
            }
        } else {
            data_list.push_back(py::reinterpret_steal<py::object>(py_objects[i].ptr_));
        }
    };

    encode(images, params, cuda_stream, create_code_stream, post_encode_callback);

    return data_list;
}

std::vector<py::object> Encoder::encode(const std::vector<std::string>& file_names, const std::vector<Image*>& images, const std::string& codec,
    std::optional<EncodeParams> params, intptr_t cuda_stream)
{
    size_t orig_batch_size = images.size();
    if (file_names.size() != orig_batch_size) {
        throw std::invalid_argument("Size mismatch - filenames list has " + std::to_string(file_names.size()) + 
                            " items, but images list has " + std::to_string(images.size()) + " items.");
    }
    std::vector<py::object> encoded_files(file_names.size(), py::none());

    std::vector<int> valid_batch_idx;
    valid_batch_idx.reserve(orig_batch_size);

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
        CHECK_NVIMGCODEC(nvimgcodecCodeStreamCreateToFile(instance_, code_stream, file_names[i].c_str(), &out_image_info));
        valid_batch_idx.push_back(i);
    };
    auto post_encode_callback = [&](size_t i, bool skip_item, nvimgcodecCodeStream_t code_stream) -> void {
        if (skip_item) {
            NVIMGCODEC_LOG_WARNING(logger_, "Something went wrong during encoding image #" << valid_batch_idx[i] << " there will be None on corresponding output position");
        } else {
            encoded_files[valid_batch_idx[i]] = py::str(file_names[valid_batch_idx[i]]);
        }
    };
    encode(images, params, cuda_stream, create_code_stream, post_encode_callback);
    
    return encoded_files;
}


std::vector<py::object> Encoder::encode(
    const std::vector<py::handle>& py_images, const std::string& codec, std::optional<EncodeParams> params, intptr_t cuda_stream)
{
    std::vector<Image*> images;
    auto images_raii = convertPyImagesToImages(py_images, images, cuda_stream);
    return encode(images, codec, params, cuda_stream);
}

py::object Encoder::encode(py::handle image_source, const std::string& codec, std::optional<EncodeParams> params, intptr_t cuda_stream)
{
    if (py::isinstance<py::list>(image_source)) {
        auto images = image_source.cast<std::vector<py::handle>>();
        std::vector<py::object> data_list = encode(images, codec, params, cuda_stream);
        return py::cast(data_list);
    } else {
        std::vector<py::handle> images{image_source};

        std::vector<py::object> data_list = encode(images, codec, params, cuda_stream);
        if (data_list.size() == 1)
            return py::cast<py::object>(data_list[0]);
        else
            return py::none();
    }
}

py::object Encoder::encode(
    const std::string& file_name, py::handle image, const std::string& codec, std::optional<EncodeParams> params, intptr_t cuda_stream)
{
    std::vector<py::handle> images{image};
    std::vector<std::string> file_names{file_name};

    std::vector<py::object> encoded_files = encode(file_names, images, codec, params, cuda_stream);
    if (encoded_files.size() == 1)
        return py::cast<py::object>(encoded_files[0]);
    else
        return py::none();
}

std::vector<py::object> Encoder::encode(const std::vector<std::string>& file_names, const std::vector<py::handle>& py_images, const std::string& codec,
    std::optional<EncodeParams> params, intptr_t cuda_stream)
{
    std::vector<Image*> images;
    auto images_raii = convertPyImagesToImages(py_images, images, cuda_stream);
    return encode(file_names, images, codec, params, cuda_stream);
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
                Encoded file name, or None if the input image could not be encoded for any reason. 
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
