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

#include "decoder.h"

#include <iostream>
#include <string_view>

#include <ilogger.h>
#include <log.h>

#include "backend.h"
#include "error_handling.h"
#include "type_utils.h"

namespace nvimgcodec {

Decoder::Decoder(nvimgcodecInstance_t instance, ILogger* logger, int device_id, int max_num_cpu_threads,
    std::optional<std::vector<Backend>> backends, const std::string& options)
    : decoder_(nullptr)
    , instance_(instance)
    , logger_(logger)
{
    nvimgcodecDecoder_t decoder;
    std::vector<nvimgcodecBackend_t> nvimgcds_backends(backends.has_value() ? backends.value().size() : 0);
    if (backends.has_value()) {
        for (size_t i = 0; i < backends.value().size(); ++i) {
            nvimgcds_backends[i] = backends.value()[i].backend_;
        }
    }

    auto backends_ptr = nvimgcds_backends.size() ? nvimgcds_backends.data() : nullptr;
    nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
    exec_params.device_id = device_id;
    exec_params.max_num_cpu_threads = max_num_cpu_threads;
    exec_params.num_backends = nvimgcds_backends.size();
    exec_params.backends = backends_ptr;
    nvimgcodecDecoderCreate(instance, &decoder, &exec_params, options.c_str());

    decoder_ = std::shared_ptr<std::remove_pointer<nvimgcodecDecoder_t>::type>(
        decoder, [](nvimgcodecDecoder_t decoder) { nvimgcodecDecoderDestroy(decoder); });
}

Decoder::Decoder(nvimgcodecInstance_t instance, ILogger* logger, int device_id, int max_num_cpu_threads,
    std::optional<std::vector<nvimgcodecBackendKind_t>> backend_kinds, const std::string& options)
    : decoder_(nullptr)
    , instance_(instance)
    , logger_(logger)
{
    nvimgcodecDecoder_t decoder;
    std::vector<nvimgcodecBackend_t> nvimgcds_backends(backend_kinds.has_value() ? backend_kinds.value().size() : 0);
    if (backend_kinds.has_value()) {
        for (size_t i = 0; i < backend_kinds.value().size(); ++i) {
            nvimgcds_backends[i].kind = backend_kinds.value()[i];
            nvimgcds_backends[i].params = {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t), nullptr, 1.0f};
        }
    }

    auto backends_ptr = nvimgcds_backends.size() ? nvimgcds_backends.data() : nullptr;
    nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
    exec_params.device_id = device_id;
    exec_params.max_num_cpu_threads = max_num_cpu_threads;
    exec_params.num_backends = nvimgcds_backends.size();
    exec_params.backends = backends_ptr;
    nvimgcodecDecoderCreate(instance, &decoder, &exec_params, options.c_str());

    decoder_ = std::shared_ptr<std::remove_pointer<nvimgcodecDecoder_t>::type>(
        decoder, [](nvimgcodecDecoder_t decoder) { nvimgcodecDecoderDestroy(decoder); });
}

Decoder::~Decoder()
{
}

py::object Decoder::decode(const std::string& file_name, std::optional<DecodeParams> params, intptr_t cuda_stream)
{
    std::vector<nvimgcodecCodeStream_t> code_streams(1);
    CHECK_NVIMGCODEC(nvimgcodecCodeStreamCreateFromFile(instance_, &code_streams[0], file_name.c_str()));
    std::vector<py::object> images = decode(code_streams, params, cuda_stream);
    return images.size() == 1 ? images[0] : py ::none();
}

py::object Decoder::decode(py::bytes data, std::optional<DecodeParams> params, intptr_t cuda_stream)
{
    std::vector<nvimgcodecCodeStream_t> code_streams(1);
    auto str_view = static_cast<std::string_view>(data);
    CHECK_NVIMGCODEC(nvimgcodecCodeStreamCreateFromHostMem(instance_, &code_streams[0], (unsigned char*)str_view.data(), str_view.size()));
    std::vector<py::object> images = decode(code_streams, params, cuda_stream);
    return images.size() == 1 ? images[0] : py ::none();
}

py::object Decoder::decode(py::array_t<uint8_t> data, std::optional<DecodeParams> params, intptr_t cuda_stream)
{
    std::vector<nvimgcodecCodeStream_t> code_streams(1);
    auto img_data = data.unchecked<1>();
    CHECK_NVIMGCODEC(nvimgcodecCodeStreamCreateFromHostMem(instance_, &code_streams[0], img_data.data(0), img_data.size()));
    std::vector<py::object> images = decode(code_streams, params, cuda_stream);
    return images.size() == 1 ? images[0] : py::none();
}

std::vector<py::object> Decoder::decode(
    const std::vector<std::string>& file_names, std::optional<DecodeParams> params, intptr_t cuda_stream)
{
    std::vector<nvimgcodecCodeStream_t> code_streams(file_names.size());
    for (uint32_t i = 0; i < file_names.size(); i++) {
        CHECK_NVIMGCODEC(nvimgcodecCodeStreamCreateFromFile(instance_, &code_streams[i], file_names[i].c_str()));
    }
    return decode(code_streams, params, cuda_stream);
}

std::vector<py::object> Decoder::decode(const std::vector<py::bytes>& data_list, std::optional<DecodeParams> params, intptr_t cuda_stream)
{
    std::vector<nvimgcodecCodeStream_t> code_streams(data_list.size());
    for (uint32_t i = 0; i < data_list.size(); i++) {
        auto str_view = static_cast<std::string_view>(data_list[i]);

        CHECK_NVIMGCODEC(
            nvimgcodecCodeStreamCreateFromHostMem(instance_, &code_streams[i], (unsigned char*)str_view.data(), str_view.size()));
    }
    return decode(code_streams, params, cuda_stream);
}

std::vector<py::object> Decoder::decode(
    const std::vector<py::array_t<uint8_t>>& data_list, std::optional<DecodeParams> params, intptr_t cuda_stream)
{
    std::vector<nvimgcodecCodeStream_t> code_streams(data_list.size());
    for (uint32_t i = 0; i < data_list.size(); i++) {
        auto img_data = data_list[i].unchecked<1>();
        CHECK_NVIMGCODEC(nvimgcodecCodeStreamCreateFromHostMem(instance_, &code_streams[i], img_data.data(0), img_data.size()));
    }
    return decode(code_streams, params, cuda_stream);
}

std::vector<py::object> Decoder::decode(
    std::vector<nvimgcodecCodeStream_t>& code_streams, std::optional<DecodeParams> params_opt, intptr_t cuda_stream)
{
    std::vector<nvimgcodecImage_t> images(code_streams.size());
    std::vector<py::object> py_images;
    py_images.reserve(code_streams.size());
    DecodeParams params = params_opt.has_value() ? params_opt.value() : DecodeParams();

    size_t skip_samples = 0;
    for (uint32_t i = 0; i < code_streams.size(); i++) {
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        CHECK_NVIMGCODEC(nvimgcodecCodeStreamGetImageInfo(code_streams[i], &image_info));

        if (image_info.num_planes > NVIMGCODEC_MAX_NUM_PLANES) {
            NVIMGCODEC_LOG_WARNING(logger_, "Num Components > " << NVIMGCODEC_MAX_NUM_PLANES << "not supported.  It will not be included in output");

            skip_samples++;
            continue;
        }

        auto sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        int precision = 0;  // full dynamic range of the type
        if (params.allow_any_depth_) {
            sample_type = image_info.plane_info[0].sample_type;
            precision = image_info.plane_info[0].precision;
        }
        int bytes_per_element = sample_type_to_bytes_per_element(sample_type);

        image_info.cuda_stream = reinterpret_cast<cudaStream_t>(cuda_stream);

        //Decode to format
        bool decode_to_interleaved = true;
        image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;

        if (params.color_spec_ == NVIMGCODEC_COLORSPEC_SRGB) {
            image_info.sample_format = decode_to_interleaved ? NVIMGCODEC_SAMPLEFORMAT_I_RGB : NVIMGCODEC_SAMPLEFORMAT_P_RGB;
            image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
            image_info.plane_info[0].num_channels = decode_to_interleaved ? 3 /*I_RGB*/ : 1 /*P_RGB*/;
            image_info.num_planes = decode_to_interleaved ? 1 : image_info.num_planes;
        } else if (params.color_spec_ == NVIMGCODEC_COLORSPEC_GRAY) {
            image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_Y;
            image_info.color_spec = NVIMGCODEC_COLORSPEC_GRAY;
            image_info.plane_info[0].num_channels = 1;
            image_info.num_planes = 1;
        } else if (params.color_spec_ == NVIMGCODEC_COLORSPEC_UNCHANGED) {
            image_info.sample_format = decode_to_interleaved ? NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED : NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED;
            image_info.color_spec = NVIMGCODEC_COLORSPEC_UNCHANGED;
            uint32_t num_channels = std::max(image_info.num_planes, image_info.plane_info[0].num_channels);
            image_info.plane_info[0].num_channels = decode_to_interleaved ? num_channels : 1;
            image_info.num_planes = decode_to_interleaved ? 1 : num_channels;
        } else {
            // TODO(janton): support more?
        }

        bool swap_wh = params.decode_params_.apply_exif_orientation && ((image_info.orientation.rotated / 90) % 2);
        if (swap_wh) {
            std::swap(image_info.plane_info[0].height, image_info.plane_info[0].width);
        }

        size_t device_pitch_in_bytes = image_info.plane_info[0].width * bytes_per_element * image_info.plane_info[0].num_channels;

        size_t buffer_size = 0;
        for (uint32_t c = 0; c < image_info.num_planes; ++c) {
            image_info.plane_info[c].height = image_info.plane_info[0].height;
            image_info.plane_info[c].width = image_info.plane_info[0].width;
            image_info.plane_info[c].row_stride = device_pitch_in_bytes;
            image_info.plane_info[c].sample_type = sample_type;
            image_info.plane_info[c].precision = precision;
            image_info.plane_info[c].num_channels = image_info.plane_info[0].num_channels;
            buffer_size += image_info.plane_info[c].row_stride * image_info.plane_info[c].height;
        }
        image_info.buffer = nullptr;
        image_info.buffer_size = buffer_size;
        image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;

        Image img(instance_, &image_info);
        images[i - skip_samples] = img.getNvImgCdcsImage();
        py_images.push_back(py::cast(std::move(img)));
    }
    nvimgcodecFuture_t decode_future;
    CHECK_NVIMGCODEC(nvimgcodecDecoderDecode(
        decoder_.get(), code_streams.data(), images.data(), code_streams.size(), &params.decode_params_, &decode_future));
    nvimgcodecFutureWaitForAll(decode_future);
    size_t status_size;
    nvimgcodecFutureGetProcessingStatus(decode_future, nullptr, &status_size);
    std::vector<nvimgcodecProcessingStatus_t> decode_status(status_size);
    nvimgcodecFutureGetProcessingStatus(decode_future, &decode_status[0], &status_size);
    skip_samples = 0;
    for (size_t i = 0; i < decode_status.size(); ++i) {
        if (decode_status[i] != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
            NVIMGCODEC_LOG_WARNING(logger_, "Something went wrong during decoding image #" << i << " it will not be included in output");
            py_images.erase(py_images.begin() + i - skip_samples);
            skip_samples++;
        }
    }
    nvimgcodecFutureDestroy(decode_future);
    for (auto& cs : code_streams) {
        nvimgcodecCodeStreamDestroy(cs);
    }

    return py_images;
}

py::object Decoder::enter()
{
    return py::cast(*this);
}

void Decoder::exit(const std::optional<pybind11::type>& exc_type, const std::optional<pybind11::object>& exc_value,
    const std::optional<pybind11::object>& traceback)
{
    decoder_.reset();
}

void Decoder::exportToPython(py::module& m, nvimgcodecInstance_t instance, ILogger* logger)
{
    py::class_<Decoder>(m, "Decoder")
        .def(py::init<>([instance, logger](int device_id, int max_num_cpu_threads, std::optional<std::vector<Backend>> backends,
                            const std::string& options) {
            return new Decoder(instance, logger, device_id, max_num_cpu_threads, backends, options);
        }),
            R"pbdoc(
            Initialize decoder.

            Args:
                device_id: Device id to execute decoding on.

                max_num_cpu_threads: Max number of CPU threads in default executor (0 means default value equal to number of cpu cores) 

                backends: List of allowed backends. If empty, all backends are allowed with default parameters.
                
                options: Decoder specific options e.g.: "nvjpeg:fancy_upsampling=1"

            )pbdoc",
            "device_id"_a = NVIMGCODEC_DEVICE_CURRENT, "max_num_cpu_threads"_a = 0, "backends"_a = py::none(),
            "options"_a = ":fancy_upsampling=0")
        .def(py::init<>([instance, logger](int device_id, int max_num_cpu_threads,
                            std::optional<std::vector<nvimgcodecBackendKind_t>> backend_kinds, const std::string& options) {
            return new Decoder(instance, logger, device_id, max_num_cpu_threads, backend_kinds, options);
        }),
            R"pbdoc(
            Initialize decoder.

            Args:
                device_id: Device id to execute decoding on.

                max_num_cpu_threads: Max number of CPU threads in default executor (0 means default value equal to number of cpu cores)

                backend_kinds: List of allowed backend kinds. If empty or None, all backends are allowed with default parameters.

                options: Decoder specific options e.g.: "nvjpeg:fancy_upsampling=1"

            )pbdoc",
            "device_id"_a = NVIMGCODEC_DEVICE_CURRENT, "max_num_cpu_threads"_a = 0, "backend_kinds"_a = py::none(),
            "options"_a = ":fancy_upsampling=0")
        .def("decode", py::overload_cast<py::bytes, std::optional<DecodeParams>, intptr_t>(&Decoder::decode),
            R"pbdoc(
            Executes decoding of data.

            Args:
                data: Buffer with bytes to decode.

                params: Decode parameters.

                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

            Returns:
                nvimgcodec.Image or None if the image cannot be decoded because of any reason. 

            )pbdoc",
            "data"_a, "params"_a = py::none(), "cuda_stream"_a = 0)

        .def("decode", py::overload_cast<py::array_t<uint8_t>, std::optional<DecodeParams>, intptr_t>(&Decoder::decode),
            R"pbdoc(
            Executes decoding of NumPy array.

            Args:
                data: Numpy array with bytes to decode.
           
                params: Decode parameters.
           
                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.
           
            Returns:
                nvimgcodec.Image or None if the image cannot be decoded because of any reason.

            )pbdoc",
            "data"_a, "params"_a = py::none(), "cuda_stream"_a = 0)

        .def("read", py::overload_cast<const std::string&, std::optional<DecodeParams>, intptr_t>(&Decoder::decode), R"pbdoc(
            Executes decoding of file.

            Args:
                file_name: File name to decode.

                params: Decode parameters.
                
                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.
            
            Returns:
                nvimgcodec.Image or None if the image cannot be decoded because of any reason.
        )pbdoc",
            "file_name"_a, "params"_a = py::none(), "cuda_stream"_a = 0)

        .def("decode", py::overload_cast<const std::vector<py::array_t<uint8_t>>&, std::optional<DecodeParams>, intptr_t>(&Decoder::decode),
            R"pbdoc(
            Executes NumPy array batch decoding.

            Args:
                data: List of NumPy arrays with bytes to decode.
                
                params: Decode parameters.
                
                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

            Returns:
                List of decoded nvimgcodec.Image's
            )pbdoc",
            "data"_a, "params"_a = py::none(), "cuda_stream"_a = 0)

        .def("decode", py::overload_cast<const std::vector<py::bytes>&, std::optional<DecodeParams>, intptr_t>(&Decoder::decode),
            R"pbdoc(
            Executes data batch decoding.

            Args:
                file_names: List of buffers with code streams to decode.

                params: Decode parameters.
                
                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.
            
            Returns:
                List of decoded nvimgcodec.Image's

            )pbdoc",
            "file_names"_a, "params"_a = py::none(), "cuda_stream"_a = 0)
        .def("read", py::overload_cast<const std::vector<std::string>&, std::optional<DecodeParams>, intptr_t>(&Decoder::decode),
            R"pbdoc(
            Executes file batch decoding.

            Args:
                data_list: List of file names to decode.

                params: Decode parameters.

                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

            Returns:
                List of decoded nvimgcodec.Image's

            )pbdoc",
            "data_list"_a, "params"_a = py::none(), "cuda_stream"_a = 0)
        .def("__enter__", &Decoder::enter, "Enter the runtime context related to this decoder.")
        .def("__exit__", &Decoder::exit, "Exit the runtime context related to this decoder and releases allocated resources.",
            "exc_type"_a = py::none(), "exc_value"_a = py::none(), "traceback"_a = py::none());
}

} // namespace nvimgcodec
