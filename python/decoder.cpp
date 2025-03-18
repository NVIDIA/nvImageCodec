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

    is_cpu_only_ = nvimgcds_backends.size() > 0;
    for (size_t i = 0; is_cpu_only_ && i < nvimgcds_backends.size(); i++) {
        if (nvimgcds_backends[i].kind != NVIMGCODEC_BACKEND_KIND_CPU_ONLY)
            is_cpu_only_ = false;
    }

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
            nvimgcds_backends[i].params = {
                NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t), nullptr, 1.0f, NVIMGCODEC_LOAD_HINT_POLICY_FIXED};
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

py::object Decoder::decode(const DecodeSource* data, std::optional<DecodeParams> params, intptr_t cuda_stream)
{
    assert(data);
    std::vector<py::object> images =
        decode_impl(std::vector<nvimgcodecCodeStream_t>{data->code_stream()->handle()}, std::vector<std::optional<Region>>{data->region()}, params, cuda_stream);
    return images.size() == 1 ? images[0] : py::none();
}

std::vector<py::object> Decoder::decode(
    const std::vector<const DecodeSource*>& decode_source_arg, 
    std::optional<DecodeParams> params_opt,
    intptr_t cuda_stream)
{
    std::vector<nvimgcodecCodeStream_t> code_streams;
    std::vector<std::optional<Region>> rois;
    code_streams.reserve(decode_source_arg.size());
    rois.reserve(decode_source_arg.size());
    for (auto& ds : decode_source_arg) {
        assert(ds);
        code_streams.push_back(ds->code_stream()->handle());
        rois.push_back(ds->region());
    }
    return decode_impl(code_streams, rois, params_opt, cuda_stream);
}

std::vector<py::object> Decoder::decode_impl(
    const std::vector<nvimgcodecCodeStream_t>& code_streams_arg,
    std::vector<std::optional<Region>> rois,
    std::optional<DecodeParams> params_opt,
    intptr_t cuda_stream)
{
    size_t orig_nsamples = code_streams_arg.size();
    assert(rois.size() == orig_nsamples);
    std::vector<nvimgcodecCodeStream_t> code_streams;
    code_streams.reserve(orig_nsamples);
    std::vector<nvimgcodecImage_t> images;
    images.reserve(orig_nsamples);
    std::vector<size_t> orig_sample_idx;
    orig_sample_idx.reserve(orig_nsamples);
    std::vector<py::object> py_images;
    py_images.reserve(orig_nsamples);


    DecodeParams params = params_opt.has_value() ? params_opt.value() : DecodeParams();
    auto has_any_roi_set = [](const std::vector<std::optional<Region>>& rois) {
        for (auto& roi : rois)
            if (roi)
                return true;
        return false;
    };
    params.decode_params_.enable_roi = has_any_roi_set(rois);

    for (size_t i = 0; i < orig_nsamples; i++) {
        const auto& code_stream = code_streams_arg[i];
        const auto& roi = rois[i];
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        nvimgcodecStatus_t ret_getimginfo(NVIMGCODEC_STATUS_NOT_INITIALIZED);
        {
            py::gil_scoped_release release;
            ret_getimginfo = nvimgcodecCodeStreamGetImageInfo(code_stream, &image_info);
        }
        if (ret_getimginfo != NVIMGCODEC_STATUS_SUCCESS) {
            // not logging here again, the specific error should be logged by the function
            
            py_images.push_back(py::none());
            continue;
        }

        if (image_info.num_planes > NVIMGCODEC_MAX_NUM_PLANES) {
            NVIMGCODEC_LOG_WARNING(logger_, "Number of components exceeds the maximum value allowed by the library: "
                                                << image_info.num_planes << " > " << NVIMGCODEC_MAX_NUM_PLANES
                                                << ". If your application requires more components, please report it to "
                                                   "https://github.com/NVIDIA/nvImageCodec/issues.");
            py_images.push_back(py::none());                                                   
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

        int decode_out_height = image_info.plane_info[0].height;
        int decode_out_width = image_info.plane_info[0].width;
        if (roi) {
            image_info.region = roi.value();
            decode_out_height = image_info.region.end[0] - image_info.region.start[0];
            decode_out_width = image_info.region.end[1] - image_info.region.start[1];
        }
        bool swap_wh = params.decode_params_.apply_exif_orientation && ((image_info.orientation.rotated / 90) % 2);
        if (swap_wh) {
            std::swap(decode_out_height, decode_out_width);
        }

        size_t device_pitch_in_bytes = decode_out_width * bytes_per_element * image_info.plane_info[0].num_channels;

        int64_t buffer_size = 0;
        for (uint32_t c = 0; c < image_info.num_planes; ++c) {
            image_info.plane_info[c].height = decode_out_height;
            image_info.plane_info[c].width = decode_out_width;
            image_info.plane_info[c].row_stride = device_pitch_in_bytes;
            image_info.plane_info[c].sample_type = sample_type;
            image_info.plane_info[c].precision = precision;
            image_info.plane_info[c].num_channels = image_info.plane_info[0].num_channels;
            buffer_size += image_info.plane_info[c].row_stride * image_info.plane_info[c].height;
        }
        image_info.buffer = nullptr;
        image_info.buffer_size = buffer_size;
        image_info.buffer_kind = is_cpu_only_ ? NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST : NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;

        static const char* max_image_size_str = std::getenv("NVIMGCODEC_MAX_IMAGE_SIZE");
        static const int64_t max_image_sz = max_image_size_str && atol(max_image_size_str);
        if (max_image_sz > 0 && buffer_size > max_image_sz) {
            NVIMGCODEC_LOG_WARNING(
                logger_, "Total image volume (height x width x channels x bytes_per_sample) exceeds the maximum configured value: "
                             << buffer_size << " > NVIMGCODEC_MAX_IMAGE_SIZE(" << max_image_sz
                             << "). Use NVIMGCODEC_MAX_IMAGE_SIZE env variable to control this maximum value.");
            continue;
        }

        code_streams.push_back(code_stream);

        Image img(instance_, &image_info);
        images.push_back(img.getNvImgCdcsImage());
        orig_sample_idx.push_back(i);
        py_images.push_back(py::cast(std::move(img)));
    }

    if (images.empty()) {
        return py_images;
    }

    std::vector<nvimgcodecProcessingStatus_t> decode_status;
    {
        py::gil_scoped_release release;
        nvimgcodecFuture_t decode_future;
        CHECK_NVIMGCODEC(nvimgcodecDecoderDecode(
            decoder_.get(), code_streams.data(), images.data(), code_streams.size(), &params.decode_params_, &decode_future));
        nvimgcodecFutureWaitForAll(decode_future);
        size_t status_size;
        decode_status.resize(code_streams.size());
        nvimgcodecFutureGetProcessingStatus(decode_future, &decode_status[0], &status_size);
        assert(status_size == code_streams.size());
        nvimgcodecFutureDestroy(decode_future);
    }

    for (size_t i = 0; i < decode_status.size(); ++i) {
        if (decode_status[i] != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
            NVIMGCODEC_LOG_WARNING(logger_, "Something went wrong during decoding image #" << i << " there will be None on corresponding output position");
            py_images[orig_sample_idx[i]] = py::none();
        }
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
    // clang-format off
    py::class_<Decoder>(m, "Decoder", "Decoder for image decoding operations. "
        "It provides methods to decode images from various sources such as files or data streams. "
        "The decoding process can be configured with parameters like the applied backend or execution settings.")
        .def(py::init<>(
            [instance, logger](int device_id, int max_num_cpu_threads, std::optional<std::vector<Backend>> backends,
                               const std::string& options) {
                return new Decoder(instance, logger, device_id, max_num_cpu_threads, backends, options);
            }),
            R"pbdoc(
            Initialize decoder.

            Args:
                device_id: Device id to execute decoding on.

                max_num_cpu_threads: Max number of CPU threads in default executor (0 means default value equal to number of cpu cores).

                backends: List of allowed backends. If empty, all backends are allowed with default parameters.
                
                options: Decoder specific options e.g.: "nvjpeg:fancy_upsampling=1"

            )pbdoc",
            "device_id"_a = NVIMGCODEC_DEVICE_CURRENT, "max_num_cpu_threads"_a = 0, "backends"_a = py::none(),
            "options"_a = ":fancy_upsampling=0")
        .def(py::init<>(
            [instance, logger](int device_id, int max_num_cpu_threads, std::optional<std::vector<nvimgcodecBackendKind_t>> backend_kinds,
                               const std::string& options) {
                return new Decoder(instance, logger, device_id, max_num_cpu_threads, backend_kinds, options);
            }),
            R"pbdoc(
            Initialize decoder.

            Args:
                device_id: Device id to execute decoding on.

                max_num_cpu_threads: Max number of CPU threads in default executor (0 means default value equal to number of cpu cores).

                backend_kinds: List of allowed backend kinds. If empty or None, all backends are allowed with default parameters.

                options: Decoder specific options e.g.: "nvjpeg:fancy_upsampling=1"

            )pbdoc",
            "device_id"_a = NVIMGCODEC_DEVICE_CURRENT, "max_num_cpu_threads"_a = 0, "backend_kinds"_a = py::none(),
            "options"_a = ":fancy_upsampling=0")
        .def("read", py::overload_cast<const DecodeSource*, std::optional<DecodeParams>, intptr_t>(&Decoder::decode), R"pbdoc(
            Executes decoding from a filename.

            Args:
                path: File path to decode.

                params: Decode parameters.

                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

            Returns:
                nvimgcodec.Image or None if the image cannot be decoded because of any reason.
        )pbdoc",
            "path"_a, "params"_a = py::none(), "cuda_stream"_a = 0)
        .def("read", py::overload_cast<const std::vector<const DecodeSource*>&, std::optional<DecodeParams>, intptr_t>(&Decoder::decode),
            R"pbdoc(
            Executes decoding from a batch of file paths.

            Args:
                path: List of file paths to decode.

                params: Decode parameters.

                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

            Returns:
                List of decoded nvimgcodec.Image's. There is None in returned list on positions which could not be decoded.

            )pbdoc",
            "paths"_a, "params"_a = py::none(), "cuda_stream"_a = 0)
        .def("decode", py::overload_cast<const DecodeSource*, std::optional<DecodeParams>, intptr_t>(&Decoder::decode), R"pbdoc(
            Executes decoding of data from a DecodeSource handle (code stream handle and an optional region of interest).

            Args:
                src: decode source object.

                params: Decode parameters.

                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

            Returns:
                nvimgcodec.Image or None if the image cannot be decoded because of any reason.

            )pbdoc",
            "src"_a, "params"_a = py::none(), "cuda_stream"_a = 0)
        .def("decode", py::overload_cast<const std::vector<const DecodeSource*>&, std::optional<DecodeParams>, intptr_t>(&Decoder::decode),
            R"pbdoc(
            Executes decoding from a batch of DecodeSource handles (code stream handle and an optional region of interest).

            Args:
                srcs: List of DecodeSource objects

                params: Decode parameters.

                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

            Returns:
                List of decoded nvimgcodec.Image's. There is None in returned list on positions which could not be decoded.
            )pbdoc",
            "srcs"_a, "params"_a = py::none(), "cuda_stream"_a = 0)
        .def("__enter__", &Decoder::enter, "Enter the runtime context related to this decoder.")
        .def("__exit__", &Decoder::exit, "Exit the runtime context related to this decoder and releases allocated resources.",
            "exc_type"_a = py::none(), "exc_value"_a = py::none(), "traceback"_a = py::none());
    // clang-format on
}

} // namespace nvimgcodec
