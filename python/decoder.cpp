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

#include "decoder.h"

#include <iostream>
#include <string_view>

#include <ilogger.h>
#include <log.h>

#include "backend.h"
#include "error_handling.h"
#include "type_utils.h"
#include "region.h"
#include "metadata.h"
#include "metadata_kind.h"

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
            nvimgcds_backends[i] = {NVIMGCODEC_STRUCTURE_TYPE_BACKEND, sizeof(nvimgcodecBackend_t), nullptr};
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

py::object Decoder::decode(const CodeStream* code_stream, std::optional<DecodeParams> params, intptr_t cuda_stream)
{
    if(!code_stream) { // if none was passed
        py::gil_scoped_acquire acquire;
        return py::none();
    }

    assert(code_stream);
    std::vector<py::object> images = decode_impl(std::vector<const CodeStream*>{code_stream}, params, cuda_stream);
    return images.size() == 1 ? images[0] : py::none();
}

std::vector<py::object> Decoder::decode(
    const std::vector<const CodeStream*>& code_streams, 
    std::optional<DecodeParams> params_opt,
    intptr_t cuda_stream)
{
    return decode_impl(code_streams, params_opt, cuda_stream);
}

std::vector<py::object> Decoder::decode_impl(
    const std::vector<const CodeStream*>& code_streams_arg,
    std::optional<DecodeParams> params_opt,
    intptr_t cuda_stream)
{
    size_t orig_nsamples = code_streams_arg.size();
    std::vector<nvimgcodecCodeStream_t> code_streams;
    code_streams.reserve(orig_nsamples);
    std::vector<nvimgcodecImage_t> images;
    images.reserve(orig_nsamples);
    std::vector<size_t> orig_sample_idx;
    orig_sample_idx.reserve(orig_nsamples);
    std::vector<py::object> py_images;
    py_images.reserve(orig_nsamples);

    DecodeParams params = params_opt.has_value() ? params_opt.value() : DecodeParams();

    for (size_t i = 0; i < orig_nsamples; i++) {
        const auto& code_stream = code_streams_arg[i];
        if (!code_stream) {
            NVIMGCODEC_LOG_WARNING(
                logger_,
                "None was passed for input bitstream #" << i << " there will be None on corresponding output position.\n"
            );
            py::gil_scoped_acquire acquire;
            py_images.push_back(py::none());
            continue;
        }

        try {
            const auto& view = code_stream->view();
            const auto& roi = view ? view.value().region() : std::nullopt;
            auto image_info = code_stream->getImageInfo();
            if (image_info.num_planes > NVIMGCODEC_MAX_NUM_PLANES) {
                NVIMGCODEC_LOG_WARNING(logger_, "Number of components for input bitstream #" << i << "exceeds the maximum value allowed by the library: "
                                                    << image_info.num_planes << " > " << NVIMGCODEC_MAX_NUM_PLANES
                                                    << " there will be None on corresponding output position."
                                                    << " If your application requires more components, please report it to "
                                                       "https://github.com/NVIDIA/nvImageCodec/issues.");
                py::gil_scoped_acquire acquire;
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
                nvimgcodecRegion_t region = static_cast<nvimgcodecRegion_t>(roi.value());
                decode_out_height = region.end[0] - region.start[0];
                decode_out_width = region.end[1] - region.start[1];
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
                py::gil_scoped_acquire acquire;
                py_images.push_back(py::none());
                continue;
            }
            try {
                py::gil_scoped_acquire acquire;
                Image img(instance_, logger_, &image_info);
                images.push_back(img.getNvImgCdcsImage());
                code_streams.push_back(code_stream->handle());
                py_images.push_back(py::cast(std::move(img)));
                orig_sample_idx.push_back(i);
            } catch (...) {
                NVIMGCODEC_LOG_WARNING(logger_, "Something went wrong during decoding image #" << i << " there will be None on corresponding output position");
                py::gil_scoped_acquire acquire;
                py_images.push_back(py::none());
                continue;
            }
        } catch (...) {
            NVIMGCODEC_LOG_WARNING(logger_, "Could not parse input bitstream #" << i << " there will be None on corresponding output position");
            py::gil_scoped_acquire acquire;
            py_images.push_back(py::none());
            continue;
        }
    }

    if (images.empty()) {
        return py_images;
    }

    try
    {
        py::gil_scoped_release release;
        std::vector<nvimgcodecProcessingStatus_t> decode_status;
        nvimgcodecFuture_t decode_future;
        CHECK_NVIMGCODEC(nvimgcodecDecoderDecode(
            decoder_.get(), code_streams.data(), images.data(), code_streams.size(), &params.decode_params_, &decode_future));
        CHECK_NVIMGCODEC(nvimgcodecFutureWaitForAll(decode_future));
        size_t status_size;
        decode_status.resize(code_streams.size());
        CHECK_NVIMGCODEC(nvimgcodecFutureGetProcessingStatus(decode_future, &decode_status[0], &status_size));
        assert(status_size == code_streams.size());
        CHECK_NVIMGCODEC(nvimgcodecFutureDestroy(decode_future));

        for (size_t i = 0; i < decode_status.size(); ++i) {
            if (decode_status[i] != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                NVIMGCODEC_LOG_WARNING(logger_, "Something went wrong during decoding image #" << i << " there will be None on corresponding output position");
                py::gil_scoped_acquire acquire;
                py_images[orig_sample_idx[i]] = py::none();
            }
        }
    } catch (...) {
        NVIMGCODEC_LOG_WARNING(logger_, "Something went wrong during decoding and there will be None on corresponding output positions");
        py::gil_scoped_acquire acquire;
        py_images.assign(orig_nsamples, py::none());

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

py::list Decoder::getMetadata(const CodeStream& code_stream, std::optional<nvimgcodecMetadataKind_t> kind)
{
    int metadata_count = 0;
    //First call to get metadata count
    CHECK_NVIMGCODEC(nvimgcodecDecoderGetMetadata(decoder_.get(), code_stream.handle(), nullptr, &metadata_count));

    if (metadata_count == 0) {
        return py::list();
    }

    py::list metadata;
    std::vector<nvimgcodecMetadata_t*> metadata_ptrs(metadata_count);
    
    //Create initial list of metadata objects
    for (int i = 0; i < metadata_count; i++) {
        metadata.append(Metadata());
        metadata_ptrs[i] = py::cast<Metadata&>(metadata[i]).handle();
    }

    //Second call to get buffer sizes
    CHECK_NVIMGCODEC(nvimgcodecDecoderGetMetadata(decoder_.get(), code_stream.handle(), metadata_ptrs.data(), &metadata_count));

    //Filter metadata based on kind if specified
    if (kind) {
        int filtered_count = 0;
        for (int i = 0; i < metadata_count; i++) {
            if (py::cast<Metadata&>(metadata[i]).kind() == kind.value()) {
                if (i != filtered_count) {
                    metadata[filtered_count] = std::move(metadata[i]);
                    metadata_ptrs[filtered_count] = py::cast<Metadata&>(metadata[filtered_count]).handle();
                }
                filtered_count++;
            }
        }
        metadata_count = filtered_count;
        while (py::len(metadata) > (size_t)metadata_count) {
            metadata.attr("pop")();
        }
        metadata_ptrs.resize(metadata_count);
    }

    if (metadata_count == 0) {
        return py::list();
    }

    //Allocate buffers for filtered metadata
    for (int i = 0; i < metadata_count; i++) {
        py::cast<Metadata&>(metadata[i]).allocateBuffer();
    }

    //Third call to get actual metadata
    CHECK_NVIMGCODEC(nvimgcodecDecoderGetMetadata(decoder_.get(), code_stream.handle(), metadata_ptrs.data(), &metadata_count));

    return metadata;
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
        .def("read", py::overload_cast<const CodeStream*, std::optional<DecodeParams>, intptr_t>(&Decoder::decode), R"pbdoc(
            Executes decoding from a filename.

            Args:
                path: File path to decode.

                params: Decode parameters.

                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

            Returns:
                nvimgcodec.Image or None if the image cannot be decoded because of any reason.
        )pbdoc",
            "path"_a, "params"_a = py::none(), "cuda_stream"_a = 0)
        .def("read", py::overload_cast<const std::vector<const CodeStream*>&, std::optional<DecodeParams>, intptr_t>(&Decoder::decode),
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
        .def("decode", py::overload_cast<const CodeStream*, std::optional<DecodeParams>, intptr_t>(&Decoder::decode), R"pbdoc(
            Executes decoding of data from a CodeStream handle (code stream handle and an optional region of interest).

            Args:
                src: decode source object.

                params: Decode parameters.

                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

            Returns:
                nvimgcodec.Image or None if the image cannot be decoded because of any reason.

            )pbdoc",
            "src"_a, "params"_a = py::none(), "cuda_stream"_a = 0)
        .def("decode", py::overload_cast<const std::vector<const CodeStream*>&, std::optional<DecodeParams>, intptr_t>(&Decoder::decode),
            R"pbdoc(
            Executes decoding from a batch of CodeStream handles (code stream handle and an optional region of interest).

            Args:
                srcs: List of CodeStream objects

                params: Decode parameters.

                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

            Returns:
                List of decoded nvimgcodec.Image's. There is None in returned list on positions which could not be decoded.
            )pbdoc",
            "srcs"_a, "params"_a = py::none(), "cuda_stream"_a = 0)
        .def("getMetadata", &Decoder::getMetadata, R"pbdoc(
            Retrieves metadata from a code stream.

            Args:
                code_stream: The code stream to get metadata from.
                kind: Optional metadata kind to filter by. If specified, only metadata of that kind will be returned.

            Returns:
                A list of Metadata objects. If no metadata exists or no metadata matches the specified kind,
                returns an empty list.
            )pbdoc",
            "code_stream"_a, "kind"_a = py::none())
        .def("__enter__", &Decoder::enter, "Enter the runtime context related to this decoder.")
        .def("__exit__", &Decoder::exit, "Exit the runtime context related to this decoder and releases allocated resources.",
            "exc_type"_a = py::none(), "exc_value"_a = py::none(), "traceback"_a = py::none());
    // clang-format on
}

} // namespace nvimgcodec
