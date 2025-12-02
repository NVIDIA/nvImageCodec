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
#include "imgproc/exception.h"
#include "imgproc/type_utils.h"
#include "nvimgcodec.h"
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
    nvimgcodecDecoder_t decoder = nullptr;
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

    nvimgcodecStatus_t status = nvimgcodecDecoderCreate(instance, &decoder, &exec_params, options.c_str());
    if (status != NVIMGCODEC_STATUS_SUCCESS)
        throw Exception(ARCH_MISMATCH, "Could not create decoder. The requested backends are not supported.");

    decoder_ = std::shared_ptr<std::remove_pointer<nvimgcodecDecoder_t>::type>(
        decoder, [](nvimgcodecDecoder_t decoder) { nvimgcodecDecoderDestroy(decoder); });
}

Decoder::~Decoder()
{
}

py::object Decoder::decode(const CodeStream* code_stream, std::optional<Image*> image, std::optional<DecodeParams> params, intptr_t cuda_stream)
{
    if(!code_stream) { // if none was passed
        py::gil_scoped_acquire acquire;
        return py::none();
    }

    assert(code_stream);
    std::optional<std::vector<Image*>> images;
    if (image.has_value()) {
        images = std::vector<Image*>{image.value()};
    }
    std::vector<py::object> result = decode(std::vector<const CodeStream*>{code_stream}, images, params, cuda_stream);
    return result.size() == 1 ? result[0] : py::none();
}

std::vector<py::object> Decoder::decode(
    const std::vector<const CodeStream*>& code_streams_arg, 
    std::optional<std::vector<Image*>> images_arg,
    std::optional<DecodeParams> params_opt,
    intptr_t cuda_stream)
{
    size_t orig_nsamples = code_streams_arg.size();
    
    // Validate images list size if provided
    if (images_arg.has_value() && images_arg.value().size() != orig_nsamples) {
        throw std::invalid_argument("Size mismatch - images list has " + std::to_string(images_arg.value().size()) + 
                            " items, but code_streams list has " + std::to_string(orig_nsamples) + " items.");
    }
    
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
            bool decode_to_interleaved = true; //TODO introduce sample_forat param to decode  function and base on it on this code

    
            if (params.color_spec_ == NVIMGCODEC_COLORSPEC_SRGB) {
                image_info.sample_format = decode_to_interleaved ? NVIMGCODEC_SAMPLEFORMAT_I_RGB : NVIMGCODEC_SAMPLEFORMAT_P_RGB;
                image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
                image_info.plane_info[0].num_channels = decode_to_interleaved ? 3 /*I_RGB*/ : 1 /*P_RGB*/;
                image_info.num_planes = decode_to_interleaved ? 1 : 3;
                image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
            } else if (params.color_spec_ == NVIMGCODEC_COLORSPEC_GRAY) {
                image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_Y;
                image_info.color_spec = NVIMGCODEC_COLORSPEC_GRAY;
                image_info.plane_info[0].num_channels = 1;
                image_info.num_planes = 1;
                image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_GRAY;
            } else if (params.color_spec_ == NVIMGCODEC_COLORSPEC_UNCHANGED) {
                if (image_info.color_spec == NVIMGCODEC_COLORSPEC_GRAY || image_info.color_spec == NVIMGCODEC_COLORSPEC_SRGB) {
                   
                     //  This is temporary as there is not support for plannar output yet so always decode to interleaved
                     // TODO should be : image_info.sample_format = intentionally not changed as it is specified in decode params
                     if (decode_to_interleaved) {
                        switch (image_info.sample_format) {
                            case NVIMGCODEC_SAMPLEFORMAT_P_Y:
                            case NVIMGCODEC_SAMPLEFORMAT_I_Y:
                                image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_Y;
                                image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_GRAY;
                                break;
                            case NVIMGCODEC_SAMPLEFORMAT_I_YA:
                            case NVIMGCODEC_SAMPLEFORMAT_P_YA:
                                image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_YA;
                                image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_GRAY;
                                break;
                            case NVIMGCODEC_SAMPLEFORMAT_I_RGB:
                            case NVIMGCODEC_SAMPLEFORMAT_P_RGB:
                                image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
                                image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
                                break;
                            case NVIMGCODEC_SAMPLEFORMAT_I_BGR:
                            case NVIMGCODEC_SAMPLEFORMAT_P_BGR:
                                image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_BGR;
                                image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
                                break;
                            case NVIMGCODEC_SAMPLEFORMAT_I_YUV:
                            case NVIMGCODEC_SAMPLEFORMAT_P_YUV:
                                image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_YUV;
                                break;
                            case NVIMGCODEC_SAMPLEFORMAT_I_RGBA:
                            case NVIMGCODEC_SAMPLEFORMAT_P_RGBA:
                                image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGBA;
                                image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
                                break;
                            default:
                                image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED;
                                break;
                        }
                     } else {
                        image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED;
                        image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
                     }
                    
                     // image_info.color_spec intensionally is not changed as it is specified in decode params
                } else {
                    image_info.sample_format = decode_to_interleaved ? NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED : NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED;
                    //TODO Now there is limitation that other input color spaces are not handled correctly so it is not supported yet
                    // and we have to decode to sRGB
                    image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
                    image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
                }
                uint32_t num_channels = std::max(image_info.num_planes, image_info.plane_info[0].num_channels);
                image_info.plane_info[0].num_channels = decode_to_interleaved ? num_channels : 1;
                image_info.num_planes = decode_to_interleaved ? 1 : num_channels;
            } else if (params.color_spec_ == NVIMGCODEC_COLORSPEC_SYCC) {
                image_info.sample_format = decode_to_interleaved ? NVIMGCODEC_SAMPLEFORMAT_I_YCC : NVIMGCODEC_SAMPLEFORMAT_P_YCC;
                image_info.color_spec = NVIMGCODEC_COLORSPEC_SYCC;
                image_info.plane_info[0].num_channels = decode_to_interleaved ? 3 /*I_YUV*/ : 1 /*P_YUV*/;
                image_info.num_planes = decode_to_interleaved ? 1 : 3;
                
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

            for (uint32_t c = 0; c < image_info.num_planes; ++c) {
                image_info.plane_info[c].height = decode_out_height;
                image_info.plane_info[c].width = decode_out_width;
                image_info.plane_info[c].row_stride = device_pitch_in_bytes;
                image_info.plane_info[c].sample_type = sample_type;
                image_info.plane_info[c].precision = precision;
                image_info.plane_info[c].num_channels = image_info.plane_info[0].num_channels;
            }
            image_info.buffer = nullptr;
            image_info.buffer_kind = is_cpu_only_ ? NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST : NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
    
            static const char* max_image_size_str = std::getenv("NVIMGCODEC_MAX_IMAGE_SIZE");
            static const size_t max_image_sz = max_image_size_str && atol(max_image_size_str);
            if (max_image_sz > 0 && GetBufferSize(image_info) > max_image_sz) {
                NVIMGCODEC_LOG_WARNING(
                    logger_, "Total image volume (height x width x channels x bytes_per_sample) exceeds the maximum configured value: "
                                 << GetBufferSize(image_info) << " > NVIMGCODEC_MAX_IMAGE_SIZE(" << max_image_sz
                                 << "). Use NVIMGCODEC_MAX_IMAGE_SIZE env variable to control this maximum value.");
                py::gil_scoped_acquire acquire;
                py_images.push_back(py::none());
                continue;
            }
            try {
                py::gil_scoped_acquire acquire;
                if (images_arg.has_value() && i < images_arg.value().size() && images_arg.value()[i] != nullptr) {
                    // Reuse existing Image
                    Image* reuse_img = images_arg.value()[i];
                    reuse_img->reuse(&image_info);
                    images.push_back(reuse_img->getNvImgCdcsImage());
                    code_streams.push_back(code_stream->handle());
                    py_images.push_back(py::cast(reuse_img, py::return_value_policy::reference));
                    orig_sample_idx.push_back(i);
                } else {
                    // Create new Image
                    Image img(instance_, logger_, &image_info);
                    images.push_back(img.getNvImgCdcsImage());
                    code_streams.push_back(code_stream->handle());
                    py_images.push_back(py::cast(std::move(img)));
                    orig_sample_idx.push_back(i);
                }
            } catch (const std::invalid_argument& e) {
                py::gil_scoped_acquire acquire;
                throw e;
            } catch (const std::exception& e) {
                NVIMGCODEC_LOG_WARNING(logger_, "Something went wrong during decoding image #" << i << " (" << e.what() << "). There will be None on corresponding output position");
                py::gil_scoped_acquire acquire;
                py_images.push_back(py::none());
                continue;
            }
        } catch (const std::invalid_argument& e) {
            py::gil_scoped_acquire acquire;
            throw e;
        } catch (const std::exception& e) {
            NVIMGCODEC_LOG_WARNING(logger_, "Could not parse input bitstream #" << i << " (" << e.what() << "). There will be None on corresponding output position");
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
    py::list metadata;
    int metadata_count = 0;
    
    //First call to get metadata count
    CHECK_NVIMGCODEC(nvimgcodecDecoderGetMetadata(decoder_.get(), code_stream.handle(), nullptr, &metadata_count));

    if (metadata_count == 0) {
        return py::list();
    }

    //Create initial list of metadata objects
    std::vector<nvimgcodecMetadata_t*> metadata_ptrs(metadata_count);
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

    //Last call to get actual metadata
    CHECK_NVIMGCODEC(nvimgcodecDecoderGetMetadata(decoder_.get(), code_stream.handle(), metadata_ptrs.data(), &metadata_count));

    return metadata;
}

Metadata Decoder::getMetadata(const CodeStream& code_stream, uint16_t id, nvimgcodecMetadataKind_t kind)
{
    Metadata metadata_obj;
    nvimgcodecMetadata_t* metadata_ptr = metadata_obj.handle();
    
    // Set up metadata request
    metadata_ptr->id = id;
    metadata_ptr->kind = kind;
    metadata_ptr->format = NVIMGCODEC_METADATA_FORMAT_RAW;
    
    // Fields to retrieve:
    metadata_ptr->value_type = NVIMGCODEC_METADATA_VALUE_TYPE_UNKNOWN;
    metadata_ptr->value_count = 0;
    metadata_ptr->buffer = nullptr;
    metadata_ptr->buffer_size = 0;
    
    int metadata_count = 1;
    try {
        // First call to retrieve type and buffer size
        CHECK_NVIMGCODEC(nvimgcodecDecoderGetMetadata(decoder_.get(), code_stream.handle(), &metadata_ptr, &metadata_count));
        
        if (metadata_count == 0) {
            throw std::runtime_error("Metadata with ID " + std::to_string(id) + " not found");
        }
        
        // Allocate buffer for metadata
        metadata_obj.allocateBuffer();
        
        // Second call to get actual metadata
        CHECK_NVIMGCODEC(nvimgcodecDecoderGetMetadata(decoder_.get(), code_stream.handle(), &metadata_ptr, &metadata_count));
    } catch (const std::exception& e) {
        throw std::runtime_error("Could not get metadata with ID " + std::to_string(id));
    }
    return metadata_obj;
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
        .def("read", py::overload_cast<const CodeStream*, std::optional<Image*>, std::optional<DecodeParams>, intptr_t>(&Decoder::decode), R"pbdoc(
            Executes decoding from a CodeStream (typically created from a file).

            Args:
                path: CodeStream object (typically from a file path).

                image: Optional Image object to reuse for decoding. If provided, the Image's buffer will be
                       resized and reused instead of allocating a new Image. Only Images with internally
                       managed buffers can be reused. Defaults to None (create new Image).

                params: Decode parameters. Defaults to None (use default parameters).

                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.
                            Defaults to 0.

            Returns:
                nvimgcodec.Image or None if the image cannot be decoded because of any reason.
        )pbdoc",
            "path"_a, py::kw_only(), "image"_a = py::none(), "params"_a = py::none(), "cuda_stream"_a = 0)
        .def("read", py::overload_cast<const std::vector<const CodeStream*>&, std::optional<std::vector<Image*>>, std::optional<DecodeParams>, intptr_t>(&Decoder::decode),
            R"pbdoc(
            Executes decoding from a batch of CodeStreams (typically created from file paths).

            Args:
                paths: List of CodeStream objects (typically from file paths).

                images: Optional list of Image objects to reuse for decoding. If provided, the Images' buffers will be
                        resized and reused instead of allocating new Images. The list must have the same size as paths.
                        Only Images with internally managed buffers can be reused. Defaults to None (create new Images).

                params: Decode parameters. Defaults to None (use default parameters).

                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.
                            Defaults to 0.

            Returns:
                List of decoded nvimgcodec.Image's. There is None in returned list on positions which could not be decoded.

            )pbdoc",
            "paths"_a, py::kw_only(), "images"_a = py::none(), "params"_a = py::none(), "cuda_stream"_a = 0)
        .def("decode", py::overload_cast<const CodeStream*, std::optional<Image*>, std::optional<DecodeParams>, intptr_t>(&Decoder::decode), R"pbdoc(
            Executes decoding of data from a CodeStream handle.

            Args:
                src: CodeStream object.

                image: Optional Image object to reuse for decoding. If provided, the Image's buffer will be
                       resized and reused instead of allocating a new Image. Only Images with internally
                       managed buffers can be reused. Defaults to None (create new Image).

                params: Decode parameters. Defaults to None (use default parameters).

                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.
                            Defaults to 0.

            Returns:
                nvimgcodec.Image or None if the image cannot be decoded because of any reason.

            )pbdoc",
            "src"_a, py::kw_only(), "image"_a = py::none(), "params"_a = py::none(), "cuda_stream"_a = 0)
        .def("decode", py::overload_cast<const std::vector<const CodeStream*>&, std::optional<std::vector<Image*>>, std::optional<DecodeParams>, intptr_t>(&Decoder::decode),
            R"pbdoc(
            Executes decoding from a batch of CodeStream handles.

            Args:
                srcs: List of CodeStream objects

                images: Optional list of Image objects to reuse for decoding. If provided, the Images' buffers will be
                        resized and reused instead of allocating new Images. The list must have the same size as srcs.
                        Only Images with internally managed buffers can be reused. Defaults to None (create new Images).

                params: Decode parameters. Defaults to None (use default parameters).

                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.
                            Defaults to 0.

            Returns:
                List of decoded nvimgcodec.Image's. There is None in returned list on positions which could not be decoded.
            )pbdoc",
            "srcs"_a, py::kw_only(), "images"_a = py::none(), "params"_a = py::none(), "cuda_stream"_a = 0)
        .def("get_metadata", py::overload_cast<const CodeStream&, std::optional<nvimgcodecMetadataKind_t>>(&Decoder::getMetadata), R"pbdoc(
            Retrieves metadata from a code stream.

            Args:
                code_stream: The code stream to get metadata from.
                kind: Optional metadata kind to filter by. If specified, only metadata of that kind will be returned.

            Returns:
                A list of Metadata objects. If no metadata exists or no metadata matches the specified kind,
                returns an empty list.
            )pbdoc",
            "code_stream"_a, py::kw_only(), "kind"_a = py::none())
        .def("get_metadata", py::overload_cast<const CodeStream&, uint16_t, nvimgcodecMetadataKind_t>(&Decoder::getMetadata), R"pbdoc(
            Retrieves a specific metadata by ID from a code stream.

            Args:
                code_stream: The code stream to get metadata from.
                id: The specific metadata ID to retrieve.
                kind: Metadata kind. Defaults to TIFF tag kind if not specified.

            Returns:
                A single Metadata object for the specified id.
                
            Raises:
                RuntimeError: If metadata with the specified ID is not found.
            )pbdoc",
            "code_stream"_a, py::kw_only(), "id"_a, "kind"_a = NVIMGCODEC_METADATA_KIND_TIFF_TAG)

        .def("__enter__", &Decoder::enter, "Enter the runtime context related to this decoder.")
        .def("__exit__", &Decoder::exit, "Exit the runtime context related to this decoder and releases allocated resources.",
            "exc_type"_a = py::none(), "exc_value"_a = py::none(), "traceback"_a = py::none());
    // clang-format on
}

} // namespace nvimgcodec
