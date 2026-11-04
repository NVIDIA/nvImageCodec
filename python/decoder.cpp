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

#include <algorithm>
#include <iostream>
#include <optional>
#include <string_view>

#include <ilogger.h>
#include <log.h>

#include "backend.h"
#include "device_utils.h"
#include "error_handling.h"
#include "imgproc/exception.h"
#include "imgproc/type_utils.h"
#include "metadata.h"
#include "metadata_kind.h"
#include "nvimgcodec.h"
#include "plugin_selection_arg.h"
#include "region.h"
#include "type_utils.h"

namespace nvimgcodec {

Decoder::Decoder(nvimgcodecInstance_t instance, ILogger* logger, int device_id, int max_num_cpu_threads,
    std::optional<std::vector<Backend>> backends, const std::string& options, const py::object& decoder_ids)
    : decoder_(nullptr)
    , instance_(instance)
    , logger_(logger)
    , device_id_(resolve_device_id(device_id))
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
    PluginSelectionArg plugin_selection(decoder_ids, "decoder_ids");
    plugin_selection.validateDecoderIds(instance);
    plugin_selection.apply(&exec_params);

    is_cpu_only_ = nvimgcds_backends.size() > 0;
    for (size_t i = 0; is_cpu_only_ && i < nvimgcds_backends.size(); i++) {
        if (nvimgcds_backends[i].kind != NVIMGCODEC_BACKEND_KIND_CPU_ONLY)
            is_cpu_only_ = false;
    }

    nvimgcodecStatus_t status = nvimgcodecDecoderCreate(instance, &decoder, &exec_params, options.c_str());
    if (status != NVIMGCODEC_STATUS_SUCCESS) {
        throw Exception(INTERNAL_ERROR, "Could not create decoder. Code: " + std::to_string(status));
    }
    decoder_ = std::shared_ptr<std::remove_pointer<nvimgcodecDecoder_t>::type>(
        decoder, [](nvimgcodecDecoder_t decoder) { nvimgcodecDecoderDestroy(decoder); });
}

Decoder::~Decoder()
{
}

py::object Decoder::decode(const CodeStream* code_stream, std::optional<Image*> image, std::optional<DecodeParams> params, CudaStreamArg cuda_stream)
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
    CudaStreamArg cuda_stream)
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
    
            image_info.cuda_stream = reinterpret_cast<cudaStream_t>(cuda_stream.handle);

            // Resolve the desired output format from the user's DecodeParams
            // (sample_format + color_spec) plus the source's detected metadata.
            // The resolver enforces compatibility between sample_format and
            // color_spec and picks layout (interleaved vs planar) and concrete
            // sample_format / color_spec / component count.
            // The source's subsampling, captured before image_info is repurposed as the
            // *output* description below. resolveChromaSubsampling needs it: decoders emit
            // chroma at the code stream's rate, so a mismatched request would size the
            // output planes for a rate the decoder does not write.
            const nvimgcodecChromaSubsampling_t source_chroma_subsampling = image_info.chroma_subsampling;
            int source_num_channels = 0;
            for (uint32_t c = 0; c < image_info.num_planes; ++c) {
                source_num_channels += image_info.plane_info[c].num_channels;
            }
            ResolvedFormat resolved = resolveOutputFormat(
                params.sample_format_,
                params.color_spec_,
                image_info.sample_format,
                image_info.color_spec,
                source_num_channels);

            image_info.sample_format = resolved.sample_format;
            image_info.color_spec    = resolved.color_spec;
            image_info.num_planes    = resolved.is_interleaved ? 1u : static_cast<uint32_t>(resolved.num_components);

            // The caller's chroma_subsampling wins; with none given this reproduces the
            // previous behaviour exactly - GRAY for luma-only outputs, NONE otherwise.
            image_info.chroma_subsampling =
                resolveChromaSubsampling(params.chroma_subsampling_, resolved, source_chroma_subsampling);
    
            int decode_out_height = image_info.plane_info[0].height;
            int decode_out_width = image_info.plane_info[0].width;
            const bool swap_wh = params.decode_params_.apply_exif_orientation && ((image_info.orientation.rotated / 90) % 2);
            if (roi) {
                // When apply_exif_orientation is on, the region is already in display
                // (post-orientation) coords, so its extent is the output buffer size and
                // does not need an additional swap. When apply_exif_orientation is off,
                // the region is in codestream coords; output dims also match without swap.
                nvimgcodecRegion_t region = static_cast<nvimgcodecRegion_t>(roi.value());
                decode_out_height = region.end[0] - region.start[0];
                decode_out_width = region.end[1] - region.start[1];
            } else if (swap_wh) {
                std::swap(decode_out_height, decode_out_width);
            }

            // The sampling axes rotate with the image: a 90/270 EXIF rotation turns a
            // horizontally subsampled source into a vertically subsampled output, so both the
            // divisors and the reported chroma_subsampling label transpose alongside the
            // extents (e.g. 422 + 90 degrees becomes 440) - otherwise Image.chroma_subsampling
            // would keep describing the pre-rotation layout while the planes themselves (and
            // anything that sizes buffers from the label, like Encoder::encode_batch_impl)
            // moved on. This is independent of how the extents were derived above - an ROI is
            // already in display coordinates, but its chroma grid is rotated just the same.
            if (swap_wh) {
                image_info.chroma_subsampling = transposeChromaSubsampling(image_info.chroma_subsampling);
            }
            const auto css = chromaSubsamplingFactors(image_info.chroma_subsampling);
            const uint32_t css_x = css.first;
            const uint32_t css_y = css.second;
            for (uint32_t c = 0; c < image_info.num_planes; ++c) {
                // Plane 0 is luma; the remaining planes carry chroma at the reduced rate.
                const bool is_chroma = (c > 0) && !resolved.is_interleaved;
                const uint32_t pw = is_chroma ? (decode_out_width + css_x - 1) / css_x
                                              : static_cast<uint32_t>(decode_out_width);
                const uint32_t ph = is_chroma ? (decode_out_height + css_y - 1) / css_y
                                              : static_cast<uint32_t>(decode_out_height);
                auto& plane_info = image_info.plane_info[c];
                plane_info.height = ph;
                plane_info.width = pw;
                plane_info.sample_type = sample_type;
                plane_info.precision = precision;
                plane_info.num_channels = resolved.is_interleaved ? static_cast<uint32_t>(resolved.num_components) : 1u;
                plane_info.row_stride = static_cast<size_t>(pw) * bytes_per_element * plane_info.num_channels;
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
                    // Two independent stream concerns:
                    // 1) detachFromUserStream — buffer allocation may use
                    //    cudaMallocAsync on the caller's stream. Decoding only
                    //    borrows that stream for this call; the C++ deleter must
                    //    not keep it (the user may destroy the stream when
                    //    read/decode returns).
                    // 2) setCudaStreamOwner — if cuda_stream was a protocol
                    //    object, pin it so CUDA Array Interface /
                    //    Image.cuda_stream can still advertise that handle.
                    //    Bare int handles are not pinned.
                    reuse_img->detachFromUserStream();
                    reuse_img->setCudaStreamOwner(cuda_stream.owner);
                    images.push_back(reuse_img->getNvImgCdcsImage());
                    code_streams.push_back(code_stream->handle());
                    py_images.push_back(py::cast(reuse_img, py::return_value_policy::reference));
                    orig_sample_idx.push_back(i);
                } else {
                    // Create new Image
                    Image img(instance_, logger_, &image_info, device_id_);
                    // Same split as the reuse path above: detach the deleter
                    // from the caller's stream, but pin a protocol object so
                    // CUDA Array Interface / Image.cuda_stream keep a live handle.
                    img.detachFromUserStream();
                    img.setCudaStreamOwner(cuda_stream.owner);
                    images.push_back(img.getNvImgCdcsImage());
                    code_streams.push_back(code_stream->handle());
                    py_images.push_back(py::cast(std::move(img)));
                    orig_sample_idx.push_back(i);
                }
            } catch (const std::invalid_argument& e) {
                py::gil_scoped_acquire acquire;
                throw;
            } catch (const std::exception& e) {
                NVIMGCODEC_LOG_WARNING(logger_, "Something went wrong during decoding image #" << i << " (" << e.what() << "). There will be None on corresponding output position");
                py::gil_scoped_acquire acquire;
                py_images.push_back(py::none());
                continue;
            }
        } catch (const std::invalid_argument& e) {
            py::gil_scoped_acquire acquire;
            throw;
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
            } else if (params.decode_params_.apply_exif_orientation) {
                // The output Image's orientation was left over from the source code stream's
                // parsed EXIF tag and was only used by the decoder backend to know how to
                // rotate/flip the pixels it just wrote. With apply_exif_orientation on, that
                // transform has now been applied to the buffer, so a non-identity value here
                // would be stale metadata describing a transform the pixels no longer need -
                // encoding this image would then be refused by encoders (which do not apply
                // orientation) for a rotation that was never actually requested. Reset it to
                // identity now that decoding is done.
                nvimgcodecImageInfo_t out_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), nullptr};
                CHECK_NVIMGCODEC(nvimgcodecImageGetImageInfo(images[i], &out_info));
                const auto& o = out_info.orientation;
                if (o.rotated != 0 || o.flip_x || o.flip_y) {
                    out_info.orientation = {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 0, 0, 0};
                    // Not a new image: with a non-null handle, nvimgcodecImageCreate updates that
                    // image's info in place (the same reuse Image::reuse() relies on).
                    assert(images[i] != nullptr);
                    CHECK_NVIMGCODEC(nvimgcodecImageCreate(instance_, &images[i], &out_info));
                }
            }
        }
    } catch (...) {
        NVIMGCODEC_LOG_WARNING(logger_, "Something went wrong during decoding and there will be None on corresponding output positions");
        py::gil_scoped_acquire acquire;
        py_images.assign(orig_nsamples, py::none());

    }
    return py_images;
}

int Decoder::getDeviceId() const
{
    return device_id_;
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

py::object Decoder::getMetadata(const CodeStream& code_stream, std::optional<nvimgcodecMetadataKind_t> kind)
{
    // Kinds that return a single metadata object
    if (kind && kind.value() == NVIMGCODEC_METADATA_KIND_TIFF_TAG_LIST) {
        return py::cast(getMetadata(code_stream, 0, kind.value()));
    }

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

std::optional<Metadata> Decoder::getMetadata(const CodeStream& code_stream, int id, nvimgcodecMetadataKind_t kind)
{
    if (id < 0 || id > 65535) {
        throw std::invalid_argument("TIFF tag ID must be in range [0, 65535] (16-bit unsigned), got " + std::to_string(id));
    }

    Metadata metadata_obj;
    nvimgcodecMetadata_t* metadata_ptr = metadata_obj.handle();
    
    // Set up metadata request
    metadata_ptr->id = static_cast<uint16_t>(id);
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
            return std::nullopt;
        }
        
        // Allocate buffer for metadata
        metadata_obj.allocateBuffer();
        
        // Second call to get actual metadata
        CHECK_NVIMGCODEC(nvimgcodecDecoderGetMetadata(decoder_.get(), code_stream.handle(), &metadata_ptr, &metadata_count));
    } catch (const std::exception& e) {
        if (kind == NVIMGCODEC_METADATA_KIND_TIFF_TAG) {
            throw std::runtime_error("Could not get metadata with tag ID " + std::to_string(id));
        } else {
            throw std::runtime_error("Could not get metadata with kind " + std::to_string(static_cast<int>(kind)));
        }
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
                               const std::string& options, const py::object& decoder_ids) {
                return new Decoder(instance, logger, device_id, max_num_cpu_threads, backends, options, decoder_ids);
            }),
            R"pbdoc(
            Initialize decoder.

            Args:
                device_id: Device id to execute decoding on.

                max_num_cpu_threads: Max number of CPU threads in default executor (0 means default value equal to number of cpu cores).

                backends: List of allowed backends. If empty, all backends are allowed with default parameters.

                options: Optional space-separated decoder options. Use ``:<option>=<value>`` for global
                    options (e.g. ``:num_cuda_streams=4``) or options applied by any decoder that
                    supports them (e.g. ``:fancy_upsampling=1``). Use ``<decoder_id>:<option>=<value>``
                    for options that apply only to a specific decoder (e.g.
                    ``nvjpeg_cuda_decoder:hybrid_huffman_threshold=0``). See the documentation
                    section "Decoder options format" for the full list of options per decoder.
                    Default is ``""`` which keeps each extension defaults.

                decoder_ids: Optional ordered decoder allow-list. Pass a sequence of identifiers
                    returned by ``get_decoder_ids()`` to select implementations across codecs, or
                    a mapping from codec names to ordered identifier sequences. Unlisted
                    implementations and, for mappings, codecs without an entry are disabled.
                    Backend restrictions take precedence over this list.

            )pbdoc",
            "device_id"_a = NVIMGCODEC_DEVICE_CURRENT, "max_num_cpu_threads"_a = 0, "backends"_a = py::none(),
            "options"_a = "", py::kw_only(), "decoder_ids"_a = py::none())
        .def("read", py::overload_cast<const CodeStream*, std::optional<Image*>, std::optional<DecodeParams>, CudaStreamArg>(&Decoder::decode), R"pbdoc(
            Executes decoding from a CodeStream (typically created from a file).

            Args:
                path: CodeStream object (typically from a file path).

                image: Optional Image object to reuse for decoding. If provided, the Image's buffer will be
                       resized and reused instead of allocating a new Image. Only Images with internally
                       managed buffers can be reused. Defaults to None (create new Image).

                params: Decode parameters. Defaults to None (use default parameters).

                cuda_stream: Optional CUDA stream upon which synchronization must take place.
                            Accepts ``None`` (unspecified), a ``cudaStream_t`` handle as a
                            Python integer, or any object implementing the CUDA Stream
                            Protocol (``__cuda_stream__()`` -> ``(version, stream_ptr)``,
                            version 0). Unspecified (``None`` or omitted) is handle ``0``
                            (legacy default stream).
                            Each decoded Image holds a protocol object until that Image is
                            released, so ``Image.cuda_stream`` and
                            ``__cuda_array_interface__['stream']`` stay valid. A bare
                            integer handle is not retained; the caller must keep that
                            stream alive while those values are used.

            Returns:
                nvimgcodec.Image or None if the image cannot be decoded because of any reason.
        )pbdoc",
            "path"_a, py::kw_only(), "image"_a = py::none(), "params"_a = py::none(), "cuda_stream"_a = py::none())
        .def("read", py::overload_cast<const std::vector<const CodeStream*>&, std::optional<std::vector<Image*>>, std::optional<DecodeParams>, CudaStreamArg>(&Decoder::decode),
            R"pbdoc(
            Executes decoding from a batch of CodeStreams (typically created from file paths).

            Args:
                paths: List of CodeStream objects (typically from file paths).

                images: Optional list of Image objects to reuse for decoding. If provided, the Images' buffers will be
                        resized and reused instead of allocating new Images. The list must have the same size as paths.
                        Only Images with internally managed buffers can be reused. Defaults to None (create new Images).

                params: Decode parameters. Defaults to None (use default parameters).

                cuda_stream: Optional CUDA stream upon which synchronization must take place.
                            Accepts ``None`` (unspecified), a ``cudaStream_t`` handle as a
                            Python integer, or any object implementing the CUDA Stream
                            Protocol (``__cuda_stream__()`` -> ``(version, stream_ptr)``,
                            version 0). Unspecified (``None`` or omitted) is handle ``0``
                            (legacy default stream).
                            Each decoded Image holds a protocol object until that Image is
                            released, so ``Image.cuda_stream`` and
                            ``__cuda_array_interface__['stream']`` stay valid. A bare
                            integer handle is not retained; the caller must keep that
                            stream alive while those values are used.

            Returns:
                List of decoded nvimgcodec.Image's. There is None in returned list on positions which could not be decoded.

            )pbdoc",
            "paths"_a, py::kw_only(), "images"_a = py::none(), "params"_a = py::none(), "cuda_stream"_a = py::none())
        .def("decode", py::overload_cast<const CodeStream*, std::optional<Image*>, std::optional<DecodeParams>, CudaStreamArg>(&Decoder::decode), R"pbdoc(
            Executes decoding of data from a CodeStream handle.

            Args:
                src: CodeStream object.

                image: Optional Image object to reuse for decoding. If provided, the Image's buffer will be
                       resized and reused instead of allocating a new Image. Only Images with internally
                       managed buffers can be reused. Defaults to None (create new Image).

                params: Decode parameters. Defaults to None (use default parameters).

                cuda_stream: Optional CUDA stream upon which synchronization must take place.
                            Accepts ``None`` (unspecified), a ``cudaStream_t`` handle as a
                            Python integer, or any object implementing the CUDA Stream
                            Protocol (``__cuda_stream__()`` -> ``(version, stream_ptr)``,
                            version 0). Unspecified (``None`` or omitted) is handle ``0``
                            (legacy default stream).
                            Each decoded Image holds a protocol object until that Image is
                            released, so ``Image.cuda_stream`` and
                            ``__cuda_array_interface__['stream']`` stay valid. A bare
                            integer handle is not retained; the caller must keep that
                            stream alive while those values are used.

            Returns:
                nvimgcodec.Image or None if the image cannot be decoded because of any reason.

            )pbdoc",
            "src"_a, py::kw_only(), "image"_a = py::none(), "params"_a = py::none(), "cuda_stream"_a = py::none())
        .def("decode", py::overload_cast<const std::vector<const CodeStream*>&, std::optional<std::vector<Image*>>, std::optional<DecodeParams>, CudaStreamArg>(&Decoder::decode),
            R"pbdoc(
            Executes decoding from a batch of CodeStream handles.

            Args:
                srcs: List of CodeStream objects

                images: Optional list of Image objects to reuse for decoding. If provided, the Images' buffers will be
                        resized and reused instead of allocating new Images. The list must have the same size as srcs.
                        Only Images with internally managed buffers can be reused. Defaults to None (create new Images).

                params: Decode parameters. Defaults to None (use default parameters).

                cuda_stream: Optional CUDA stream upon which synchronization must take place.
                            Accepts ``None`` (unspecified), a ``cudaStream_t`` handle as a
                            Python integer, or any object implementing the CUDA Stream
                            Protocol (``__cuda_stream__()`` -> ``(version, stream_ptr)``,
                            version 0). Unspecified (``None`` or omitted) is handle ``0``
                            (legacy default stream).
                            Each decoded Image holds a protocol object until that Image is
                            released, so ``Image.cuda_stream`` and
                            ``__cuda_array_interface__['stream']`` stay valid. A bare
                            integer handle is not retained; the caller must keep that
                            stream alive while those values are used.

            Returns:
                List of decoded nvimgcodec.Image's. There is None in returned list on positions which could not be decoded.
            )pbdoc",
            "srcs"_a, py::kw_only(), "images"_a = py::none(), "params"_a = py::none(), "cuda_stream"_a = py::none())
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
        .def("get_metadata", py::overload_cast<const CodeStream&, int, nvimgcodecMetadataKind_t>(&Decoder::getMetadata), R"pbdoc(
            Retrieves a specific metadata by ID from a code stream.

            Args:
                code_stream: The code stream to get metadata from.
                id: The specific metadata ID to retrieve.
                kind: Metadata kind. Defaults to TIFF tag kind if not specified.

            Returns:
                A Metadata object for the specified id, or None if the metadata is not found.
            )pbdoc",
            "code_stream"_a, py::kw_only(), "id"_a, "kind"_a = NVIMGCODEC_METADATA_KIND_TIFF_TAG)

        .def_property_readonly("device_id", &Decoder::getDeviceId,
            R"pbdoc(
            The CUDA device id this decoder executes on.
            )pbdoc")
        .def("__enter__", &Decoder::enter, "Enter the runtime context related to this decoder.")
        .def("__exit__", &Decoder::exit, "Exit the runtime context related to this decoder and releases allocated resources.",
            "exc_type"_a = py::none(), "exc_value"_a = py::none(), "traceback"_a = py::none());
    // clang-format on
}

} // namespace nvimgcodec
