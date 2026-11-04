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
#include "type_utils.h"

#include <string.h>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <optional>
#include <type_traits>
#include <vector>
#include <memory>

#include <ilogger.h>
#include <log.h>

#include "backend.h"
#include "device_utils.h"
#include "error_handling.h"
#include "imgproc/exception.h"
#include "plugin_selection_arg.h"

namespace fs = std::filesystem;

namespace nvimgcodec {

namespace {

std::string default_codec_name_for_file_extension(nvimgcodecInstance_t instance, const std::string& file_extension)
{
    const nvimgcodecFrameworkDesc_t* framework = nullptr;
    nvimgcodecStatus_t status = nvimgcodecGetFrameworkDesc(instance, &framework);
    if (status != NVIMGCODEC_STATUS_SUCCESS) {
        return {};
    }
    int total_count = 0;
    status = framework->getCodecNames(framework->instance, nullptr, &total_count, file_extension.c_str());
    if (status != NVIMGCODEC_STATUS_SUCCESS || total_count == 0) {
        return {};
    }
    char name[NVIMGCODEC_MAX_CODEC_NAME_SIZE] = {};
    int count = 1;
    status = framework->getCodecNames(framework->instance, name, &count, file_extension.c_str());
    if (status != NVIMGCODEC_STATUS_SUCCESS || count == 0) {
        return {};
    }
    return std::string(name);
}

// Chooses the chroma subsampling to use when the caller didn't specify one.
// Images with at least 3 channels use 4:2:0 for JPEG - it is the broadly
// compatible choice and the only mode the nvJPEG hardware encoder accepts -
// and full-resolution 4:4:4 for other codecs. Images with fewer than 3 channels
// (grayscale and grayscale+alpha) default to grayscale.
nvimgcodecChromaSubsampling_t default_chroma_subsampling(const std::string& codec_name, uint32_t total_channels)
{
    if (total_channels < 3)
        return NVIMGCODEC_SAMPLING_GRAY;
    if (codec_name == "jpeg")
        return NVIMGCODEC_SAMPLING_420;
    return NVIMGCODEC_SAMPLING_444;
}

} // namespace

Encoder::Encoder(nvimgcodecInstance_t instance, ILogger* logger, int device_id, int max_num_cpu_threads,
    std::optional<std::vector<Backend>> backends, const std::string& options, const py::object& encoder_ids)
    : encoder_(nullptr)
    , instance_(instance)
    , logger_(logger)
    , device_id_(resolve_device_id(device_id))
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
    PluginSelectionArg plugin_selection(encoder_ids, "encoder_ids");
    plugin_selection.validateEncoderIds(instance);
    plugin_selection.apply(&exec_params);

    is_cpu_only_ = device_id_ == NVIMGCODEC_DEVICE_CPU_ONLY;
    if (!is_cpu_only_ && nvimgcds_backends.size() > 0) {
        is_cpu_only_ = true;
        for (size_t i = 0; is_cpu_only_ && i < nvimgcds_backends.size(); ++i) {
            if (nvimgcds_backends[i].kind != NVIMGCODEC_BACKEND_KIND_CPU_ONLY)
                is_cpu_only_ = false;
        }
    }

    nvimgcodecStatus_t status = nvimgcodecEncoderCreate(instance, &encoder, &exec_params, options.c_str());
    if (status != NVIMGCODEC_STATUS_SUCCESS) {
        throw Exception(INTERNAL_ERROR, "Could not create encoder. Code: " + std::to_string(status));
    }

    encoder_ = std::shared_ptr<std::remove_pointer<nvimgcodecEncoder_t>::type>(
        encoder, [](nvimgcodecEncoder_t encoder) { nvimgcodecEncoderDestroy(encoder); });
}

Encoder::~Encoder()
{
}

void Encoder::encode_batch_impl(const std::vector<const Image*>& images, const std::optional<EncodeParams>& params_opt, intptr_t cuda_stream,
    const std::vector<std::string>& codec_names,
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

    // Encoders read every input plane as full size, so a chroma-subsampled image (as
    // decoded with DecodeParams.chroma_subsampling) would have its smaller chroma planes
    // misread. Checked for the whole batch up front so that a rejected batch leaves any
    // caller-provided code streams untouched.
    for (size_t i = 0; i < orig_batch_size; i++) {
        if (!images[i])
            continue;
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        nvimgcodecImageGetImageInfo(images[i]->getNvImgCdcsImage(), &image_info);
        if (!is_unsubsampled_chroma(image_info.chroma_subsampling)) {
            throw std::invalid_argument("Image #" + std::to_string(i) +
                                        " is chroma-subsampled. Encoding from chroma-subsampled planes is not "
                                        "supported yet; decode without DecodeParams.chroma_subsampling to get an "
                                        "image that can be encoded.");
        }
    }

    for (size_t i = 0; i < orig_batch_size; i++) {
        if (images[i]) {
            auto img = images[i]->getNvImgCdcsImage();
            valid_images.push_back(img); 
            valid_image_idx[i] = valid_images.size() - 1;

            nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            nvimgcodecImageGetImageInfo(valid_images.back(), &image_info);

            nvimgcodecImageInfo_t out_image_info(image_info);

            // chroma_subsampling: caller's value wins; otherwise pick a codec
            // dependent default per image (the codec can differ per image when
            // writing to files with different extensions). Total channel count is
            // num_planes for planar (one channel per plane) and
            // plane[0].num_channels for interleaved - max() covers both.
            if (params.chroma_subsampling_.has_value()) {
                out_image_info.chroma_subsampling = params.chroma_subsampling_.value();
            } else {
                const uint32_t total_channels =
                    std::max(image_info.num_planes, image_info.plane_info[0].num_channels);
                out_image_info.chroma_subsampling = default_chroma_subsampling(codec_names[i], total_channels);
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
        nvimgcodecFuture_t encode_future = nullptr;
        CHECK_NVIMGCODEC(nvimgcodecEncoderEncode(
            encoder_.get(), valid_images.data(), code_streams.data(), valid_images.size(), params.handle(), &encode_future));
        // Take ownership before any other checked call can throw. This keeps
        // error paths from leaking the native future returned by the C API.
        std::unique_ptr<std::remove_pointer_t<nvimgcodecFuture_t>, decltype(&nvimgcodecFutureDestroy)> future(
            encode_future, &nvimgcodecFutureDestroy);
        CHECK_NVIMGCODEC(nvimgcodecFutureWaitForAll(future.get()));

        size_t status_size;
        CHECK_NVIMGCODEC(nvimgcodecFutureGetProcessingStatus(future.get(), encode_status.data(), &status_size));
        assert(status_size == encode_status.size());
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
    std::string codec_name = codec[0] == '.' ? default_codec_name_for_file_extension(instance_, codec) : codec;
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
            new_code_streams[i] = std::make_unique<CodeStream>(instance_, logger_, out_image_info, !is_cpu_only_);
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

    // encode() targets a single codec for the whole batch.
    std::vector<std::string> codec_names(orig_batch_size, codec_name);

    encode_batch_impl(images, params, cuda_stream, codec_names, create_code_stream, post_encode_callback);

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
py::object Encoder::encode(const py::object& image_s, const std::string& codec, std::optional<py::object> code_stream_s, const std::optional<EncodeParams>& params, CudaStreamArg cuda_stream)
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
        std::vector<py::object> result = encode_batch(image_ptrs, codec, code_streams, params, cuda_stream.handle);
        
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
        return encode_image(image_ptrs[0], codec, single_code_stream, params, cuda_stream.handle);
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

    // Resolve the codec per file: the codec (and therefore the chroma-subsampling
    // default) can differ from image to image when writing to file names with
    // different extensions. Resolved up front so the value is shared between the
    // chroma-subsampling default and code stream creation, with any fallback
    // warning logged once per (valid) item.
    std::vector<std::string> codec_names(orig_batch_size);
    for (size_t i = 0; i < orig_batch_size; ++i) {
        if (!images[i])
            continue; // skipped items never reach create_code_stream / the chroma default
        std::string& item_codec_name = codec_names[i];
        if (codec.empty()) {
            auto file_extension = fs::path(file_names[i]).extension();
            item_codec_name = default_codec_name_for_file_extension(instance_, file_extension.string());
            if (item_codec_name.empty()) {
                NVIMGCODEC_LOG_WARNING(logger_, "File '" << file_names[i] << "' without extension. As default choosing jpeg codec");
                item_codec_name = "jpeg";
            }
        } else {
            item_codec_name = codec[0] == '.' ? default_codec_name_for_file_extension(instance_, codec) : codec;
            if (item_codec_name.empty()) {
                NVIMGCODEC_LOG_WARNING(logger_, "Unsupported codec.  As default choosing jpeg codec");
                item_codec_name = "jpeg";
            }
        }
    }

    auto create_code_stream = [&](size_t i, nvimgcodecImageInfo_t& out_image_info, nvimgcodecCodeStream_t* code_stream) -> void {
        strcpy(out_image_info.codec_name, codec_names[i].c_str());
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
    encode_batch_impl(images, params, cuda_stream, codec_names, create_code_stream, post_encode_callback);

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

py::object Encoder::write(const std::string& file_name, const py::object& image, const std::string& codec, const std::optional<EncodeParams>& params, CudaStreamArg cuda_stream)
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

    return write_image(file_name, image_ptrs[0], codec, params, cuda_stream.handle);
}

std::vector<py::object> Encoder::write(const std::vector<std::string>& file_names, const std::vector<py::object>& images, const std::string& codec,
    const std::optional<EncodeParams>& params, CudaStreamArg cuda_stream)
{
    std::vector<Image> image_raii;
    std::vector<const Image*> image_ptrs = convertPyObjectsToImages(images, cuda_stream, image_raii);
    return write_batch(file_names, image_ptrs, codec, params, cuda_stream.handle);
}

// Helper function to convert Python objects to Image pointers with exception handling
std::vector<const Image*> Encoder::convertPyObjectsToImages(const std::vector<py::object>& py_images, CudaStreamArg cuda_stream, std::vector<Image>& image_raii)
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

int Encoder::getDeviceId() const
{
    return device_id_;
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
                     const std::string& options, const py::object& encoder_ids) {
                     return new Encoder(instance, logger, device_id, max_num_cpu_threads, backends, options, encoder_ids);
                 }),
            R"pbdoc(
            Initialize encoder.

            Args:
                device_id: Device id to execute encoding on.

                max_num_cpu_threads: Max number of CPU threads in default executor (0 means default value equal to number of cpu cores)
                
                backends: List of allowed backends. If empty, all backends are allowed with default parameters.

                options: Optional space-separated encoder options. Use ``:<option>=<value>`` for global
                    options (e.g. ``:num_cuda_streams=4``) or ``<encoder_id>:<option>=<value>`` for
                    encoder-specific options. Encoder IDs include ``nvjpeg_cuda_encoder``, ``nvjpeg2k_encoder``,
                    ``nvtiff_cuda_encoder``, ``nvpnm_encoder``, ``nvbmp_encoder``, and OpenCV encoders
                    (e.g. ``opencv_jpeg_encoder``). See the documentation section "Encoder options format"
                    for the full list and details; built-in encoders currently only honor global options.

                encoder_ids: Optional ordered encoder allow-list. Pass a sequence of identifiers
                    returned by ``get_encoder_ids()`` to select implementations across codecs, or
                    a mapping from codec names to ordered identifier sequences. Unlisted
                    implementations and, for mappings, codecs without an entry are disabled.
                    Backend restrictions take precedence over this list.

            )pbdoc",
            "device_id"_a = NVIMGCODEC_DEVICE_CURRENT, "max_num_cpu_threads"_a = 0, "backends"_a = py::none(), "options"_a = "",
            py::kw_only(), "encoder_ids"_a = py::none())
        .def("encode", py::overload_cast<const py::object&, const std::string&, std::optional<py::object>, const std::optional<EncodeParams>&, CudaStreamArg>(&Encoder::encode),
            R"pbdoc(
            Encode image(s) to CodeStream(s).

            Args:
                image_s: Image to encode or list of images to encode (can be any Python object that can be converted to Image, or a list of such objects)

                codec: Output codec name such as ``"jpeg2k"``, or a registered file extension
                       including its leading period such as ``".jp2"``. An extension resolves to
                       its default registered codec.
                
                code_stream_s: CodeStream or list of CodeStreams to encode image to. If None, a new CodeStream(s) will be created.
                
                params: Encode parameters.
                
                cuda_stream: Optional CUDA stream used when wrapping an input that is
                             not already an ``Image`` (array-like objects). Accepts
                             ``None`` (unspecified), a ``cudaStream_t`` handle as a
                             Python integer, or any object implementing the CUDA Stream
                             Protocol (``__cuda_stream__()`` -> ``(version, stream_ptr)``,
                             version 0). Unspecified (``None`` or omitted) uses handle
                             ``0`` (legacy default stream) unless the wrapped source
                             exposes ``__cuda_array_interface__['stream']``, in which
                             case that stream is used. An explicit ``cuda_stream``
                             (including ``0``) takes precedence over the CUDA Array
                             Interface value. Inputs that are already ``Image`` keep
                             the stream they were created with; this argument is ignored
                             for those. Encode itself uses each image's associated
                             stream, not this argument. Any Image constructed from this
                             argument holds a protocol object until that Image is
                             released, so ``Image.cuda_stream`` and
                             ``__cuda_array_interface__['stream']`` stay valid. A bare
                             integer handle is not retained; the caller must keep that
                             stream alive while those values are used.

            Returns:
                For single image: CodeStream with encoded image, or None if the input cannot be converted to Image or if encoding failed.
                For list of images: List of CodeStreams with encoded images. If an input cannot be converted to Image or if encoding failed, the corresponding position will contain None.

            Raises:
                ValueError: If any image is chroma-subsampled, as produced by decoding with
                            ``DecodeParams.chroma_subsampling``. Decode without it to get an image
                            that can be encoded. The whole batch is rejected before any
                            CodeStream is modified.
            )pbdoc",
            "image_s"_a, "codec"_a, py::kw_only(), "code_stream_s"_a = py::none(), "params"_a = py::none(), "cuda_stream"_a = py::none())
        .def("write",
            py::overload_cast<const std::string&, const py::object&, const std::string&, const std::optional<EncodeParams>&, CudaStreamArg>(&Encoder::write),
            R"pbdoc(
            Encode image to file.

            Args:
                file_name: File name to save encoded code stream.

                image: Image to encode (can be any Python object that can be converted to Image)

                codec: Optional output codec name or registered file extension. When omitted, the
                       codec is inferred from ``file_name`` using the registered file-extension
                       mappings. If the filename has no registered extension, JPEG is used.

                params: Encode parameters.

                cuda_stream: Optional CUDA stream used when wrapping an input that is
                             not already an ``Image`` (array-like objects). Accepts
                             ``None`` (unspecified), a ``cudaStream_t`` handle as a
                             Python integer, or any object implementing the CUDA Stream
                             Protocol (``__cuda_stream__()`` -> ``(version, stream_ptr)``,
                             version 0). Unspecified (``None`` or omitted) uses handle
                             ``0`` (legacy default stream) unless the wrapped source
                             exposes ``__cuda_array_interface__['stream']``, in which
                             case that stream is used. An explicit ``cuda_stream``
                             (including ``0``) takes precedence over the CUDA Array
                             Interface value. Inputs that are already ``Image`` keep
                             the stream they were created with; this argument is ignored
                             for those. Encode itself uses each image's associated
                             stream, not this argument. Any Image constructed from this
                             argument holds a protocol object until that Image is
                             released, so ``Image.cuda_stream`` and
                             ``__cuda_array_interface__['stream']`` stay valid. A bare
                             integer handle is not retained; the caller must keep that
                             stream alive while those values are used.

            Returns: 
                Encoded file name, or None if the input image could not be encoded for any reason.

            Raises:
                ValueError: If the image is chroma-subsampled, as produced by decoding with
                            ``DecodeParams.chroma_subsampling``. Decode without it to get an image
                            that can be encoded.
            )pbdoc",
            "file_name"_a, "image"_a, "codec"_a = "", "params"_a = py::none(), "cuda_stream"_a = py::none())
        .def("write",
            py::overload_cast<const std::vector<std::string>&, const std::vector<py::object>&, const std::string&,
                const std::optional<EncodeParams>&, CudaStreamArg>(&Encoder::write),
            R"pbdoc(
            Encode batch of images to files.

            Args:
                file_names: List of file names to save encoded code streams.

                images: List of images to encode (can contain any Python objects that can be converted to Image)

                codec: Optional output codec name or registered file extension. When omitted, the
                       codec is inferred independently for every filename using the registered
                       file-extension mappings. This permits a batch to contain different output
                       codecs. A filename without a registered extension uses JPEG.
                
                params: Encode parameters.
                
                cuda_stream: Optional CUDA stream used when wrapping an input that is
                             not already an ``Image`` (array-like objects). Accepts
                             ``None`` (unspecified), a ``cudaStream_t`` handle as a
                             Python integer, or any object implementing the CUDA Stream
                             Protocol (``__cuda_stream__()`` -> ``(version, stream_ptr)``,
                             version 0). Unspecified (``None`` or omitted) uses handle
                             ``0`` (legacy default stream) unless the wrapped source
                             exposes ``__cuda_array_interface__['stream']``, in which
                             case that stream is used. An explicit ``cuda_stream``
                             (including ``0``) takes precedence over the CUDA Array
                             Interface value. Inputs that are already ``Image`` keep
                             the stream they were created with; this argument is ignored
                             for those. Encode itself uses each image's associated
                             stream, not this argument. Any Image constructed from this
                             argument holds a protocol object until that Image is
                             released, so ``Image.cuda_stream`` and
                             ``__cuda_array_interface__['stream']`` stay valid. A bare
                             integer handle is not retained; the caller must keep that
                             stream alive while those values are used.

            Returns:
                List of encoded file names. If an image could not be encoded for any reason, the corresponding position in the list will contain None.

            Raises:
                ValueError: If any image is chroma-subsampled, as produced by decoding with
                            ``DecodeParams.chroma_subsampling``. Decode without it to get an image
                            that can be encoded.
            )pbdoc",
            "file_names"_a, "images"_a, "codec"_a = "", "params"_a = py::none(), "cuda_stream"_a = py::none())
        .def_property_readonly("device_id", &Encoder::getDeviceId,
            R"pbdoc(
            The CUDA device id this encoder executes on.
            )pbdoc")
        .def("__enter__", &Encoder::enter, "Enter the runtime context related to this encoder.")
        .def("__exit__", &Encoder::exit,
            "Exit the runtime context related to this encoder and releases allocated resources.",
            "exc_type"_a = py::none(),
            "exc_value"_a = py::none(), "traceback"_a = py::none());
    // clang-format on
}


} // namespace nvimgcodec
