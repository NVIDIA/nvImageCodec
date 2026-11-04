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
#include <optional>
#include <string>
#include <vector>

#include <pybind11/stl_bind.h>

#include <ilogger.h>
#include <log.h>

#include <nvimgcodec.h>
#include "error_handling.h"
#include "image.h"
#include "imgproc/exception.h"
#include "module.h"

#include <vector>

namespace nvimgcodec {

namespace {

const nvimgcodecFrameworkDesc_t* getFrameworkDesc(nvimgcodecInstance_t instance)
{
    const nvimgcodecFrameworkDesc_t* framework = nullptr;
    CHECK_NVIMGCODEC(nvimgcodecGetFrameworkDesc(instance, &framework));
    return framework;
}

std::vector<std::string> getExtensionIds(nvimgcodecInstance_t instance)
{
    int count = 0;
    CHECK_NVIMGCODEC(nvimgcodecGetExtensions(instance, nullptr, &count));
    std::vector<nvimgcodecExtension_t> extensions(count);
    if (count != 0) {
        CHECK_NVIMGCODEC(nvimgcodecGetExtensions(instance, extensions.data(), &count));
        extensions.resize(count);
    }

    std::vector<std::string> names;
    names.reserve(extensions.size());
    for (nvimgcodecExtension_t extension : extensions) {
        const nvimgcodecExtensionDesc_t* desc = nullptr;
        CHECK_NVIMGCODEC(nvimgcodecExtensionGetExtensionDesc(extension, &desc));
        if (desc == nullptr || desc->id == nullptr) {
            throw std::runtime_error("Registered extension has no identifier");
        }
        names.emplace_back(desc->id);
    }
    return names;
}

std::vector<std::string> getCodecNames(
    nvimgcodecInstance_t instance, const std::optional<std::string>& file_extension)
{
    const nvimgcodecFrameworkDesc_t* framework = getFrameworkDesc(instance);
    const char* file_extension_filter = file_extension ? file_extension->c_str() : nullptr;
    int count = 0;
    CHECK_NVIMGCODEC(framework->getCodecNames(framework->instance, nullptr, &count, file_extension_filter));
    std::vector<char> buffer(static_cast<size_t>(count) * NVIMGCODEC_MAX_CODEC_NAME_SIZE);
    if (count != 0) {
        CHECK_NVIMGCODEC(framework->getCodecNames(framework->instance, buffer.data(), &count, file_extension_filter));
    }

    std::vector<std::string> names;
    names.reserve(count);
    for (int i = 0; i < count; ++i) {
        names.emplace_back(buffer.data() + static_cast<size_t>(i) * NVIMGCODEC_MAX_CODEC_NAME_SIZE);
    }
    return names;
}

template <typename Desc, typename Query>
std::vector<std::string> getDescriptionIds(Query&& query)
{
    int count = 0;
    CHECK_NVIMGCODEC(query(nullptr, &count));
    std::vector<const Desc*> descriptions(count);
    if (count != 0) {
        CHECK_NVIMGCODEC(query(descriptions.data(), &count));
        descriptions.resize(count);
    }

    std::vector<std::string> names;
    names.reserve(descriptions.size());
    for (const Desc* desc : descriptions) {
        if (desc == nullptr || desc->id == nullptr) {
            throw std::runtime_error("Registered plugin has no identifier");
        }
        names.emplace_back(desc->id);
    }
    return names;
}

} // namespace

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
    int verbosity = 2;
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
         "as_image",
         [instance, logger](py::handle source, CudaStreamArg cuda_stream,
                           std::optional<nvimgcodecSampleFormat_t> sample_format,
                           std::optional<nvimgcodecColorSpec_t> color_spec,
                           std::optional<int> precision) -> Image {
             return Image(instance, logger, source.ptr(), cuda_stream, sample_format, color_spec, precision);
         },
         R"pbdoc(
        Wraps an external buffer as an Image and ties the buffer lifetime to the Image.

        The buffer must be C-style contiguous, but rows may have additional padding.

        Args:
            source: Input DLPack tensor encapsulated in a PyCapsule, or any object
                    exposing ``__cuda_array_interface__``, ``__array_interface__``
                    or ``__dlpack__``/``__dlpack_device__``.

            cuda_stream: Optional CUDA stream used for any synchronization the created
                         Image needs. Accepts ``None`` (unspecified), a ``cudaStream_t``
                         handle as a Python integer, or any object implementing the CUDA
                         Stream Protocol (``__cuda_stream__()`` -> ``(version, stream_ptr)``,
                         version 0). Unspecified (``None`` or omitted) uses handle ``0``
                         (legacy default stream) unless the source exposes
                         ``__cuda_array_interface__['stream']``, in which case that stream
                         is used. If the source also exposes an associated stream via
                         ``__cuda_array_interface__['stream']``, an explicit ``cuda_stream``
                         (including ``0``) takes precedence over it, and the caller is
                         responsible for any ordering between the two.
                         The Image holds a protocol object until the Image is released,
                         so ``Image.cuda_stream`` and ``__cuda_array_interface__['stream']``
                         stay valid. An explicit handle ``0`` (legacy default stream) is
                         reported as ``1`` in ``__cuda_array_interface__['stream']``,
                         because that protocol forbids ``0``. A bare integer handle is
                         not retained; the caller must keep that stream alive while those
                         values are used. See
                         the `CUDA Stream Protocol
                         <https://nvidia.github.io/cuda-python/cuda-core/latest/interoperability.html#cuda-stream-protocol>`_.

            sample_format: (keyword-only) ``nvimgcodec.SampleFormat`` selecting the
                           output layout:

                           * ``I_*`` (concrete interleaved) -> HWC; the last axis
                             must have at least the format's channel arity (2-D
                             ``(H, W)`` only for ``I_Y``).
                           * ``P_*`` (concrete planar)      -> CHW; the leading
                             axis must have at least the format's channel arity
                             (2-D ``(H, W)`` only for ``P_Y``).
                           * ``I_UNCHANGED``                -> HWC, no arity check.
                           * ``P_UNCHANGED``                -> CHW, no arity check.
                           * ``UNKNOWN``                    -> raises ``ValueError``.

                           Extra channels beyond the format's arity are kept; too
                           few raise ``ValueError``.

                           When omitted, the value is inferred from the channel
                           count of the array using the HWC defaults:

                           * 1 channel  -> ``I_Y``
                           * 2 channels -> ``I_YA``
                           * 3 channels -> ``I_RGB``
                           * 4 channels -> ``I_RGBA``
                           * 5+ channels -> ``UNKNOWN``

            color_spec: (keyword-only) ``nvimgcodec.ColorSpec`` override. When omitted
                        the value is inferred from the channel count:

                        * 1 / 2 channels -> ``GRAY``
                        * 3 / 4 channels -> ``SRGB``
                        * 5+ channels   -> ``UNKNOWN``

            precision: (keyword-only) Optional integer giving the number of significant
                       bits per sample. Use this to describe lower-precision data
                       stored in a wider container, e.g. a 12-bit image held in a
                       uint16 buffer (``precision=12``). Accepted values are 0 (or
                       ``None``, both meaning "use the full bitdepth of the sample
                       data type") and any integer in ``1..bitdepth(dtype)``. For
                       floating-point dtypes only 0/None or the dtype's full bitdepth
                       (e.g. 32 for float32) is accepted; sub-bitdepth values are an
                       integer-image concept and have no meaningful interpretation
                       for floats.

        Returns:
            nvimgcodec.Image

        )pbdoc",
         "source"_a, "cuda_stream"_a = py::none(), py::kw_only(), "sample_format"_a = py::none(), "color_spec"_a = py::none(),
         "precision"_a = py::none(), py::keep_alive<0, 1>())
        .def(
            "as_images",
            [instance, logger](const std::vector<py::handle>& sources, CudaStreamArg cuda_stream,
                              std::optional<nvimgcodecSampleFormat_t> sample_format,
                              std::optional<nvimgcodecColorSpec_t> color_spec,
                              std::optional<int> precision) -> std::vector<py::object> {
                std::vector<py::object> py_images;
                py_images.reserve(sources.size());
                for (auto& source : sources) {
                    Image img(instance, logger, source.ptr(), cuda_stream, sample_format, color_spec, precision);
                    py::object py_img = py::cast(img);
                    py_images.push_back(py_img);
                    py::detail::keep_alive_impl(py_img, source);
                }
                return py_images;
            },
            R"pbdoc(
            Wrap a list of external buffers as Images and tie each buffer's lifetime
            to the corresponding Image. Equivalent to calling :func:`as_image` on
            each element of ``sources`` with the same ``sample_format``, ``color_spec``
            and ``precision`` overrides.

            Layout rules and sample_format / color_spec / precision inference are
            identical to :func:`as_image`. The same overrides are applied uniformly
            to every input - the elements of ``sources`` may otherwise differ in
            shape, dtype and channel count.

            Args:
                sources: List of input DLPack tensors (PyCapsules) or objects exposing
                         ``__cuda_array_interface__``, ``__array_interface__`` or
                         ``__dlpack__``/``__dlpack_device__``.

                cuda_stream: Optional CUDA stream used for any synchronization the
                             created Images need. Accepts ``None`` (unspecified), a
                             ``cudaStream_t`` handle as a Python integer, or any object
                             implementing the CUDA Stream Protocol
                             (``__cuda_stream__()`` -> ``(version, stream_ptr)``,
                             version 0). Unspecified (``None`` or omitted) uses handle
                             ``0`` unless a source exposes
                             ``__cuda_array_interface__['stream']``, in which case that
                             stream is used. If a source also exposes an associated stream
                             via ``__cuda_array_interface__['stream']``, an explicit
                             ``cuda_stream`` (including ``0``) takes precedence over it,
                             and the caller is responsible for any ordering between the two.
                             Each Image holds a protocol object until that Image is
                             released, so ``Image.cuda_stream`` and
                             ``__cuda_array_interface__['stream']`` stay valid. An
                             explicit handle ``0`` is reported as ``1`` in
                             ``__cuda_array_interface__['stream']``, because that
                             protocol forbids ``0``. A bare integer handle is not
                             retained; the caller must keep that stream alive while those
                             values are used.

                sample_format: (keyword-only) ``nvimgcodec.SampleFormat`` override
                               applied to every Image, selecting the output layout:

                               * ``I_*`` (concrete interleaved) -> HWC; the last axis
                                 must have at least the format's channel arity (2-D
                                 ``(H, W)`` only for ``I_Y``).
                               * ``P_*`` (concrete planar)      -> CHW; the leading
                                 axis must have at least the format's channel arity
                                 (2-D ``(H, W)`` only for ``P_Y``).
                               * ``I_UNCHANGED``                -> HWC, no arity check.
                               * ``P_UNCHANGED``                -> CHW, no arity check.
                               * ``UNKNOWN``                    -> raises ``ValueError``.

                               Extra channels beyond the format's arity are kept; too
                               few raise ``ValueError``.

                               When omitted, each Image's value is inferred from its
                               own channel count using the HWC defaults:

                               * 1 channel  -> ``I_Y``
                               * 2 channels -> ``I_YA``
                               * 3 channels -> ``I_RGB``
                               * 4 channels -> ``I_RGBA``
                               * 5+ channels -> ``UNKNOWN``

                color_spec: (keyword-only) ``nvimgcodec.ColorSpec`` override applied
                            to every Image. When omitted, each Image's value is
                            inferred from its own channel count:

                            * 1 / 2 channels -> ``GRAY``
                            * 3 / 4 channels -> ``SRGB``
                            * 5+ channels   -> ``UNKNOWN``

                precision: (keyword-only) Optional integer giving the number of
                           significant bits per sample, applied uniformly to every
                           Image. Use this to describe lower-precision data stored in
                           a wider container, e.g. a 12-bit image held in a uint16
                           buffer (``precision=12``). Accepted values are 0 (or
                           ``None``, both meaning "use the full bitdepth of the sample
                           data type") and any integer in ``1..bitdepth(dtype)``. For
                           floating-point dtypes only 0/None or the dtype's full
                           bitdepth (e.g. 32 for float32) is accepted; sub-bitdepth
                           values are an integer-image concept and have no meaningful
                           interpretation for floats. The value is validated against
                           each source's dtype.

            Returns:
                List of nvimgcodec.Image objects.
            )pbdoc",
            "sources"_a, "cuda_stream"_a = py::none(), py::kw_only(), "sample_format"_a = py::none(), "color_spec"_a = py::none(),
            "precision"_a = py::none())
        .def(
            "from_dlpack",
            [instance, logger](py::handle source, CudaStreamArg cuda_stream,
                               std::optional<nvimgcodecSampleFormat_t> sample_format,
                               std::optional<nvimgcodecColorSpec_t> color_spec,
                               std::optional<int> precision) -> Image {
                return Image(instance, logger, source.ptr(), cuda_stream, sample_format, color_spec, precision);
            },
            R"pbdoc(
            Zero-copy conversion from a DLPack tensor to an Image.

            The DLPack source must be a 2- or 3-dimensional tensor on CPU or
            CUDA-accessible memory (device, CUDA-host, or CUDA-managed). Layout
            selection and sample_format / color_spec inference are identical
            to :func:`as_image`; see that function for the full rules.

            Args:
                source: Input DLPack tensor encapsulated in a PyCapsule, or any object
                        exposing ``__dlpack__`` and ``__dlpack_device__`` methods.

                cuda_stream: Optional CUDA stream used for any synchronization the
                             created Image needs. Accepts ``None`` (unspecified), a
                             ``cudaStream_t`` handle as a Python integer, or any object
                             implementing the CUDA Stream Protocol
                             (``__cuda_stream__()`` -> ``(version, stream_ptr)``,
                             version 0). Unspecified (``None`` or omitted) uses handle
                             ``0`` unless the source exposes
                             ``__cuda_array_interface__['stream']``, in which case that
                             stream is used. If the source also exposes an associated
                             stream via ``__cuda_array_interface__['stream']``, an explicit
                             ``cuda_stream`` (including ``0``) takes precedence over it,
                             and the caller is responsible for any ordering between the two.
                             The Image holds a protocol object until the Image is
                             released, so ``Image.cuda_stream`` and
                             ``__cuda_array_interface__['stream']`` stay valid. An
                             explicit handle ``0`` is reported as ``1`` in
                             ``__cuda_array_interface__['stream']``, because that
                             protocol forbids ``0``. A bare integer handle is not
                             retained; the caller must keep that stream alive while those
                             values are used.

                sample_format: (keyword-only) ``nvimgcodec.SampleFormat`` selecting the
                               output layout:

                               * ``I_*`` (concrete interleaved) -> HWC; the last axis
                                 must have at least the format's channel arity.
                               * ``P_*`` (concrete planar)      -> CHW; the leading
                                 axis must have at least the format's channel arity.
                               * ``I_UNCHANGED``                -> HWC, no arity check.
                               * ``P_UNCHANGED``                -> CHW, no arity check.
                               * ``UNKNOWN``                    -> raises ``ValueError``.

                               Extra channels beyond the format's arity are kept; too
                               few raise ``ValueError``.

                               When omitted, the value is inferred from the channel
                               count of the tensor using the HWC defaults:

                               * 1 channel  -> ``I_Y``
                               * 2 channels -> ``I_YA``
                               * 3 channels -> ``I_RGB``
                               * 4 channels -> ``I_RGBA``
                               * 5+ channels -> ``UNKNOWN``

                color_spec: (keyword-only) ``nvimgcodec.ColorSpec`` override. When omitted
                            the value is inferred from the channel count:

                            * 1 / 2 channels -> ``GRAY``
                            * 3 / 4 channels -> ``SRGB``
                            * 5+ channels   -> ``UNKNOWN``

                precision: (keyword-only) Optional integer giving the number of significant
                           bits per sample. Use this to describe lower-precision data
                           stored in a wider container, e.g. a 12-bit image held in a
                           uint16 buffer (``precision=12``). Accepted values are 0 (or
                           ``None``, both meaning "use the full bitdepth of the sample
                           data type") and any integer in ``1..bitdepth(dtype)``. See
                           :func:`as_image` for details.

            Returns:
                nvimgcodec.Image

            )pbdoc",
            "source"_a, "cuda_stream"_a = py::none(), py::kw_only(), "sample_format"_a = py::none(), "color_spec"_a = py::none(),
            "precision"_a = py::none(), py::keep_alive<0, 1>());

    m.def(
        "get_file_extensions",
        [instance](const std::optional<std::string>& codec_name) {
            const nvimgcodecFrameworkDesc_t* framework = getFrameworkDesc(instance);
            const char* codec_filter = codec_name ? codec_name->c_str() : nullptr;
            int count = 0;
            nvimgcodecStatus_t status =
                framework->getFileExtensions(framework->instance, nullptr, &count, codec_filter);
            if (status != NVIMGCODEC_STATUS_SUCCESS) {
                throw Exception(INVALID_PARAMETER, "Could not list file extensions. Code: " + std::to_string(status));
            }
            std::vector<std::string> names;
            if (count == 0) {
                return names;
            }
            std::vector<char> buffer(static_cast<size_t>(count) * NVIMGCODEC_MAX_FILE_EXTENSION_SIZE, '\0');
            status = framework->getFileExtensions(framework->instance, buffer.data(), &count, codec_filter);
            if (status != NVIMGCODEC_STATUS_SUCCESS) {
                throw Exception(INVALID_PARAMETER, "Could not list file extensions. Code: " + std::to_string(status));
            }
            names.reserve(count);
            for (int i = 0; i < count; ++i) {
                names.emplace_back(buffer.data() + static_cast<size_t>(i) * NVIMGCODEC_MAX_FILE_EXTENSION_SIZE);
            }
            return names;
        },
        R"pbdoc(
        Return registered file extensions, optionally filtered by codec name.

        Without a filter, every registered file extension is returned once. Pass an
        extension to :func:`get_codec_names` to retrieve its codec mappings and default codec.

        Args:
            codec_name: Optional codec-name filter (e.g. ``"jpeg"``).

        Returns:
            List of file extension strings including the leading period (e.g. ``[".jpg", ".jpeg"]``).
            Empty if the filter names an unknown codec.
        )pbdoc",
        "codec_name"_a = py::none());

    m.def(
        "register_file_extension",
        [instance](const std::string& codec_name, const std::string& file_extension, bool is_default) {
            const nvimgcodecFrameworkDesc_t* framework = getFrameworkDesc(instance);
            nvimgcodecStatus_t status = framework->registerFileExtension(
                framework->instance, codec_name.c_str(), file_extension.c_str(), is_default ? 1 : 0);
            if (status == NVIMGCODEC_STATUS_INVALID_PARAMETER) {
                throw py::value_error("Could not register file extension for codec");
            }
            if (status == NVIMGCODEC_STATUS_RESOURCE_IN_USE) {
                throw std::runtime_error(
                    "Cannot register a file extension while decoders, encoders, or code streams are active");
            }
            if (status != NVIMGCODEC_STATUS_SUCCESS) {
                throw std::runtime_error("Could not register file extension for codec. Code: " + std::to_string(status));
            }
        },
        R"pbdoc(
        Register a file extension for a codec.

        Registrations are reference-counted. Every successful call must be balanced by one
        :func:`unregister_file_extension` call. Registering an unknown codec name creates a
        codec entry that is returned by :func:`get_codec_names`, even if it has no parser,
        decoder, or encoder plugins.

        Args:
            codec_name: Codec name (e.g. ``"jpeg"``).
            file_extension: File extension with or without a leading period (e.g. ``".jpe"``).
            is_default: Whether this codec should become the default for the file extension.

        Raises:
            ValueError: If the codec name or file extension is invalid.
            RuntimeError: If a decoder, encoder, or code stream is active, or registration fails
                          for another reason.
        )pbdoc",
        "codec_name"_a, "file_extension"_a, py::kw_only(), "is_default"_a = false);

    m.def(
        "unregister_file_extension",
        [instance](const std::string& codec_name, const std::string& file_extension) {
            const nvimgcodecFrameworkDesc_t* framework = getFrameworkDesc(instance);
            nvimgcodecStatus_t status =
                framework->unregisterFileExtension(framework->instance, codec_name.c_str(), file_extension.c_str());
            if (status == NVIMGCODEC_STATUS_INVALID_PARAMETER) {
                throw py::value_error("Could not unregister file extension from codec");
            }
            if (status == NVIMGCODEC_STATUS_RESOURCE_IN_USE) {
                throw std::runtime_error(
                    "Cannot unregister a file extension while decoders, encoders, or code streams are active");
            }
            if (status != NVIMGCODEC_STATUS_SUCCESS) {
                throw std::runtime_error("Could not unregister file extension from codec. Code: " + std::to_string(status));
            }
        },
        R"pbdoc(
        Remove one file-extension registration from a codec.

        Call this once for each successful :func:`register_file_extension` call made by the
        application. The mapping remains registered while another owner still holds a
        reference to the same codec and file-extension pair.

        Args:
            codec_name: Codec name used during registration.
            file_extension: File extension used during registration.

        Raises:
            ValueError: If the codec name or file extension is invalid.
            RuntimeError: If a decoder, encoder, or code stream is active, or unregistration fails
                          for another reason.
        )pbdoc",
        "codec_name"_a, "file_extension"_a);

    m.def(
        "get_extension_ids", [instance] { return getExtensionIds(instance); }, R"pbdoc(
        Return the identifiers of registered nvImageCodec extensions.

        Extension identifiers name loaded plugin containers; they are not filename extensions.
        )pbdoc");

    m.def(
        "get_codec_names",
        [instance](const std::optional<std::string>& file_extension) { return getCodecNames(instance, file_extension); },
        R"pbdoc(
        Return registered codec names, optionally filtered by file extension.

        Without a filter, codecs with at least one registered parser, decoder, encoder, or
        file-extension mapping are included. A returned codec may therefore have no executable
        plugin identifiers.

        With ``file_extension`` set, only mapped codecs are returned and the default codec is
        first. This is the codec used by operations such as
        ``Encoder.encode(..., codec=".jpg")`` and ``Encoder.write("out.jpg", ...)``.
        The mapping is not used when reading encoded input: ``CodeStream`` parsers inspect the
        bitstream contents, independently of the input filename extension.

        Args:
            file_extension: Optional file-extension filter, with or without a leading period.
        )pbdoc",
        "file_extension"_a = py::none());

    m.def(
        "get_decoder_ids",
        [instance](const std::optional<std::string>& codec_name, const std::optional<nvimgcodecBackendKind_t>& backend_kind) {
            const nvimgcodecFrameworkDesc_t* framework = getFrameworkDesc(instance);
            const char* codec_filter = codec_name ? codec_name->c_str() : nullptr;
            nvimgcodecBackendKind_t backend_filter = backend_kind.value_or(NVIMGCODEC_BACKEND_KIND_ANY);
            return getDescriptionIds<nvimgcodecDecoderDesc_t>([&](const nvimgcodecDecoderDesc_t** descriptions, int* count) {
                return framework->getDecoderDescs(framework->instance, descriptions, count, codec_filter, backend_filter);
            });
        },
        R"pbdoc(
        Return registered decoder identifiers in effective priority order.

        Args:
            codec_name: Optional codec-name filter, such as ``"jpeg"``.
            backend_kind: Optional :class:`BackendKind` filter.

        A listed decoder is registered for the selected backend kind but may still require
        hardware or runtime dependencies that are unavailable on the current system.
        )pbdoc",
        "codec_name"_a = py::none(), "backend_kind"_a = py::none());

    m.def(
        "get_encoder_ids",
        [instance](const std::optional<std::string>& codec_name, const std::optional<nvimgcodecBackendKind_t>& backend_kind) {
            const nvimgcodecFrameworkDesc_t* framework = getFrameworkDesc(instance);
            const char* codec_filter = codec_name ? codec_name->c_str() : nullptr;
            nvimgcodecBackendKind_t backend_filter = backend_kind.value_or(NVIMGCODEC_BACKEND_KIND_ANY);
            return getDescriptionIds<nvimgcodecEncoderDesc_t>([&](const nvimgcodecEncoderDesc_t** descriptions, int* count) {
                return framework->getEncoderDescs(framework->instance, descriptions, count, codec_filter, backend_filter);
            });
        },
        R"pbdoc(
        Return registered encoder identifiers in effective priority order.

        Args:
            codec_name: Optional codec-name filter, such as ``"jpeg"``.
            backend_kind: Optional :class:`BackendKind` filter.

        A listed encoder is registered for the selected backend kind but may still require
        hardware or runtime dependencies that are unavailable on the current system.
        )pbdoc",
        "codec_name"_a = py::none(), "backend_kind"_a = py::none());

    m.def(
        "get_parser_ids",
        [instance](const std::optional<std::string>& codec_name) {
            const nvimgcodecFrameworkDesc_t* framework = getFrameworkDesc(instance);
            const char* codec_filter = codec_name ? codec_name->c_str() : nullptr;
            return getDescriptionIds<nvimgcodecParserDesc_t>([&](const nvimgcodecParserDesc_t** descriptions, int* count) {
                return framework->getParserDescs(framework->instance, descriptions, count, codec_filter);
            });
        },
        R"pbdoc(
        Return registered parser identifiers in effective priority order.

        Args:
            codec_name: Optional codec-name filter, such as ``"jpeg"``.
        )pbdoc",
        "codec_name"_a = py::none());
}

} // namespace nvimgcodec
