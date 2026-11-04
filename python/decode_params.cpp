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

#include "decode_params.h"

#include <iostream>
#include <optional>

#include "error_handling.h"

namespace nvimgcodec {

DecodeParams::DecodeParams()
    : decode_params_{NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), nullptr, true}
    , color_spec_{NVIMGCODEC_COLORSPEC_SRGB}
    , allow_any_depth_{false}
    , sample_format_{NVIMGCODEC_SAMPLEFORMAT_I_AUTO_COMPONENTS}
    , chroma_subsampling_{std::nullopt}
{
}

void DecodeParams::exportToPython(py::module& m)
{
    // clang-format off
    py::class_<DecodeParams>(m, "DecodeParams", "Class to define parameters for image decoding operations. "
        "Requesting a subsampled chroma_subsampling requires a planar YUV output: sample_format set to "
        "SampleFormat.P_YUV and color_spec set to ColorSpec.SYCC.")
        .def(py::init([]() { return DecodeParams{}; }),
            "Default constructor that initializes the DecodeParams object with default settings.")
        .def(py::init([](bool apply_exif_orientation, nvimgcodecColorSpec_t color_spec, bool allow_any_depth,
                         nvimgcodecSampleFormat_t sample_format, std::optional<nvimgcodecChromaSubsampling_t> chroma_subsampling) {
            DecodeParams p;
            p.decode_params_.apply_exif_orientation = apply_exif_orientation;
            p.color_spec_ = color_spec;
            p.allow_any_depth_ = allow_any_depth;
            p.sample_format_ = sample_format;
            p.chroma_subsampling_ = chroma_subsampling;
            return p;
        }),
            "apply_exif_orientation"_a = true, "color_spec"_a = NVIMGCODEC_COLORSPEC_SRGB,
            "allow_any_depth"_a = false, "sample_format"_a = NVIMGCODEC_SAMPLEFORMAT_I_AUTO_COMPONENTS,
            "chroma_subsampling"_a = py::none(),
            R"pbdoc(
            Constructor with parameters to control the decoding process.

            Args:
                apply_exif_orientation: Boolean flag to apply EXIF orientation if available. Defaults to True.

                color_spec: Desired color specification for decoding. Defaults to sRGB.

                allow_any_depth: Boolean flag to allow any native bit depth. If not enabled, the
                dynamic range is scaled to uint8. Defaults to False.

                sample_format: Desired output sample format. Selects layout (interleaved ``I_*``
                vs planar ``P_*``) and color components; see the ``sample_format`` property below for
                the full compatibility rules. Defaults to ``I_AUTO_COMPONENTS``, which derives an interleaved
                output format from ``color_spec``.

                chroma_subsampling (ChromaSubsampling): Chroma subsampling of the decoded output. When not
                specified, the output is not subsampled: chroma is co-sampled with luma. See the
                ``chroma_subsampling`` property below for the rules and failure modes.
            )pbdoc")
        .def_property("apply_exif_orientation", &DecodeParams::getEnableOrientation, &DecodeParams::setEnableOrientation,
            R"pbdoc(
            Boolean property to enable or disable applying EXIF orientation during decoding.

            When set to True, the image is rotated and/or flipped according to its EXIF orientation
            metadata if present. Defaults to True.
            )pbdoc")
        .def_property("allow_any_depth", &DecodeParams::getAllowAnyDepth, &DecodeParams::setAllowAnyDepth,
            R"pbdoc(
            Boolean property to permit any native bit depth during decoding.

            When set to True, it allows decoding of images with their native bit depth.
            If False, the pixel values are scaled to the 8-bit range (0-255). Defaults to False.
            )pbdoc")
        .def_property("color_spec", &DecodeParams::getColorSpec, &DecodeParams::setColorSpec,
            R"pbdoc(
            Property to get or set the color specification for the decoding process.

            This determines the color space or color profile to use during decoding.
            For instance, sRGB is a common color specification. Defaults to sRGB.

            When set to ``UNCHANGED`` (or ``UNKNOWN``), the decoder derives the
            output ``color_spec`` from the source codestream..
            )pbdoc")
        .def_property("sample_format", &DecodeParams::getSampleFormat, &DecodeParams::setSampleFormat,
            R"pbdoc(
            Desired output sample format.

            Selects both the layout (interleaved ``I_*`` vs planar ``P_*``) and the color
            components produced by the decoder. The value must be compatible with
            ``color_spec`` (matching family), or be ``I_AUTO_COMPONENTS`` /
            ``P_AUTO_COMPONENTS``; otherwise a ``ValueError`` is raised when
            ``decode()`` / ``read()`` is called.

            ``I_AUTO_COMPONENTS`` / ``P_AUTO_COMPONENTS`` pin only the channel
            interleaving and let ``color_spec`` choose the components; they are the
            preferred names whenever ``color_spec`` is concrete. ``I_UNCHANGED`` /
            ``P_UNCHANGED`` are kept as aliases and behave identically.

            With ``*_AUTO_COMPONENTS`` the decoder picks the sample_format whose color
            family matches ``color_spec`` (e.g. ``color_spec=SRGB`` -> ``I_RGB``); when
            ``color_spec`` is also ``UNCHANGED`` the source's layout is preserved
            where possible, falling back to ``*_AUTO_COMPONENTS`` for non-natural source
            families (``SYCC``, ``CMYK``, ``YCCK``).
            )pbdoc")
        .def_property("chroma_subsampling", &DecodeParams::getChromaSubsampling, &DecodeParams::setChromaSubsampling,
            R"pbdoc(
            Chroma subsampling of the decoded output.

            When not specified, the output is not subsampled: chroma is co-sampled with luma.
            Only planar YUV outputs (``sample_format=P_YUV``, ``color_spec=SYCC``) may be
            subsampled, and chroma is not resampled while decoding, so the requested value must
            match the source's own chroma subsampling. Read the resulting planes through
            ``Image.planes``.

            A request that is invalid on its own terms (non-planar-YUV output, a value that does
            not match the source, or one that cannot be described) raises ``ValueError`` from
            ``decode()`` / ``read()``. A valid request that the selected backend cannot produce
            (for example a CPU-only decoder) is a decode failure like any other: the result for
            that image is ``None``.

            Combining a region of interest with a subsampled output may be declined on nvJPEG
            older than 13.2 unless the decoder is created with
            ``options=":enable_roi_fancy_upsampling=1"``.
            )pbdoc");
    // clang-format on
}


} // namespace nvimgcodec
