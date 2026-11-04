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

#pragma once

#include <cassert>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

#include <nvimgcodec.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace nvimgcodec {

inline size_t sample_type_to_bytes_per_element(nvimgcodecSampleDataType_t sample_type)
{
    //Shift by 8 since 8..15 bits represents type bitdepth,  then shift by 3 to convert to # bytes
    return static_cast<unsigned int>(sample_type) >> (8 + 3);
}

// Resolve a plane_info.precision value to its user-visible bitcount: pass-through when
// non-zero, otherwise the sample type's full bitdepth (the C-layer's 0 sentinel).
inline int resolve_precision(uint8_t precision, nvimgcodecSampleDataType_t sample_type)
{
    return precision != 0 ? precision : static_cast<int>(sample_type_to_bytes_per_element(sample_type) * 8);
}

inline bool is_sample_format_interleaved(nvimgcodecSampleFormat_t sample_format)
{
    //First bit of sample format says if this is interleaved or not
    return static_cast<int>(sample_format) % 2 == 0 ;
}

// Flip a concrete sample_format between interleaved (I_*) and planar (P_*)
// without changing its color family (e.g. with_layout(P_BGR, true) -> I_BGR).
// P_X and I_X are always adjacent in the enum (P_X = N (odd), I_X = N+1 (even)),
// so flipping is just +/-1. UNKNOWN is returned unchanged.
inline nvimgcodecSampleFormat_t with_layout(nvimgcodecSampleFormat_t sf, bool is_interleaved)
{
    if (sf == NVIMGCODEC_SAMPLEFORMAT_UNKNOWN)
        return sf;
    const bool currently_interleaved = is_sample_format_interleaved(sf);
    if (currently_interleaved == is_interleaved)
        return sf;
    const int v = static_cast<int>(sf);
    return static_cast<nvimgcodecSampleFormat_t>(is_interleaved ? v + 1 : v - 1);
}

inline std::string format_str_from_type(nvimgcodecSampleDataType_t type)
{
    switch (type) {
    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT8:
        return "|i1";
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8:
        return "|u1";
    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT16:
        return "<i2";
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16:
        return "<u2";
    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT32:
        return "<i4";
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT32:
        return "<u4";
    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT64:
        return "<i8";
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT64:
        return "<u8";
    case NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT16:
        return "<f2";
    case NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32:
        return "<f4";
    case NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT64:
        return "<f8";
    default:
        break;
    }
    return "";
}

inline nvimgcodecSampleDataType_t type_from_format_str(const std::string& typestr)
{
    pybind11::ssize_t itemsize = py::dtype(typestr).itemsize();
    if (itemsize == 1) {
        if (py::dtype(typestr).kind() == 'i')
            return NVIMGCODEC_SAMPLE_DATA_TYPE_INT8;
        if (py::dtype(typestr).kind() == 'u')
            return NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
    } else if (itemsize == 2) {
        if (py::dtype(typestr).kind() == 'i')
            return NVIMGCODEC_SAMPLE_DATA_TYPE_INT16;
        if (py::dtype(typestr).kind() == 'u')
            return NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16;
        if (py::dtype(typestr).kind() == 'f')
            return NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT16;
    } else if (itemsize == 4) {
        if (py::dtype(typestr).kind() == 'i')
            return NVIMGCODEC_SAMPLE_DATA_TYPE_INT32;
        if (py::dtype(typestr).kind() == 'u')
            return NVIMGCODEC_SAMPLE_DATA_TYPE_UINT32;
        if (py::dtype(typestr).kind() == 'f')
            return NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32;
    } else if (itemsize == 8) {
        if (py::dtype(typestr).kind() == 'i')
            return NVIMGCODEC_SAMPLE_DATA_TYPE_INT64;
        if (py::dtype(typestr).kind() == 'u')
            return NVIMGCODEC_SAMPLE_DATA_TYPE_UINT64;
        if (py::dtype(typestr).kind() == 'f')
            return NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT64;
    }
    return NVIMGCODEC_SAMPLE_DATA_TYPE_UNKNOWN;
}

inline std::string dtype_to_str(const py::dtype& t)
{
    if (t.itemsize() == 1) {
        if (t.kind() == 'i')
            return "int8";
        if (t.kind() == 'u')
            return "uint8";
    } else if (t.itemsize() == 2) {
        if (t.kind() == 'i')
            return "int16";
        if (t.kind() == 'u')
            return "uint16";
        if (t.kind() == 'f')
            return "float16";
    } else if (t.itemsize() == 4) {
        if (t.kind() == 'i')
            return "int32";
        if (t.kind() == 'u')
            return "uint32";
        if (t.kind() == 'f')
            return "float32";
    } else if (t.itemsize() == 8) {
        if (t.kind() == 'i')
            return "int64";
        if (t.kind() == 'u')
            return "uint64";
        if (t.kind() == 'f')
            return "float64";
    }
    return "unknown type";
}

// True when sample_format is one of the layout-only "pass-through" variants
// (I_UNCHANGED / P_UNCHANGED). These don't carry a color family, so callers
// that need to know "did the user / source request an UNCHANGED output?" use
// this predicate rather than the color_spec returned by
// sample_format_color_family.
inline bool is_sample_format_unchanged(nvimgcodecSampleFormat_t sf)
{
    return sf == NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED ||
           sf == NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED;
}

// Color family implied by a sample_format, expressed as a nvimgcodecColorSpec_t.
// Returns UNKNOWN for *_UNCHANGED and UNKNOWN sample formats (use
// is_sample_format_unchanged to disambiguate when needed).
inline nvimgcodecColorSpec_t sample_format_color_family(nvimgcodecSampleFormat_t sf)
{
    switch (sf) {
    case NVIMGCODEC_SAMPLEFORMAT_P_Y:
    case NVIMGCODEC_SAMPLEFORMAT_I_Y:
    case NVIMGCODEC_SAMPLEFORMAT_P_YA:
    case NVIMGCODEC_SAMPLEFORMAT_I_YA:
        return NVIMGCODEC_COLORSPEC_GRAY;
    case NVIMGCODEC_SAMPLEFORMAT_P_RGB:
    case NVIMGCODEC_SAMPLEFORMAT_I_RGB:
    case NVIMGCODEC_SAMPLEFORMAT_P_BGR:
    case NVIMGCODEC_SAMPLEFORMAT_I_BGR:
    case NVIMGCODEC_SAMPLEFORMAT_P_RGBA:
    case NVIMGCODEC_SAMPLEFORMAT_I_RGBA:
        return NVIMGCODEC_COLORSPEC_SRGB;
    case NVIMGCODEC_SAMPLEFORMAT_P_YUV:
    case NVIMGCODEC_SAMPLEFORMAT_I_YUV:
        // NVIMGCODEC_SAMPLEFORMAT_{P,I}_YCC are aliases of the YUV variants.
        return NVIMGCODEC_COLORSPEC_SYCC;
    case NVIMGCODEC_SAMPLEFORMAT_P_CMYK:
    case NVIMGCODEC_SAMPLEFORMAT_I_CMYK:
        return NVIMGCODEC_COLORSPEC_CMYK;
    case NVIMGCODEC_SAMPLEFORMAT_P_YCCK:
    case NVIMGCODEC_SAMPLEFORMAT_I_YCCK:
        return NVIMGCODEC_COLORSPEC_YCCK;
    default:
        // UNKNOWN / *_UNCHANGED / any future enum addition.
        return NVIMGCODEC_COLORSPEC_UNKNOWN;
    }
}

// Minimum/expected channel count for a concrete sample format. Returns -1 for
// UNKNOWN/UNCHANGED (any count is acceptable).
inline int num_components_for_sample_format(nvimgcodecSampleFormat_t sf)
{
    switch (sf) {
    case NVIMGCODEC_SAMPLEFORMAT_P_Y:
    case NVIMGCODEC_SAMPLEFORMAT_I_Y:
        return 1;
    case NVIMGCODEC_SAMPLEFORMAT_P_YA:
    case NVIMGCODEC_SAMPLEFORMAT_I_YA:
        return 2;
    case NVIMGCODEC_SAMPLEFORMAT_P_RGB:
    case NVIMGCODEC_SAMPLEFORMAT_I_RGB:
    case NVIMGCODEC_SAMPLEFORMAT_P_BGR:
    case NVIMGCODEC_SAMPLEFORMAT_I_BGR:
    case NVIMGCODEC_SAMPLEFORMAT_P_YUV:
    case NVIMGCODEC_SAMPLEFORMAT_I_YUV:
        return 3;
    case NVIMGCODEC_SAMPLEFORMAT_P_RGBA:
    case NVIMGCODEC_SAMPLEFORMAT_I_RGBA:
    case NVIMGCODEC_SAMPLEFORMAT_P_CMYK:
    case NVIMGCODEC_SAMPLEFORMAT_I_CMYK:
    case NVIMGCODEC_SAMPLEFORMAT_P_YCCK:
    case NVIMGCODEC_SAMPLEFORMAT_I_YCCK:
        return 4;
    default:
        return -1;
    }
}

inline std::string sample_format_name(nvimgcodecSampleFormat_t sf)
{
    switch (sf) {
    case NVIMGCODEC_SAMPLEFORMAT_UNKNOWN:     return "UNKNOWN";
    case NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED: return "P_AUTO_COMPONENTS";
    case NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED: return "I_AUTO_COMPONENTS";
    case NVIMGCODEC_SAMPLEFORMAT_P_Y:         return "P_Y";
    case NVIMGCODEC_SAMPLEFORMAT_I_Y:         return "I_Y";
    case NVIMGCODEC_SAMPLEFORMAT_P_YA:        return "P_YA";
    case NVIMGCODEC_SAMPLEFORMAT_I_YA:        return "I_YA";
    case NVIMGCODEC_SAMPLEFORMAT_P_RGB:       return "P_RGB";
    case NVIMGCODEC_SAMPLEFORMAT_I_RGB:       return "I_RGB";
    case NVIMGCODEC_SAMPLEFORMAT_P_BGR:       return "P_BGR";
    case NVIMGCODEC_SAMPLEFORMAT_I_BGR:       return "I_BGR";
    case NVIMGCODEC_SAMPLEFORMAT_P_YUV:       return "P_YUV";
    case NVIMGCODEC_SAMPLEFORMAT_I_YUV:       return "I_YUV";
    case NVIMGCODEC_SAMPLEFORMAT_P_RGBA:      return "P_RGBA";
    case NVIMGCODEC_SAMPLEFORMAT_I_RGBA:      return "I_RGBA";
    case NVIMGCODEC_SAMPLEFORMAT_P_YCCK:      return "P_YCCK";
    case NVIMGCODEC_SAMPLEFORMAT_I_YCCK:      return "I_YCCK";
    case NVIMGCODEC_SAMPLEFORMAT_P_CMYK:      return "P_CMYK";
    case NVIMGCODEC_SAMPLEFORMAT_I_CMYK:      return "I_CMYK";
    default:                                  return "UNSUPPORTED";
    }
}

inline std::string color_spec_name(nvimgcodecColorSpec_t cs)
{
    switch (cs) {
    case NVIMGCODEC_COLORSPEC_UNKNOWN:     return "UNKNOWN";  // also UNCHANGED
    case NVIMGCODEC_COLORSPEC_SRGB:        return "SRGB";
    case NVIMGCODEC_COLORSPEC_GRAY:        return "GRAY";
    case NVIMGCODEC_COLORSPEC_SYCC:        return "SYCC";
    case NVIMGCODEC_COLORSPEC_CMYK:        return "CMYK";
    case NVIMGCODEC_COLORSPEC_YCCK:        return "YCCK";
    case NVIMGCODEC_COLORSPEC_PALETTE:     return "PALETTE";
    case NVIMGCODEC_COLORSPEC_ICC_PROFILE: return "ICC_PROFILE";
    default:                               return "UNSUPPORTED";
    }
}

// ---------------------------------------------------------------------------
// as_image input-wrapping helpers (shared by the array-interface and DLPack
// paths, so layout selection and the sample_format / color_spec defaults live
// in one place).
// ---------------------------------------------------------------------------

// HWC-vs-CHW decision and the resulting plane/channel counts for an as_image
// wrap of a host/device buffer described by `shape` (2 or 3 dims).
struct AsImageLayout
{
    bool     is_interleaved;  // true -> HWC (1 plane), false -> CHW (planar)
    uint32_t num_planes;
    uint32_t height;
    uint32_t width;
    uint32_t num_channels;    // channels per plane: last axis for HWC, 1 for CHW
};

// Resolve the layout implied by the caller's sample_format and the input shape:
//   (a) Concrete planar sample_format -> CHW with arity validation against the
//       leading axis (accepts >= arity; 2-D shape only for arity 1).
//   (b) P_UNCHANGED                  -> CHW, plane count from shape[0] (2-D -> 1).
//   (c) Concrete interleaved          -> HWC with arity validation against the
//       last axis (or 1 for 2-D).
//   (d) I_UNCHANGED / no override     -> HWC, channels from the last axis.
// sample_format=UNKNOWN is rejected. Throws std::invalid_argument on UNKNOWN or
// an arity mismatch (the message quotes the offending shape).
inline AsImageLayout resolveAsImageLayout(
    const std::vector<long>& shape, std::optional<nvimgcodecSampleFormat_t> sample_format)
{
    const size_t ndim = shape.size();
    assert(ndim == 2 || ndim == 3);
    auto format_shape = [&]() {
        std::string s = "(";
        for (size_t i = 0; i < ndim; ++i) {
            if (i != 0) s += ", ";
            s += std::to_string(shape[i]);
        }
        return s + ")";
    };

    bool is_interleaved = true;
    int planar_num_planes = -1;
    if (sample_format.has_value()) {
        const nvimgcodecSampleFormat_t sf = *sample_format;
        if (sf == NVIMGCODEC_SAMPLEFORMAT_UNKNOWN) {
            throw std::invalid_argument(
                "sample_format=UNKNOWN is not a valid as_image argument. Use a "
                "concrete I_*/P_* format, I_UNCHANGED / P_UNCHANGED, or omit the "
                "argument to use the channel-count default.");
        }
        if (sf == NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED) {
            is_interleaved = false;
            planar_num_planes = ndim == 3 ? shape[0] : 1;
        } else if (!is_sample_format_interleaved(sf) && !is_sample_format_unchanged(sf)) {
            const int arity = num_components_for_sample_format(sf);
            const bool shape_fits_3d = ndim == 3 && shape[0] >= arity;
            const bool shape_fits_2d = ndim == 2 && arity == 1;
            if (arity > 0 && (shape_fits_3d || shape_fits_2d)) {
                is_interleaved = false;
                planar_num_planes = shape_fits_3d ? shape[0] : 1;
            } else {
                std::string expected = "at least (" + std::to_string(arity) + ", H, W)";
                if (arity == 1)
                    expected += " or (H, W)";
                throw std::invalid_argument(
                    std::string("Planar sample_format ") + sample_format_name(sf) +
                    " requires a CHW-shaped input " + expected + "; got shape " + format_shape() + ".");
            }
        } else if (is_sample_format_interleaved(sf) && !is_sample_format_unchanged(sf)) {
            const int arity = num_components_for_sample_format(sf);
            const int input_channels = ndim == 3 ? shape[2] : 1;
            if (arity > 0 && input_channels < arity) {
                std::string expected = "at least (H, W, " + std::to_string(arity) + ")";
                if (arity == 1)
                    expected += " or (H, W)";
                throw std::invalid_argument(
                    std::string("Interleaved sample_format ") + sample_format_name(sf) +
                    " requires an HWC-shaped input " + expected + "; got shape " + format_shape() + ".");
            }
        }
    }

    AsImageLayout out{};
    out.is_interleaved = is_interleaved;
    if (is_interleaved) {
        out.num_planes = 1;
        out.height = shape[0];
        out.width = shape[1];
        out.num_channels = ndim == 3 ? shape[2] : 1;
    } else {
        out.num_planes = static_cast<uint32_t>(planar_num_planes);
        out.height = ndim == 3 ? shape[1] : shape[0];
        out.width = ndim == 3 ? shape[2] : shape[1];
        out.num_channels = 1;
    }
    return out;
}

// Resolved sample_format / color_spec / chroma labels for an as_image wrap.
struct AsImageLabels
{
    nvimgcodecSampleFormat_t      sample_format;
    nvimgcodecColorSpec_t         color_spec;
    nvimgcodecChromaSubsampling_t chroma_subsampling;
};

// Assign the sample_format / color_spec / chroma labels, falling back to the
// channel-count defaults when the caller did not override them. `num_channels`
// is the effective component count (last axis for HWC, plane count for CHW).
inline AsImageLabels resolveAsImageLabels(int num_channels,
    std::optional<nvimgcodecSampleFormat_t> sample_format, std::optional<nvimgcodecColorSpec_t> color_spec)
{
    if (num_channels <= 0)
        throw std::runtime_error("Unexpected number of channels. At least 1 channel is expected.");

    AsImageLabels r{};
    switch (num_channels) {
    case 1:
        r.color_spec = color_spec.value_or(NVIMGCODEC_COLORSPEC_GRAY);
        r.chroma_subsampling = NVIMGCODEC_SAMPLING_GRAY;
        r.sample_format = sample_format.value_or(NVIMGCODEC_SAMPLEFORMAT_I_Y);
        break;
    case 2:
        r.color_spec = color_spec.value_or(NVIMGCODEC_COLORSPEC_GRAY);
        r.chroma_subsampling = NVIMGCODEC_SAMPLING_GRAY;
        r.sample_format = sample_format.value_or(NVIMGCODEC_SAMPLEFORMAT_I_YA);
        break;
    case 3:
        r.color_spec = color_spec.value_or(NVIMGCODEC_COLORSPEC_SRGB);
        r.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        r.sample_format = sample_format.value_or(NVIMGCODEC_SAMPLEFORMAT_I_RGB);
        break;
    case 4:
        r.color_spec = color_spec.value_or(NVIMGCODEC_COLORSPEC_SRGB);
        r.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        r.sample_format = sample_format.value_or(NVIMGCODEC_SAMPLEFORMAT_I_RGBA);
        break;
    default:
        r.color_spec = color_spec.value_or(NVIMGCODEC_COLORSPEC_UNKNOWN);
        r.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        r.sample_format = sample_format.value_or(NVIMGCODEC_SAMPLEFORMAT_UNKNOWN);
        break;
    }
    return r;
}

struct ResolvedFormat
{
    nvimgcodecSampleFormat_t sample_format;   // concrete I_*/P_* (never UNKNOWN on success)
    nvimgcodecColorSpec_t    color_spec;      // concrete (or UNKNOWN if source provides nothing for an UNCHANGED request)
    bool                     is_interleaved;  // true for I_*, false for P_*
    int                      num_components;  // resolved component count
};

// Resolve the output format that a decoder should produce, given the user's
// requests (typically from DecodeParams) and the source's detected metadata
// (typically from the parsed code stream). Throws std::invalid_argument when
// the user's requested sample_format and color_spec disagree.
//
// Inputs:
//   requested_sample_format: NVIMGCODEC_SAMPLEFORMAT_*_UNCHANGED means "derive".
//   requested_color_spec:    NVIMGCODEC_COLORSPEC_UNKNOWN/UNCHANGED means "derive".
//   source_sample_format:    sample_format reported by the parser.
//   source_color_spec:       color_spec reported by the parser.
//   source_num_components:   max(num_planes, plane[0].num_channels) on the
//                            decode path; must be a positive integer.
inline ResolvedFormat resolveOutputFormat(
    nvimgcodecSampleFormat_t requested_sample_format,
    nvimgcodecColorSpec_t    requested_color_spec,
    nvimgcodecSampleFormat_t source_sample_format,
    nvimgcodecColorSpec_t    source_color_spec,
    int                      source_num_components)
{
    if (requested_color_spec < NVIMGCODEC_COLORSPEC_UNKNOWN || requested_color_spec >= NVIMGCODEC_COLORSPEC_PALETTE) {
        // Reject UNSUPPORTED (-1), PALETTE, ICC_PROFILE and any out-of-range value;
        // UNKNOWN/UNCHANGED (0) stays valid as the "derive from source" request.
        throw std::invalid_argument("unsupported color_spec " + color_spec_name(requested_color_spec));
    }
    if (requested_sample_format < NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED || requested_sample_format > NVIMGCODEC_SAMPLEFORMAT_I_CMYK) {
        // Reject UNKNOWN (0), UNSUPPORTED (-1) and any out-of-range value; only
        // the concrete I_*/P_* formats and the *_UNCHANGED pass-throughs are valid.
        throw std::invalid_argument("unsupported sample_format " + sample_format_name(requested_sample_format));
    }
    const nvimgcodecColorSpec_t req_sf_family = sample_format_color_family(requested_sample_format);
    const bool sf_is_concrete = req_sf_family != NVIMGCODEC_COLORSPEC_UNKNOWN;
    const bool cs_is_concrete = requested_color_spec != NVIMGCODEC_COLORSPEC_UNKNOWN;
    if (sf_is_concrete && cs_is_concrete && req_sf_family != requested_color_spec) {
        throw std::invalid_argument(
            std::string("sample_format ") + sample_format_name(requested_sample_format) +
            " is not compatible with color_spec " + color_spec_name(requested_color_spec) +
            ". Use a matching sample_format for the chosen color_spec, or set " +
            "sample_format to UNKNOWN / I_UNCHANGED / P_UNCHANGED.");
    }

    ResolvedFormat r;
    r.is_interleaved = is_sample_format_interleaved(requested_sample_format);

    if (sf_is_concrete) {
        // color_spec is UNKOWN or matching sample_format
        r.color_spec = req_sf_family;
        r.sample_format = requested_sample_format;
        r.num_components = num_components_for_sample_format(requested_sample_format);
    } else if (cs_is_concrete) {
        // requested sample_format is UNKNOWN (legacy) or UNCHANGED
        r.color_spec = requested_color_spec;
        switch (r.color_spec) {
        case NVIMGCODEC_COLORSPEC_GRAY:
            r.sample_format = with_layout(NVIMGCODEC_SAMPLEFORMAT_I_Y, r.is_interleaved); break;
        case NVIMGCODEC_COLORSPEC_SRGB:
            r.sample_format = with_layout(NVIMGCODEC_SAMPLEFORMAT_I_RGB, r.is_interleaved); break;
        case NVIMGCODEC_COLORSPEC_SYCC:
            r.sample_format = with_layout(NVIMGCODEC_SAMPLEFORMAT_I_YUV, r.is_interleaved); break;
        case NVIMGCODEC_COLORSPEC_CMYK:
            r.sample_format = with_layout(NVIMGCODEC_SAMPLEFORMAT_I_CMYK, r.is_interleaved); break;
        case NVIMGCODEC_COLORSPEC_YCCK:
            r.sample_format = with_layout(NVIMGCODEC_SAMPLEFORMAT_I_YCCK, r.is_interleaved); break;
        default:
            r.sample_format = with_layout(NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED, r.is_interleaved); break;
        }
        r.num_components = num_components_for_sample_format(r.sample_format);
    } else {
        // requested sample_format and color_spec are UNCHANGED/UNKNOWN
        if (source_color_spec == NVIMGCODEC_COLORSPEC_GRAY || source_color_spec == NVIMGCODEC_COLORSPEC_SRGB) {
            r.color_spec = source_color_spec;
            assert(
                source_sample_format != NVIMGCODEC_SAMPLEFORMAT_UNKNOWN &&
                source_sample_format != NVIMGCODEC_SAMPLEFORMAT_UNSUPPORTED
            );
            r.sample_format = with_layout(source_sample_format, r.is_interleaved);
        } else {
            // TODO: source color spaces other than GRAY and SRGB are not fully
            // supported yet. Force the output color_spec to GRAY (1-2 channels) or
            // SRGB (3+ channels) and leave sample_format as I_UNCHANGED /
            // P_UNCHANGED to preserve the legacy decode behavior. Downstream
            // codecs are inconsistent about which of the two fields they consult,
            // so both are kept aligned with the prior path until color_spec and
            // sample_format are jointly reworked.
            r.color_spec = source_num_components <= 2 ? NVIMGCODEC_COLORSPEC_GRAY : NVIMGCODEC_COLORSPEC_SRGB;
            r.sample_format = with_layout(NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED, r.is_interleaved);
        }
        r.num_components = source_num_components;
    }
    assert(r.num_components > 0);
    return r;
}

// Horizontal and vertical divisors that take a luma extent to a chroma extent.
inline std::pair<uint32_t, uint32_t> chromaSubsamplingFactors(nvimgcodecChromaSubsampling_t css)
{
    switch (css) {
    case NVIMGCODEC_SAMPLING_NONE:  // == NVIMGCODEC_SAMPLING_444
    case NVIMGCODEC_SAMPLING_GRAY:
        return {1u, 1u};
    case NVIMGCODEC_SAMPLING_422:
        return {2u, 1u};
    case NVIMGCODEC_SAMPLING_420:
        return {2u, 2u};
    case NVIMGCODEC_SAMPLING_440:
        return {1u, 2u};
    case NVIMGCODEC_SAMPLING_411:
        return {4u, 1u};
    case NVIMGCODEC_SAMPLING_410:
        return {4u, 2u};
    default:
        // Includes NVIMGCODEC_SAMPLING_410V, whose luma/chroma ratio is described
        // inconsistently in-tree (the JPEG parser derives it from luma sampling
        // factors 2x4, while the Python enum docstring describes 4x2), so no
        // plane geometry can be derived for it safely. Thrown as
        // std::invalid_argument so Decoder::decode re-raises it to Python instead
        // of logging it and yielding None.
        throw std::invalid_argument("Unsupported chroma subsampling requested for the decoded output.");
    }
}

// Relabels a chroma subsampling to match a 90/270-degree rotation of the extents it
// describes: the horizontal and vertical chroma factors swap. NONE/444/GRAY/420 are
// symmetric (their factors are equal), so they are returned unchanged; 422 and 440 are
// each other's transpose. 411 (4,1) and 410 (4,2) have no representable transposed
// value ((1,4) and (2,4) are not expressible chroma-subsampling ratios), so those raise
// rather than silently mislabel the rotated output.
inline nvimgcodecChromaSubsampling_t transposeChromaSubsampling(nvimgcodecChromaSubsampling_t css)
{
    switch (css) {
    case NVIMGCODEC_SAMPLING_NONE: // == NVIMGCODEC_SAMPLING_444
    case NVIMGCODEC_SAMPLING_GRAY:
    case NVIMGCODEC_SAMPLING_420:
        return css;
    case NVIMGCODEC_SAMPLING_422:
        return NVIMGCODEC_SAMPLING_440;
    case NVIMGCODEC_SAMPLING_440:
        return NVIMGCODEC_SAMPLING_422;
    default:
        throw std::invalid_argument(
            "Decoding a 90/270-degree rotated image would require a chroma subsampling with no "
            "representable label (the transpose of the requested/source subsampling). Only "
            "CSS_444/CSS_NONE, CSS_GRAY, CSS_420, CSS_422 and CSS_440 support EXIF rotation.");
    }
}

// True for the values that all mean "the chroma is not subsampled". GRAY is included:
// a luma-only output has no chroma to subsample.
inline bool is_unsubsampled_chroma(nvimgcodecChromaSubsampling_t css)
{
    return css == NVIMGCODEC_SAMPLING_NONE || css == NVIMGCODEC_SAMPLING_444 ||
           css == NVIMGCODEC_SAMPLING_GRAY;
}

// The output subsampling: the caller's value when given, otherwise exactly what the
// decoder produced before this parameter existed.
//
// `source_chroma_subsampling` is the subsampling of the code stream being decoded.
// Decoders emit the chroma components at the *code stream's* sampling rate - they do
// not resample - so a subsampled request that disagrees with the source would make us
// size and stride the output planes for one rate while the decoder writes another,
// overrunning the allocation. Such requests are rejected here.
//
// Errors are thrown as std::invalid_argument (rather than std::runtime_error) so that
// Decoder::decode's per-sample exception handling (which only re-raises
// invalid_argument to the caller; other std::exception subclasses are logged and
// turned into a None result) actually surfaces them to Python.
inline nvimgcodecChromaSubsampling_t resolveChromaSubsampling(
    std::optional<nvimgcodecChromaSubsampling_t> requested, const ResolvedFormat& resolved,
    nvimgcodecChromaSubsampling_t source_chroma_subsampling)
{
    const nvimgcodecColorSpec_t family = sample_format_color_family(resolved.sample_format);
    const bool is_luma_only = family == NVIMGCODEC_COLORSPEC_GRAY || resolved.num_components == 1;

    if (!requested.has_value())
        return is_luma_only ? NVIMGCODEC_SAMPLING_GRAY : NVIMGCODEC_SAMPLING_NONE;

    const nvimgcodecChromaSubsampling_t css = requested.value();

    // Subsampled chroma only means anything when the chroma lives in its own plane.
    const bool is_planar_sycc_output = !resolved.is_interleaved && family == NVIMGCODEC_COLORSPEC_SYCC;

    if (is_unsubsampled_chroma(css)) {
        // GRAY, 444 and NONE all describe an output with full-size chroma, which is
        // exactly what the unset path asks for; the default behaviour is unchanged.
        // GRAY additionally claims there is no chroma at all, so it must not be
        // accepted for an output that has chroma planes - the planes would be
        // allocated full size, reported as "not subsampled" by the flat array
        // interfaces, and left uninitialised by decoders that honour GRAY by writing
        // luma only.
        if (css == NVIMGCODEC_SAMPLING_GRAY && !is_luma_only) {
            throw std::invalid_argument(
                "ChromaSubsampling.CSS_GRAY describes an output with no chroma and requires a "
                "single-component output; use ChromaSubsampling.CSS_444 for a full-resolution "
                "multi-component output.");
        }
        // CSS_444/CSS_NONE on a planar YUV output asks for full-resolution chroma planes, but
        // decoders write planar YUV chroma at the code stream's native rate (see the mismatch
        // check below) rather than resampling it. On a genuinely subsampled source this would
        // allocate full-size planes that the decoder only partially fills, leaving the rest
        // uninitialised - verified empirically: decoding a 4:2:0 JPEG to CSS_444/P_YUV leaves
        // everything but the top-left quadrant of each chroma plane as zero. GRAY is exempt
        // since it has no chroma planes to under-fill.
        if (css != NVIMGCODEC_SAMPLING_GRAY && is_planar_sycc_output && !is_unsubsampled_chroma(source_chroma_subsampling)) {
            throw std::invalid_argument(
                "The requested full-resolution chroma subsampling (CSS_444) cannot be honoured "
                "for a planar YUV output of a chroma-subsampled source: decoding does not "
                "resample chroma, so only the "
                "native-resolution corner of the full-size chroma planes would be written. "
                "Request the source's native chroma_subsampling instead, or decode to an "
                "interleaved/RGB output to get true full-resolution chroma.");
        }
        return css;
    }

    // Subsampled chroma only means anything when the chroma lives in its own plane.
    if (!is_planar_sycc_output) {
        throw std::invalid_argument(
            "Chroma subsampling other than CSS_444 requires a planar YUV output; "
            "set sample_format to SampleFormat.P_YUV and color_spec to ColorSpec.SYCC.");
    }

    if (css != source_chroma_subsampling) {
        throw std::invalid_argument(
            "The requested chroma subsampling does not match the one of the input code stream. "
            "Decoding does not resample chroma: the requested subsampling must match the source's, "
            "or be left unset (or CSS_444) to get a full-resolution output.");
    }

    // Rejects values we cannot describe (currently only 410V). Only reachable when the
    // *source* itself is such a value: every other unsupported css is already caught by
    // the mismatch check above, since a source subsampling comes from the parser and can
    // never be one of the values chromaSubsamplingFactors() rejects. No fixture with a
    // genuinely 410V-encoded source exists in this repo, so this specific branch is
    // untested; test_css_410v_raises_instead_of_returning_none below exercises the
    // mismatch branch instead (a 410V request against a 4:2:0 source), not this one.
    chromaSubsamplingFactors(css);
    return css;
}

// Shared padding check for is_padding_correct (array-interface byte strides) and
// is_dlpack_padding_correct (DLPack element strides). Only the row axis may carry
// padding; every other axis must be packed against the running natural stride.
// `base_stride` is the size of one element in the same units as `strides` - the
// item size in bytes for array interfaces, or 1 for DLPack element strides.
inline bool are_strides_row_padded_only(const std::vector<int64_t>& shape,
    const std::vector<int64_t>& strides, int64_t base_stride, bool is_interleaved)
{
    if (shape.size() != strides.size()) {
        return false; // malformed input: one stride per axis is required
    }
    const int ndim = static_cast<int>(shape.size());

    // A negative stride (e.g. a reversed/flipped view) is rejected outright, not just
    // for the non-row axes the loop below already constrains: row_stride is assigned
    // into nvimgcodecPlaneInfo_t::row_stride, which is size_t, so a negative value
    // would silently wrap to a huge unsigned pitch and read far outside the buffer.
    for (int64_t s : strides) {
        if (s < 0) {
            return false;
        }
    }

    // Row axis is at index 1 only in 3-D planar (CHW). Everything else -
    // HWC, 2-D HW (with or without a planar sample_format) - puts the
    // row axis at index 0. Plane gaps in CHW's outer axis are caught by
    // the standard equality check below: the C image_info struct has no
    // per-plane offset, so any plane stride other than height*row_stride
    // would be silently misread by downstream codec consumers.
    const int row_idx = (ndim == 3 && !is_interleaved) ? 1 : 0;

    int64_t expected_stride = base_stride;
    for (int i = ndim - 1; i >= 0; --i) {
        // first dimension in plane can have any stride - we allow padding for rows
        if (i == row_idx) {
            expected_stride = strides[i];
        } else if (strides[i] != expected_stride) {
            return false;
        }
        // Advance the natural-stride accumulator on every axis - the
        // row axis must update it too so the next-outer axis (e.g.
        // CHW's plane axis at i=0) is compared against
        // row_stride * shape[row_idx], not just row_stride.
        expected_stride *= shape[i];
    }
    return true;
}

inline bool is_padding_correct(const py::dict& iface, bool is_interleaved)
{
    if (!iface.contains("strides")) {
        return true; // Assumed None which is for packed arrays
    }
    py::object strides = iface["strides"];
    if (strides.is(py::none())) {
        return true; // None strides means a packed array
    }

    py::tuple t_strides = strides.cast<py::tuple>();
    std::string type_str = iface["typestr"].cast<std::string>();
    int64_t item_size = py::dtype(type_str).itemsize();
    py::tuple t_shape = iface["shape"].cast<py::tuple>();

    std::vector<int64_t> shape, vstrides;
    shape.reserve(t_shape.size());
    vstrides.reserve(t_strides.size());
    for (auto& o : t_shape)
        shape.push_back(o.cast<int64_t>());
    for (auto& o : t_strides)
        vstrides.push_back(o.cast<int64_t>());

    return are_strides_row_padded_only(shape, vstrides, item_size, is_interleaved);
}

// Distinguishes a negative-stride (reversed/flipped view) rejection from a genuine
// non-row-padding rejection, so callers can throw an accurate message instead of
// always reporting "padding" for something that is not a padding issue at all.
inline bool has_negative_stride(const py::dict& iface)
{
    if (!iface.contains("strides")) {
        return false;
    }
    py::object strides = iface["strides"];
    if (strides.is(py::none())) {
        return false;
    }
    for (auto& o : strides.cast<py::tuple>()) {
        if (o.cast<int64_t>() < 0) {
            return true;
        }
    }
    return false;
}

} // namespace nvimgcodec