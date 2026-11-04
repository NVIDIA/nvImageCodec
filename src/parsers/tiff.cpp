/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "parsers/tiff.h"
#include "parsers/codec_file_extensions.h"
#include <nvimgcodec.h>
#include <string.h>
#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <type_traits>
#include <vector>

#include "imgproc/exception.h"
#include "imgproc/math_util.h"
#include "exif_orientation.h"
#include "log_ext.h"

#include "parsers/byte_io.h"
#include "parsers/exif.h"
#include "tiff_utils.h"

namespace nvimgcodec {

namespace {

constexpr int ENTRY_SIZE = 12;
constexpr int BIG_ENTRY_SIZE = 20;

enum class TiffVariant {
    Classic,       // Version 42, 32-bit offsets
    BigTiff,       // Version 43, 64-bit offsets
    HamamatsuNDPI  // Version 42 with 64-bit offset extension (files >4GB)
};

struct TiffParsedIfd
{
    size_t next_ifd_offset = 0;
    std::vector<size_t> subifd_offsets;
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), nullptr};
    nvimgcodecTileGeometryInfo_t tile_geometry_info{NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO, sizeof(nvimgcodecTileGeometryInfo_t), nullptr};
};

} // namespace

struct TiffParserCache
{
    // Endianness + variant of the backing stream, detected once (NDPI detection rescans
    // the first IFD, so we avoid repeating it for every metadata call / substream).
    struct TiffLayout {
        bool little_endian;
        TiffVariant variant;
    };
    std::optional<TiffLayout> layout;

    bool root_ifd_offsets_parsed = false;
    std::vector<size_t> root_ifd_offsets;
    std::set<size_t> root_ifd_offsets_seen;
    std::map<size_t, TiffParsedIfd> ifds_by_offset;
};

namespace {

constexpr uint16_t NDPI_FORMAT_FLAG_TAG = 65420;

enum TiffTag : uint16_t
{
    WIDTH_TAG = 256,
    HEIGHT_TAG = 257,
    PHOTOMETRIC_INTERPRETATION_TAG = 262,
    ORIENTATION_TAG = 274,
    SAMPLESPERPIXEL_TAG = 277,
    ROWS_PER_STRIP_TAG = 278,
    BITSPERSAMPLE_TAG = 258,
    TILE_WIDTH = 322,
    TILE_LENGTH = 323,
    TILE_OFFSETS_TAG = 324,
    SUBIFD_TAG = 330,
    SAMPLE_FORMAT_TAG = 339
};

enum TagType {
    BYTE = 1,
    ASCII = 2,
    SHORT = 3,
    LONG = 4,
    RATIONAL = 5,
    SBYTE = 6,
    UNDEFINED = 7,
    SSHORT = 8,
    SLONG = 9,
    SRATIONAL = 10,
    FLOAT = 11,
    DOUBLE = 12,
    IFD = 13,
    LONG8 = 16,
    SLONG8 = 17,
    IFD8 = 18
};

enum TiffSampleFormat : uint16_t
{
    TIFF_SAMPLEFORMAT_UNINITIALIZED = 0,
    TIFF_SAMPLEFORMAT_UINT          = 1,
    TIFF_SAMPLEFORMAT_INT           = 2,
    TIFF_SAMPLEFORMAT_IEEEFP        = 3,
    TIFF_SAMPLEFORMAT_UNDEFINED     = 4
};

enum TiffPhotometricInterpretation : uint16_t
{
    PHOTOMETRIC_WHITEISZERO = 0,    /**< For grayscale: 0=white, higher values=darker */
    PHOTOMETRIC_BLACKISZERO = 1,    /**< For grayscale: 0=black, higher values=brighter */
    PHOTOMETRIC_RGB = 2,            /**< RGB color model */
    PHOTOMETRIC_PALETTE = 3,        /**< Palette color (color map) */
    PHOTOMETRIC_MASK = 4,           /**< Transparency mask (deprecated) */
    PHOTOMETRIC_CMYK = 5,           /**< CMYK color model */
    PHOTOMETRIC_YCBCR = 6,          /**< YCbCr color model */
    PHOTOMETRIC_CIELAB = 8,         /**< CIE L*a*b* color model */
    PHOTOMETRIC_ICCLAB = 9,         /**< ICC L*a*b* color model */
    PHOTOMETRIC_ITULAB = 10,        /**< ITU L*a*b* color model */
    PHOTOMETRIC_CFA = 32803,        /**< Color Filter Array (Bayer pattern) */
    PHOTOMETRIC_LINEARRAW = 34892   /**< Linear Raw */
};

using tiff_magic_t = std::array<uint8_t, 4>;
// Regular TIFF
constexpr tiff_magic_t le_header = { 'I', 'I', 42, 0 };  // Little endian
constexpr tiff_magic_t be_header = { 'M', 'M', 0, 42 };  // Big endian
// BigTIFF
constexpr tiff_magic_t le_bigtiff = { 'I', 'I', 43, 0 }; // Little endian
constexpr tiff_magic_t be_bigtiff = { 'M', 'M', 0, 43 }; // Big endian

template <typename T, bool is_little_endian>
T TiffRead(nvimgcodecIoStreamDesc_t* io_stream)
{
    if constexpr (is_little_endian) {
        return ReadValueLE<T>(io_stream);
    } else {
        return ReadValueBE<T>(io_stream);
    }
}

void seekTiffStream(nvimgcodecIoStreamDesc_t* io_stream, ptrdiff_t offset, int whence)
{
    if (io_stream->seek(io_stream->instance, offset, whence) != NVIMGCODEC_STATUS_SUCCESS) {
        throw std::runtime_error("TIFF stream seek failed");
    }
}

void seekTiffStreamTo(nvimgcodecIoStreamDesc_t* io_stream, size_t offset)
{
    if (offset > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max())) {
        throw std::runtime_error("TIFF stream seek offset is too large");
    }
    seekTiffStream(io_stream, static_cast<ptrdiff_t>(offset), SEEK_SET);
}

template <TiffVariant variant>
struct TiffLayout
{
    using OffsetType = std::conditional_t<variant == TiffVariant::Classic, uint32_t, uint64_t>;
    using EntryCountType = std::conditional_t<variant == TiffVariant::BigTiff, uint64_t, uint16_t>;
    using EntryValueCountType = std::conditional_t<variant == TiffVariant::BigTiff, uint64_t, uint32_t>;
    using EntryOffsetType = std::conditional_t<variant == TiffVariant::BigTiff, uint64_t, uint32_t>;

    static constexpr size_t entry_size = (variant == TiffVariant::BigTiff) ? BIG_ENTRY_SIZE : ENTRY_SIZE;
    static constexpr size_t inline_limit = (variant == TiffVariant::BigTiff) ? 8 : 4;
};

size_t getTypeSize(uint16_t type) {
    switch (static_cast<TagType>(type)) {
    case TagType::BYTE:
    case TagType::ASCII:
    case TagType::SBYTE:
    case TagType::UNDEFINED:
        return 1;
    case TagType::SHORT:
    case TagType::SSHORT:
        return 2;
    case TagType::LONG:
    case TagType::SLONG:
    case TagType::FLOAT:
    case TagType::IFD:
        return 4;
    case TagType::RATIONAL:
    case TagType::SRATIONAL:
    case TagType::DOUBLE:
    case TagType::LONG8:
    case TagType::SLONG8:
    case TagType::IFD8:
        return 8;
    default:
        return 0;
    }
}

size_t getTagValueSize(uint16_t value_type, uint64_t value_count)
{
    const size_t value_type_size = getTypeSize(value_type);
    if (value_type_size == 0 || value_count > std::numeric_limits<size_t>::max() / value_type_size) {
        throw std::runtime_error("Invalid TIFF tag value size");
    }
    return static_cast<size_t>(value_count) * value_type_size;
}

struct TagValueLocation
{
    bool stored_out_of_line = false;
    ptrdiff_t return_pos = 0;
};

template<typename EntryOffsetType, bool is_little_endian>
TagValueLocation seekToTagValues(
    nvimgcodecIoStreamDesc_t* io_stream, uint16_t value_type, uint64_t value_count, size_t inline_limit, const char* bounds_error)
{
    TagValueLocation location{};
    const size_t value_size = getTagValueSize(value_type, value_count);
    if (value_size <= inline_limit) {
        return location;
    }

    const auto offset = TiffRead<EntryOffsetType, is_little_endian>(io_stream);
    io_stream->tell(io_stream->instance, &location.return_pos);
    size_t stream_size;
    io_stream->size(io_stream->instance, &stream_size);
    if (offset > stream_size || value_size > stream_size - static_cast<size_t>(offset)) {
        throw std::runtime_error(bounds_error);
    }
    seekTiffStreamTo(io_stream, static_cast<size_t>(offset));
    location.stored_out_of_line = true;
    return location;
}

void restoreAfterTagValues(nvimgcodecIoStreamDesc_t* io_stream, const TagValueLocation& location)
{
    if (location.stored_out_of_line) {
        seekTiffStream(io_stream, location.return_pos, SEEK_SET);
    }
}

template<bool is_little_endian>
uint64_t readUnsignedScalarTagValue(nvimgcodecIoStreamDesc_t* io_stream, uint16_t value_type)
{
    if (value_type == SHORT) {
        return TiffRead<uint16_t, is_little_endian>(io_stream);
    }
    if (value_type == LONG) {
        return TiffRead<uint32_t, is_little_endian>(io_stream);
    }
    if (value_type == LONG8) {
        return TiffRead<uint64_t, is_little_endian>(io_stream);
    }
    throw std::runtime_error("Couldn't read TIFF tag, type should be SHORT, LONG, or LONG8 but is not.");
}

nvimgcodecSampleDataType_t convert_to_sample_type(
    uint16_t bitdepth, bool sample_format_read, TiffSampleFormat sample_format)
{
    // Convert sample_format to internal sample_type
    if (!sample_format_read || sample_format == TIFF_SAMPLEFORMAT_UNDEFINED
                            || sample_format == TIFF_SAMPLEFORMAT_UNINITIALIZED) {
        sample_format = TIFF_SAMPLEFORMAT_UINT; // default according to standard
    }

    // TODO: Do we have decoders for all bitdepths?
    switch (sample_format) {
    case TIFF_SAMPLEFORMAT_UINT:
        if (bitdepth <= 8) {
            return NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        } else if (bitdepth <= 16) {
            return NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16;
        } else if (bitdepth <= 32) {
            return NVIMGCODEC_SAMPLE_DATA_TYPE_UINT32;
        } else if (bitdepth <= 64) {
            return NVIMGCODEC_SAMPLE_DATA_TYPE_UINT64;
        }
        return NVIMGCODEC_SAMPLE_DATA_TYPE_UNSUPPORTED;
    case TIFF_SAMPLEFORMAT_INT:
        if (bitdepth <= 8) {
            return NVIMGCODEC_SAMPLE_DATA_TYPE_INT8;
        } else if (bitdepth <= 16) {
            return NVIMGCODEC_SAMPLE_DATA_TYPE_INT16;
        } else if (bitdepth <= 32) {
            return NVIMGCODEC_SAMPLE_DATA_TYPE_INT32;
        } else if (bitdepth <= 64) {
            return NVIMGCODEC_SAMPLE_DATA_TYPE_INT64;
        }
        return NVIMGCODEC_SAMPLE_DATA_TYPE_UNSUPPORTED;
    case TIFF_SAMPLEFORMAT_IEEEFP:
        if (bitdepth == 32) {
            return NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32;
        } else if (bitdepth == 16) {
            return NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT16;
        } else if (bitdepth == 64) {
            return NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT64;
        }
        return NVIMGCODEC_SAMPLE_DATA_TYPE_UNSUPPORTED;
    default:
        return NVIMGCODEC_SAMPLE_DATA_TYPE_UNSUPPORTED;
    }
}

nvimgcodecColorSpec_t convert_photometric_to_color_spec(TiffPhotometricInterpretation photometric)
{
    switch (photometric) {
        case PHOTOMETRIC_WHITEISZERO:
        case PHOTOMETRIC_BLACKISZERO:
            return NVIMGCODEC_COLORSPEC_GRAY;
        case PHOTOMETRIC_RGB:
            return NVIMGCODEC_COLORSPEC_SRGB;
        case PHOTOMETRIC_PALETTE:
            return NVIMGCODEC_COLORSPEC_PALETTE;
        case PHOTOMETRIC_CMYK:
            return NVIMGCODEC_COLORSPEC_CMYK;
        case PHOTOMETRIC_YCBCR:
            return NVIMGCODEC_COLORSPEC_SYCC;
        case PHOTOMETRIC_ICCLAB:
            return NVIMGCODEC_COLORSPEC_ICC_PROFILE;
        case PHOTOMETRIC_CIELAB:
        case PHOTOMETRIC_ITULAB:
        case PHOTOMETRIC_MASK:
        case PHOTOMETRIC_CFA:
        case PHOTOMETRIC_LINEARRAW:
        default:
            return NVIMGCODEC_COLORSPEC_UNSUPPORTED;
    }
}

// Helper to read IFD offset based on variant
template <bool is_little_endian, TiffVariant variant>
uint64_t readIFDOffset(nvimgcodecIoStreamDesc_t* io_stream)
{
    if constexpr (variant == TiffVariant::BigTiff) {
        return TiffRead<uint64_t, is_little_endian>(io_stream);
    } else if constexpr (variant == TiffVariant::HamamatsuNDPI) {
        uint32_t low = TiffRead<uint32_t, is_little_endian>(io_stream);
        uint32_t high = TiffRead<uint32_t, is_little_endian>(io_stream);
        return (static_cast<uint64_t>(high) << 32) | low;
    } else {
        return TiffRead<uint32_t, is_little_endian>(io_stream);
    }
}

// Detect Hamamatsu NDPI format (classic TIFF with 64-bit offset extension for files >4GB).
// NDPI uses version 42 but extends IFD offset to 64-bit: low 32 bits at byte 4, high 32 bits at byte 8.
// Identified by NDPI_FORMAT_FLAG tag (65420) in the first IFD.
template <bool is_little_endian>
bool detectLargeNDPI(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, nvimgcodecIoStreamDesc_t* io_stream)
{
    size_t file_size;
    io_stream->size(io_stream->instance, &file_size);

    if (file_size <= 0xFFFFFFFFull)
        return false;

    ptrdiff_t saved_pos;
    io_stream->tell(io_stream->instance, &saved_pos);

    // This operates under the assumption that the file and its offsets are stored in Hamamatsu NDPI format.
    // Any caught IO errors or missing required Hamamatsu tags indicate that the file is not a Hamamatsu NDPI.
    try {
        // Read 64-bit IFD offset: low 32 bits at byte 4, high 32 bits at byte 8
        seekTiffStream(io_stream, 4, SEEK_SET);
        uint64_t ifd_offset = readIFDOffset<is_little_endian, TiffVariant::HamamatsuNDPI>(io_stream);

        if (ifd_offset == 0 || ifd_offset >= file_size) {
            seekTiffStream(io_stream, saved_pos, SEEK_SET);
            return false;
        }

        seekTiffStreamTo(io_stream, static_cast<size_t>(ifd_offset));
        uint16_t num_entries = TiffRead<uint16_t, is_little_endian>(io_stream);

        if (num_entries == 0 || num_entries > 4096) {
            seekTiffStream(io_stream, saved_pos, SEEK_SET);
            return false;
        }

        // Scan IFD entries for NDPI_FORMAT_FLAG tag (65420)
        for (uint16_t i = 0; i < num_entries; i++) {
            uint16_t tag = TiffRead<uint16_t, is_little_endian>(io_stream);
            if (tag == NDPI_FORMAT_FLAG_TAG) {
                seekTiffStream(io_stream, saved_pos, SEEK_SET);
                NVIMGCODEC_LOG_INFO(framework, plugin_id, "Detected Hamamatsu NDPI format (file size: " << file_size << " bytes, IFD offset: " << ifd_offset << ")");
                return true;
            }
            // Skip: type (2 bytes) + count (4 bytes) + value/offset (4 bytes) = 10 bytes
            seekTiffStream(io_stream, 10, SEEK_CUR);
        }
    } catch (...) {
    }

    seekTiffStream(io_stream, saved_pos, SEEK_SET);
    return false;
}

void copyToTiffExt(nvimgcodecCodeStreamInfoTiffExt_t* ext, size_t ifd_offset, const TiffParsedIfd& ifd)
{
    if (!ext) {
        return;
    }

    ext->ifd_offset = ifd_offset;
    ext->next_ifd_offset = ifd.next_ifd_offset;
    const auto subifd_count = std::min<size_t>(ifd.subifd_offsets.size(), NVIMGCODEC_MAX_SUBIFD_OFFSETS);
    ext->subifd_count = static_cast<uint32_t>(subifd_count);
    std::fill(std::begin(ext->subifd_offsets), std::end(ext->subifd_offsets), 0);
    for (size_t i = 0; i < subifd_count; ++i) {
        ext->subifd_offsets[i] = ifd.subifd_offsets[i];
    }
}

void resetTiffExt(nvimgcodecCodeStreamInfoTiffExt_t* ext)
{
    if (!ext) {
        return;
    }

    ext->ifd_offset = 0;
    ext->next_ifd_offset = 0;
    ext->subifd_count = 0;
    std::fill(std::begin(ext->subifd_offsets), std::end(ext->subifd_offsets), 0);
}

void copyCachedImageInfo(nvimgcodecImageInfo_t* info, const TiffParsedIfd& ifd)
{
    void* struct_next = info->struct_next;
    *info = ifd.image_info;
    info->struct_next = struct_next;

    while (struct_next) {
        auto* ptr = reinterpret_cast<nvimgcodecImageInfo_t*>(struct_next);
        auto* next_struct_next = ptr->struct_next;
        if (ptr->struct_type == NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO) {
            auto* tile_geometry_info = reinterpret_cast<nvimgcodecTileGeometryInfo_t*>(ptr);
            void* tile_next = tile_geometry_info->struct_next;
            *tile_geometry_info = ifd.tile_geometry_info;
            tile_geometry_info->struct_next = tile_next;
        }
        struct_next = next_struct_next;
    }
}

template<typename EntryOffsetType, bool is_little_endian>
void readSubIfdOffsets(nvimgcodecIoStreamDesc_t* io_stream, uint16_t value_type, uint64_t value_count, size_t inline_limit, TiffParsedIfd& ifd)
{
    const auto tag_location = seekToTagValues<EntryOffsetType, is_little_endian>(
        io_stream, value_type, value_count, inline_limit, "TIFF SubIFD array points outside the stream");

    const auto stored_count = std::min<uint64_t>(value_count, NVIMGCODEC_MAX_SUBIFD_OFFSETS);
    ifd.subifd_offsets.clear();
    ifd.subifd_offsets.reserve(static_cast<size_t>(stored_count));
    for (uint64_t i = 0; i < stored_count; i++) {
        if (value_type == LONG || value_type == IFD) {
            ifd.subifd_offsets.push_back(TiffRead<uint32_t, is_little_endian>(io_stream));
        } else if (value_type == LONG8 || value_type == IFD8) {
            ifd.subifd_offsets.push_back(TiffRead<uint64_t, is_little_endian>(io_stream));
        } else {
            throw std::runtime_error("Unsupported TIFF SubIFD offset type");
        }
    }

    restoreAfterTagValues(io_stream, tag_location);
}

template<bool is_little_endian, TiffVariant variant>
nvimgcodecStatus_t ensureParsedIfd(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, TiffParserCache& cache,
    nvimgcodecIoStreamDesc_t* io_stream, size_t selected_ifd_offset, TiffParsedIfd** parsed_ifd)
{
    auto cached_ifd = cache.ifds_by_offset.find(selected_ifd_offset);
    if (cached_ifd != cache.ifds_by_offset.end()) {
        *parsed_ifd = &cached_ifd->second;
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    using Layout = TiffLayout<variant>;
    TiffParsedIfd ifd{};

    try {
        seekTiffStreamTo(io_stream, selected_ifd_offset);
        const auto entry_count = TiffRead<typename Layout::EntryCountType, is_little_endian>(io_stream);
        size_t stream_size = 0;
        if (io_stream->size(io_stream->instance, &stream_size) != NVIMGCODEC_STATUS_SUCCESS) {
            throw std::runtime_error("TIFF stream size query failed");
        }
        if (selected_ifd_offset > stream_size || sizeof(typename Layout::EntryCountType) > stream_size - selected_ifd_offset) {
            throw std::runtime_error("TIFF IFD entry count points outside the stream");
        }

        const auto entry_count_64 = static_cast<uint64_t>(entry_count);
        if (entry_count_64 > std::numeric_limits<size_t>::max() / Layout::entry_size) {
            throw std::runtime_error("TIFF IFD entry table size overflow");
        }

        const size_t entries_offset = selected_ifd_offset + sizeof(typename Layout::EntryCountType);
        const size_t entries_size = static_cast<size_t>(entry_count_64) * Layout::entry_size;
        if (entries_size > stream_size - entries_offset) {
            throw std::runtime_error("TIFF IFD entry table points outside the stream");
        }
        const size_t next_ifd_offset_pos = entries_offset + entries_size;
        if (sizeof(typename Layout::OffsetType) > stream_size - next_ifd_offset_pos) {
            throw std::runtime_error("TIFF IFD entry table points outside the stream");
        }

        nvimgcodecImageInfo_t parsed_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), nullptr};
        strcpy(parsed_info.codec_name, "tiff");
        parsed_info.color_spec = NVIMGCODEC_COLORSPEC_UNKNOWN;
        parsed_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        parsed_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
        parsed_info.orientation = {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 0, false, false};

        bool width_read = false, height_read = false, samples_per_px_read = false, palette_read = false,
            bitdepth_read = false, sample_format_read = false;
        uint32_t width = 0, height = 0, nchannels = 0;
        std::array<uint16_t, NVIMGCODEC_MAX_NUM_PLANES> bitdepth = {};
        size_t bitdepth_count = 0;
        std::array<TiffSampleFormat, NVIMGCODEC_MAX_NUM_PLANES> sample_format = {};
        size_t sample_format_count = 0;

        bool tile_width_read = false, tile_height_read = false, rows_per_strip_read = false;
        uint32_t strile_width = 0, strile_height = 0;

        for (typename Layout::EntryCountType entry_idx = 0; entry_idx < entry_count; entry_idx++) {
            const auto entry_offset = entries_offset + static_cast<size_t>(entry_idx) * Layout::entry_size;
            seekTiffStreamTo(io_stream, entry_offset);
            const auto tag_id = TiffRead<uint16_t, is_little_endian>(io_stream);
            const auto value_type = TiffRead<uint16_t, is_little_endian>(io_stream);
            const auto value_count = TiffRead<typename Layout::EntryValueCountType, is_little_endian>(io_stream);

            if (tag_id == BITSPERSAMPLE_TAG || tag_id == SAMPLE_FORMAT_TAG) {
                seekToTagValues<typename Layout::EntryOffsetType, is_little_endian>(
                    io_stream, value_type, value_count, Layout::inline_limit, "TIFF tag array points outside the stream");

                if (tag_id == BITSPERSAMPLE_TAG) {
                    if (value_type != SHORT) {
                        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Bits per sample tag should have SHORT type.");
                        return NVIMGCODEC_STATUS_BAD_CODESTREAM;
                    }

                    if (value_count > NVIMGCODEC_MAX_NUM_PLANES) {
                        NVIMGCODEC_LOG_ERROR(framework, plugin_id,
                            "Couldn't read TIFF with more than " << NVIMGCODEC_MAX_NUM_PLANES << " components. Got " << value_count
                                                                << "values for bits per sample tag."
                        );
                        return NVIMGCODEC_STATUS_CODESTREAM_UNSUPPORTED;
                    }

                    for (size_t i = 0; i < value_count; i++) {
                        bitdepth[i] = TiffRead<uint16_t, is_little_endian>(io_stream);
                    }

                    bitdepth_count = value_count;
                    bitdepth_read = true;
                } else if (tag_id == SAMPLE_FORMAT_TAG) {
                    if (value_type != SHORT) {
                        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Sample format tag should have SHORT type.");
                        return NVIMGCODEC_STATUS_BAD_CODESTREAM;
                    }

                    if (value_count > NVIMGCODEC_MAX_NUM_PLANES) {
                        NVIMGCODEC_LOG_ERROR(framework, plugin_id,
                            "Couldn't read TIFF with more than " << NVIMGCODEC_MAX_NUM_PLANES << " components. Got " << value_count
                                                                << "values for sample format tag."
                        );
                        return NVIMGCODEC_STATUS_CODESTREAM_UNSUPPORTED;
                    }

                    for (size_t i = 0; i < value_count; ++i) {
                        sample_format[i] = static_cast<TiffSampleFormat>(TiffRead<uint16_t, is_little_endian>(io_stream));
                    }
                    sample_format_count = value_count;
                    sample_format_read = true;
                }
            } else if (tag_id == WIDTH_TAG || tag_id == HEIGHT_TAG || tag_id == SAMPLESPERPIXEL_TAG || tag_id == ORIENTATION_TAG ||
                       tag_id == PHOTOMETRIC_INTERPRETATION_TAG || tag_id == TILE_WIDTH || tag_id == TILE_LENGTH || tag_id == ROWS_PER_STRIP_TAG) {
                if (value_count != 1) {
                    NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unexpected value count");
                    return NVIMGCODEC_STATUS_BAD_CODESTREAM;
                }

                const auto value = readUnsignedScalarTagValue<is_little_endian>(io_stream, value_type);

                if (value > std::numeric_limits<uint32_t>::max()) {
                    NVIMGCODEC_LOG_ERROR(framework, plugin_id, "TIFF dimension tag value exceeds 32-bit limits.");
                    return NVIMGCODEC_STATUS_CODESTREAM_UNSUPPORTED;
                }

                if (tag_id == WIDTH_TAG) {
                    width = value;
                    width_read = true;
                } else if (tag_id == HEIGHT_TAG) {
                    height = value;
                    height_read = true;
                } else if (tag_id == ORIENTATION_TAG) {
                    parsed_info.orientation = FromExifOrientation(static_cast<ExifOrientation>(value));
                } else if (tag_id == SAMPLESPERPIXEL_TAG && !palette_read) {
                    nchannels = value;
                    samples_per_px_read = true;
                    if (nchannels > NVIMGCODEC_MAX_NUM_PLANES) {
                        NVIMGCODEC_LOG_ERROR(framework, plugin_id,
                            "Couldn't read TIFF with more than " << NVIMGCODEC_MAX_NUM_PLANES << " components. Got " << nchannels
                                                                 << " value for samples per pixel tag."
                        );
                        return NVIMGCODEC_STATUS_CODESTREAM_UNSUPPORTED;
                    }
                } else if (tag_id == PHOTOMETRIC_INTERPRETATION_TAG ) {
                    auto photometric = static_cast<TiffPhotometricInterpretation>(value);
                    parsed_info.color_spec = convert_photometric_to_color_spec(photometric);
                    if (photometric == PHOTOMETRIC_PALETTE) {
                        nchannels = 3;
                        palette_read = true;
                    } else if ((photometric == PHOTOMETRIC_BLACKISZERO || photometric == PHOTOMETRIC_WHITEISZERO) && !samples_per_px_read) {
                        nchannels = 1;
                        samples_per_px_read = true;
                    }
                } else if (tag_id == TILE_LENGTH) {
                    strile_height = value;
                    tile_height_read = true;
                } else if (tag_id == TILE_WIDTH) {
                    strile_width = value;
                    tile_width_read = true;
                } else if (tag_id == ROWS_PER_STRIP_TAG) {
                    strile_height = value;
                    rows_per_strip_read = true;
                }
            } else if (tag_id == SUBIFD_TAG) {
                readSubIfdOffsets<typename Layout::EntryOffsetType, is_little_endian>(
                    io_stream, value_type, value_count, Layout::inline_limit, ifd);
            }
        }

        seekTiffStreamTo(io_stream, next_ifd_offset_pos);
        ifd.next_ifd_offset = static_cast<size_t>(readIFDOffset<is_little_endian, variant>(io_stream));

        if (!width_read || !height_read || !bitdepth_read || (!samples_per_px_read && !palette_read)) {
            NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Couldn't read TIFF image required fields - missing: "
                << (!width_read ? "ImageWidth " : "")
                << (!height_read ? "ImageHeight " : "")
                << (!bitdepth_read ? "BitsPerSample " : "")
                << ((!samples_per_px_read && !palette_read) ? "SamplesPerPixel " : "")
                << "(IFD at offset " << selected_ifd_offset << ", entry_count=" << entry_count << ")");
            return NVIMGCODEC_STATUS_BAD_CODESTREAM;
        }

        if (width == 0 || height == 0) {
            NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Width or height cannot be 0.");
            return NVIMGCODEC_STATUS_BAD_CODESTREAM;
        }

        if (tile_width_read != tile_height_read) {
            NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Both tile width and height should be present.");
            return NVIMGCODEC_STATUS_BAD_CODESTREAM;
        }

        if (tile_width_read && rows_per_strip_read) {
            NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Image should have either tiles or strips, not both.");
            return NVIMGCODEC_STATUS_BAD_CODESTREAM;
        }

        if (tile_width_read && (strile_width == 0 || strile_height == 0)) {
            NVIMGCODEC_LOG_ERROR(framework, plugin_id, "TIFF tile dimensions must be non-zero.");
            return NVIMGCODEC_STATUS_BAD_CODESTREAM;
        }
        if (rows_per_strip_read && strile_height == 0) {
            NVIMGCODEC_LOG_ERROR(framework, plugin_id, "TIFF rows-per-strip must be non-zero.");
            return NVIMGCODEC_STATUS_BAD_CODESTREAM;
        }

        // The spec gives BitsPerSample and SampleFormat one value per channel, but images in the
        // wild -- GeoTIFFs especially -- write a single value meant for all of them. Left alone,
        // the entries the tag did not cover keep their zero initialisation, and neither zero is
        // rejected: an unset SampleFormat is treated as UINT, and an unset bitdepth of 0 satisfies
        // "bitdepth <= 8" and yields UINT8 at precision 0. A signed 16-bit 3-channel image would
        // come back as one INT16 plane and two UINT8 ones rather than failing. Extend the first
        // value over the remaining channels, which is what libtiff does.
        if (bitdepth_read && bitdepth_count > 0 && bitdepth_count < nchannels) {
            for (uint32_t i = static_cast<uint32_t>(bitdepth_count); i < nchannels; ++i) {
                bitdepth[i] = bitdepth[0];
            }
        }
        if (sample_format_read && sample_format_count > 0 && sample_format_count < nchannels) {
            for (uint32_t i = static_cast<uint32_t>(sample_format_count); i < nchannels; ++i) {
                sample_format[i] = sample_format[0];
            }
        }

        if (palette_read) {
            for (uint32_t i = 0; i < nchannels; ++i) {
                bitdepth[i] = 16;
                // Palette indices are always unsigned, regardless of the declared SampleFormat,
                // matching the decoder which produces unsigned palette color values.
                sample_format[i] = TIFF_SAMPLEFORMAT_UINT;
            }
        }

        parsed_info.num_planes = nchannels;
        for (size_t p = 0; p < parsed_info.num_planes; p++) {
            parsed_info.plane_info[p].height = height;
            parsed_info.plane_info[p].width = width;
            parsed_info.plane_info[p].num_channels = 1;
            parsed_info.plane_info[p].sample_type = convert_to_sample_type(
                bitdepth[p], sample_format_read, sample_format[p]
            );
            if (parsed_info.plane_info[p].sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_UNSUPPORTED) {
                NVIMGCODEC_LOG_ERROR(framework, plugin_id,
                    "Unsupported sample format " << sample_format[p]<< " with bitdepth "
                    << bitdepth[p] << " for channel " << p
                );
                return NVIMGCODEC_STATUS_CODESTREAM_UNSUPPORTED;
            }
            parsed_info.plane_info[p].precision = bitdepth[p];
        }

        if (nchannels == 1){
            parsed_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_Y;
            parsed_info.chroma_subsampling = NVIMGCODEC_SAMPLING_GRAY;
        } else if (nchannels == 2){
            parsed_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_YA;
            parsed_info.chroma_subsampling = NVIMGCODEC_SAMPLING_GRAY;
        } else if (nchannels == 3){
            parsed_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
        } else if (nchannels == 4){
            parsed_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGBA;
        } else {
            parsed_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED;
        }

        if (rows_per_strip_read) {
            strile_width = width;
            strile_height = std::min(strile_height, height);
        } else if (!tile_height_read) {
            strile_width = width;
            strile_height = height;
        }

        ifd.tile_geometry_info.tile_height = strile_height;
        ifd.tile_geometry_info.tile_width = strile_width;
        ifd.tile_geometry_info.num_tiles_y = div_up(height, strile_height);
        ifd.tile_geometry_info.num_tiles_x = div_up(width, strile_width);
        ifd.image_info = parsed_info;
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id,  "Failed to read IFD at offset " << selected_ifd_offset << " - " << e.what());
        return NVIMGCODEC_STATUS_BAD_CODESTREAM;
    }

    auto [it, inserted] = cache.ifds_by_offset.emplace(selected_ifd_offset, ifd);
    (void)inserted;
    *parsed_ifd = &it->second;
    return NVIMGCODEC_STATUS_SUCCESS;
}

// Seeds the root IFD chain with the first IFD and parses it to learn its next offset.
template <bool is_little_endian, TiffVariant variant>
nvimgcodecStatus_t ensureFirstRootIfd(
    const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, TiffParserCache& cache, nvimgcodecIoStreamDesc_t* io_stream)
{
    if (!cache.root_ifd_offsets.empty()) {
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    try {
        seekTiffStream(io_stream, 4, SEEK_SET);
        if constexpr (variant == TiffVariant::BigTiff) {
            auto version = TiffRead<uint16_t, is_little_endian>(io_stream);
            auto bytesize = TiffRead<uint16_t, is_little_endian>(io_stream);
            if (version != 8 || bytesize != 0) {
                return NVIMGCODEC_STATUS_BAD_CODESTREAM;
            }
        }

        const auto first_ifd_offset = static_cast<size_t>(readIFDOffset<is_little_endian, variant>(io_stream));
        if (first_ifd_offset == 0) {
            NVIMGCODEC_LOG_ERROR(framework, plugin_id, "TIFF does not contain a root IFD");
            return NVIMGCODEC_STATUS_BAD_CODESTREAM;
        }

        cache.root_ifd_offsets.push_back(first_ifd_offset);
        cache.root_ifd_offsets_seen.insert(first_ifd_offset);
        TiffParsedIfd* parsed_ifd = nullptr;
        auto status =
            ensureParsedIfd<is_little_endian, variant>(plugin_id, framework, cache, io_stream, first_ifd_offset, &parsed_ifd);
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            return status;
        }
        if (parsed_ifd->next_ifd_offset == 0) {
            cache.root_ifd_offsets_parsed = true;
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    } catch (...) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Failed to read TIFF first root IFD offset");
        return NVIMGCODEC_STATUS_BAD_CODESTREAM;
    }
}

// Extends the root IFD chain by one link, parsing the new IFD and detecting cycles.
template <bool is_little_endian, TiffVariant variant>
nvimgcodecStatus_t appendNextRootIfd(
    const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, TiffParserCache& cache, nvimgcodecIoStreamDesc_t* io_stream)
{
    if (cache.root_ifd_offsets_parsed) {
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    const auto current_ifd_offset = cache.root_ifd_offsets.back();
    TiffParsedIfd* current_ifd = nullptr;
    auto status = ensureParsedIfd<is_little_endian, variant>(plugin_id, framework, cache, io_stream, current_ifd_offset, &current_ifd);
    if (status != NVIMGCODEC_STATUS_SUCCESS) {
        return status;
    }

    const size_t next_ifd_offset = current_ifd->next_ifd_offset;
    if (next_ifd_offset == 0) {
        cache.root_ifd_offsets_parsed = true;
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    if (!cache.root_ifd_offsets_seen.insert(next_ifd_offset).second) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "File have cyclic structure, IFD offset is repeated.");
        return NVIMGCODEC_STATUS_BAD_CODESTREAM;
    }

    cache.root_ifd_offsets.push_back(next_ifd_offset);
    TiffParsedIfd* next_ifd = nullptr;
    status = ensureParsedIfd<is_little_endian, variant>(plugin_id, framework, cache, io_stream, next_ifd_offset, &next_ifd);
    if (status != NVIMGCODEC_STATUS_SUCCESS) {
        return status;
    }
    if (next_ifd->next_ifd_offset == 0) {
        cache.root_ifd_offsets_parsed = true;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

// Walks the root IFD chain far enough to resolve the requested image index.
template <bool is_little_endian, TiffVariant variant>
nvimgcodecStatus_t ensureRootIfdOffsetsThrough(
    const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, TiffParserCache& cache, nvimgcodecIoStreamDesc_t* io_stream, size_t requested_index)
{
    auto status = ensureFirstRootIfd<is_little_endian, variant>(plugin_id, framework, cache, io_stream);
    if (status != NVIMGCODEC_STATUS_SUCCESS) {
        return status;
    }

    while (!cache.root_ifd_offsets_parsed && cache.root_ifd_offsets.size() <= requested_index) {
        status = appendNextRootIfd<is_little_endian, variant>(plugin_id, framework, cache, io_stream);
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            return status;
        }
    }

    if (requested_index >= cache.root_ifd_offsets.size()) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Requested image index " << requested_index << " is outside the TIFF IFD chain");
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

// Walks the complete root IFD chain so callers can report the total image count.
template <bool is_little_endian, TiffVariant variant>
nvimgcodecStatus_t ensureAllRootIfdOffsets(
    const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, TiffParserCache& cache, nvimgcodecIoStreamDesc_t* io_stream)
{
    auto status = ensureFirstRootIfd<is_little_endian, variant>(plugin_id, framework, cache, io_stream);
    if (status != NVIMGCODEC_STATUS_SUCCESS) {
        return status;
    }

    while (!cache.root_ifd_offsets_parsed) {
        status = appendNextRootIfd<is_little_endian, variant>(plugin_id, framework, cache, io_stream);
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            return status;
        }
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

// Chooses the IFD backing this view: explicit bitstream offset, image index, or first root IFD.
template <bool is_little_endian, TiffVariant variant>
nvimgcodecStatus_t resolveSelectedIfdOffset(
    const char* plugin_id,
    const nvimgcodecFrameworkDesc_t* framework,
    TiffParserCache& cache,
    nvimgcodecIoStreamDesc_t* io_stream,
    const nvimgcodecCodeStreamView_t* code_stream_view,
    size_t* selected_ifd_offset)
{
    const bool has_code_stream_view = code_stream_view != nullptr;
    const size_t bitstream_offset = code_stream_view ? code_stream_view->bitstream_offset : 0;
    const size_t requested_image_idx = code_stream_view ? code_stream_view->image_idx : 0;

    if (bitstream_offset != 0 && requested_image_idx != 0) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "image_idx cannot be combined with bitstream_offset for TIFF IFD views");
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    if (bitstream_offset != 0) {
        if constexpr (variant == TiffVariant::Classic) {
            if (bitstream_offset > std::numeric_limits<uint32_t>::max()) {
                NVIMGCODEC_LOG_ERROR(framework, plugin_id, "bitstream_offset exceeds 32-bit limit for classic TIFF");
                return NVIMGCODEC_STATUS_INVALID_PARAMETER;
            }
        }
        *selected_ifd_offset = bitstream_offset;
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    if (has_code_stream_view) {
        auto status = ensureRootIfdOffsetsThrough<is_little_endian, variant>(plugin_id, framework, cache, io_stream, requested_image_idx);
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            return status;
        }
        *selected_ifd_offset = cache.root_ifd_offsets[requested_image_idx];
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    auto status = ensureFirstRootIfd<is_little_endian, variant>(plugin_id, framework, cache, io_stream);
    if (status != NVIMGCODEC_STATUS_SUCCESS) {
        return status;
    }
    *selected_ifd_offset = cache.root_ifd_offsets.front();
    return NVIMGCODEC_STATUS_SUCCESS;
}

template<bool is_little_endian, TiffVariant variant>
nvimgcodecStatus_t GetInfoImpl(
    const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, TiffParserCache& cache, nvimgcodecImageInfo_t* info, nvimgcodecIoStreamDesc_t* io_stream, nvimgcodecCodeStreamDesc_t* code_stream)
{
    nvimgcodecCodeStreamInfoTiffExt_t info_tiff_ext{NVIMGCODEC_STRUCTURE_TYPE_TIFF_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfoTiffExt_t), nullptr};
    nvimgcodecCodeStreamInfo_t codestream_info{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), static_cast<void*>(&info_tiff_ext)};
    if (code_stream->getCodeStreamInfo(code_stream->instance, &codestream_info) != NVIMGCODEC_STATUS_SUCCESS) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Could not retrieve code stream information");
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    size_t selected_ifd_offset = info_tiff_ext.ifd_offset;
    if (selected_ifd_offset == 0) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "TIFF selected IFD offset was not resolved");
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    TiffParsedIfd* parsed_ifd = nullptr;
    auto status = ensureParsedIfd<is_little_endian, variant>(plugin_id, framework, cache, io_stream, selected_ifd_offset, &parsed_ifd);
    if (status != NVIMGCODEC_STATUS_SUCCESS) {
        return status;
    }
    copyCachedImageInfo(info, *parsed_ifd);
    return NVIMGCODEC_STATUS_SUCCESS;
}

template<bool is_little_endian, TiffVariant variant>
nvimgcodecStatus_t GetCodeStreamInfoImpl(
    const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, TiffParserCache& cache, nvimgcodecCodeStreamInfo_t* codestream_info, nvimgcodecIoStreamDesc_t* io_stream)
{
    if (codestream_info->struct_type != NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unexpected structure type");
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    strcpy(codestream_info->codec_name, "tiff");

    const bool has_code_stream_view = codestream_info->code_stream_view != nullptr;
    io_stream->size(io_stream->instance, &(codestream_info->size));

    auto* info_tiff_ext = findInfoTiffExt(codestream_info);
    resetTiffExt(info_tiff_ext);

    size_t selected_ifd_offset = 0;
    auto status = resolveSelectedIfdOffset<is_little_endian, variant>(
        plugin_id, framework, cache, io_stream, codestream_info->code_stream_view, &selected_ifd_offset);
    if (status != NVIMGCODEC_STATUS_SUCCESS) {
        return status;
    }
    if (has_code_stream_view) {
        codestream_info->num_images = 1;
    } else {
        status = ensureAllRootIfdOffsets<is_little_endian, variant>(plugin_id, framework, cache, io_stream);
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            return status;
        }
        codestream_info->num_images = cache.root_ifd_offsets.size();
    }

    TiffParsedIfd* parsed_ifd = nullptr;
    status = ensureParsedIfd<is_little_endian, variant>(plugin_id, framework, cache, io_stream, selected_ifd_offset, &parsed_ifd);
    if (status != NVIMGCODEC_STATUS_SUCCESS) {
        return status;
    }
    copyToTiffExt(info_tiff_ext, selected_ifd_offset, *parsed_ifd);

    return NVIMGCODEC_STATUS_SUCCESS;
}

} // namespace

TIFFParserPlugin::TIFFParserPlugin(const nvimgcodecFrameworkDesc_t* framework)
    : framework_(framework)
    , parser_desc_{NVIMGCODEC_STRUCTURE_TYPE_PARSER_DESC, sizeof(nvimgcodecParserDesc_t), nullptr, this, plugin_id_, "tiff", static_can_parse, static_create,
          Parser::static_destroy, Parser::static_get_codestream_info, Parser::static_get_image_info}
{
}

nvimgcodecParserDesc_t* TIFFParserPlugin::getParserDesc()
{
    return &parser_desc_;
}

nvimgcodecStatus_t TIFFParserPlugin::canParse(int* result, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "tiff_parser_can_parse");
        CHECK_NULL(result);
        CHECK_NULL(code_stream);
        nvimgcodecIoStreamDesc_t* io_stream = code_stream->io_stream;
        size_t length;
        io_stream->size(io_stream->instance, &length);
        seekTiffStream(io_stream, 0, SEEK_SET);
        if (length < 4) {
            *result = 0;
            return NVIMGCODEC_STATUS_SUCCESS;
        }
        tiff_magic_t header = ReadValue<tiff_magic_t>(io_stream);
        *result = header == le_header || header == be_header || header == le_bigtiff || header == be_bigtiff;
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if code stream can be parsed - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t TIFFParserPlugin::static_can_parse(void* instance, int* result, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        CHECK_NULL(instance);
        auto handle = reinterpret_cast<TIFFParserPlugin*>(instance);
        return handle->canParse(result, code_stream);
    } catch (const std::exception& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
}

TIFFParserPlugin::Parser::Parser(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework)
    : plugin_id_(plugin_id)
    , framework_(framework)
    , cache_(std::make_shared<TiffParserCache>())
{
    NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "tiff_parser_create");
}

nvimgcodecStatus_t TIFFParserPlugin::create(nvimgcodecParser_t* parser)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "tiff_parser_create");
        CHECK_NULL(parser);
        *parser = reinterpret_cast<nvimgcodecParser_t>(new TIFFParserPlugin::Parser(plugin_id_, framework_));
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create tiff parser - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t TIFFParserPlugin::static_create(void* instance, nvimgcodecParser_t* parser)
{
    try {
        CHECK_NULL(instance);
        auto handle = reinterpret_cast<TIFFParserPlugin*>(instance);
        handle->create(parser);
    } catch (const std::exception& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t TIFFParserPlugin::Parser::static_destroy(nvimgcodecParser_t parser)
{
    try {
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<TIFFParserPlugin::Parser*>(parser);
        delete handle;
    } catch (const std::exception& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

// Detect the stream's endianness + TIFF variant once and cache it. NDPI detection
// rescans the first IFD, so this avoids repeating it for every metadata call / substream.
// Returns nullopt for a non-TIFF header (canParse should have rejected such streams).
static std::optional<TiffParserCache::TiffLayout> ensureTiffLayout(
    const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, TiffParserCache& cache, nvimgcodecIoStreamDesc_t* io_stream)
{
    if (cache.layout.has_value()) {
        return cache.layout;
    }
    seekTiffStream(io_stream, 0, SEEK_SET);
    tiff_magic_t header = ReadValue<tiff_magic_t>(io_stream);
    if (header == le_header) {
        cache.layout = {true, detectLargeNDPI<true>(plugin_id, framework, io_stream) ? TiffVariant::HamamatsuNDPI : TiffVariant::Classic};
    } else if (header == be_header) {
        cache.layout = {false, detectLargeNDPI<false>(plugin_id, framework, io_stream) ? TiffVariant::HamamatsuNDPI : TiffVariant::Classic};
    } else if (header == le_bigtiff) {
        cache.layout = {true, TiffVariant::BigTiff};
    } else if (header == be_bigtiff) {
        cache.layout = {false, TiffVariant::BigTiff};
    } else {
        return std::nullopt;
    }
    return cache.layout;
}

// Invoke `fn.operator()<is_little_endian, variant>()` for the resolved layout. The two
// metadata entry points differ only by which templated *Impl they call, so they share
// this dispatch. (The *Impl functions seek the stream themselves, so they don't depend
// on the position ensureTiffLayout left it in.)
template <typename Fn>
static nvimgcodecStatus_t dispatchTiffLayout(const TiffParserCache::TiffLayout& layout, Fn&& fn)
{
    if (layout.little_endian) {
        switch (layout.variant) {
        case TiffVariant::Classic:       return fn.template operator()<true, TiffVariant::Classic>();
        case TiffVariant::BigTiff:       return fn.template operator()<true, TiffVariant::BigTiff>();
        case TiffVariant::HamamatsuNDPI: return fn.template operator()<true, TiffVariant::HamamatsuNDPI>();
        }
    } else {
        switch (layout.variant) {
        case TiffVariant::Classic:       return fn.template operator()<false, TiffVariant::Classic>();
        case TiffVariant::BigTiff:       return fn.template operator()<false, TiffVariant::BigTiff>();
        case TiffVariant::HamamatsuNDPI: return fn.template operator()<false, TiffVariant::HamamatsuNDPI>();
        }
    }
    return NVIMGCODEC_STATUS_INTERNAL_ERROR;
}

nvimgcodecStatus_t TIFFParserPlugin::Parser::getCodeStreamInfo(nvimgcodecCodeStreamInfo_t* codestream_info, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "tiff_parser_get_codestream_info");
        CHECK_NULL(code_stream);
        CHECK_NULL(codestream_info);
        nvimgcodecIoStreamDesc_t* io_stream = code_stream->io_stream;

        auto layout = ensureTiffLayout(plugin_id_, framework_, *cache_, io_stream);
        if (!layout) {
            // should not happen (because canParse returned result==true)
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Logic error");
            return NVIMGCODEC_STATUS_INTERNAL_ERROR;
        }
        return dispatchTiffLayout(*layout, [&]<bool is_little_endian, TiffVariant variant>() {
            return GetCodeStreamInfoImpl<is_little_endian, variant>(plugin_id_, framework_, *cache_, codestream_info, io_stream);
        });
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not retrieve code stream info from tiff stream - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
}

nvimgcodecStatus_t TIFFParserPlugin::Parser::getImageInfo(nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "tiff_parser_get_image_info");
        CHECK_NULL(code_stream);
        CHECK_NULL(image_info);
        nvimgcodecIoStreamDesc_t* io_stream = code_stream->io_stream;

        auto layout = ensureTiffLayout(plugin_id_, framework_, *cache_, io_stream);
        if (!layout) {
            // should not happen (because canParse returned result==true)
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Logic error");
            return NVIMGCODEC_STATUS_INTERNAL_ERROR;
        }
        return dispatchTiffLayout(*layout, [&]<bool is_little_endian, TiffVariant variant>() {
            return GetInfoImpl<is_little_endian, variant>(plugin_id_, framework_, *cache_, image_info, io_stream, code_stream);
        });
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not retrieve image info from tiff stream - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
}

nvimgcodecStatus_t TIFFParserPlugin::Parser::static_get_codestream_info(
    nvimgcodecParser_t parser, nvimgcodecCodeStreamInfo_t* codestream_info, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<TIFFParserPlugin::Parser*>(parser);
        return handle->getCodeStreamInfo(codestream_info, code_stream);
    } catch (const std::exception& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER; 
    }
}

nvimgcodecStatus_t TIFFParserPlugin::Parser::static_get_image_info(
    nvimgcodecParser_t parser, nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<TIFFParserPlugin::Parser*>(parser);
        return handle->getImageInfo(image_info, code_stream);
    } catch (const std::exception& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
}

class TiffParserExtension
{
  public:
    explicit TiffParserExtension(const nvimgcodecFrameworkDesc_t* framework)
        : framework_(framework)
        , tiff_parser_plugin_(framework)
    {
        framework->registerParser(framework->instance, tiff_parser_plugin_.getParserDesc(), NVIMGCODEC_PRIORITY_NORMAL);
        registerCodecFileExtensions(framework, TIFFParserPlugin::codecName(), TIFFParserPlugin::fileExtensions());
    }
    ~TiffParserExtension()
    {
        unregisterCodecFileExtensions(framework_, TIFFParserPlugin::codecName(), TIFFParserPlugin::fileExtensions());
        framework_->unregisterParser(framework_->instance, tiff_parser_plugin_.getParserDesc());
    }

    static nvimgcodecStatus_t tiff_parser_extension_create(
        void* instance, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework)
    {
        try {
            CHECK_NULL(framework)
            NVIMGCODEC_LOG_TRACE(framework, "tiff_parser_ext", "tiff_parser_extension_create");
            CHECK_NULL(extension)
            *extension = reinterpret_cast<nvimgcodecExtension_t>(new TiffParserExtension(framework));
        } catch (const std::exception& e) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static nvimgcodecStatus_t tiff_parser_extension_destroy(nvimgcodecExtension_t extension)
    {
        try {
            CHECK_NULL(extension)
            auto ext_handle = reinterpret_cast<nvimgcodec::TiffParserExtension*>(extension);
            NVIMGCODEC_LOG_TRACE(ext_handle->framework_, "tiff_parser_ext", "tiff_parser_extension_destroy");
            delete ext_handle;
        } catch (const std::exception& e) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

  private:
    const nvimgcodecFrameworkDesc_t* framework_;
    TIFFParserPlugin tiff_parser_plugin_;
};

// clang-format off
nvimgcodecExtensionDesc_t tiff_parser_extension = {
    NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC,
    sizeof(nvimgcodecExtensionDesc_t),
    NULL,
   
    NULL,
    "tiff_parser_extension",
    NVIMGCODEC_VER,
    NVIMGCODEC_VER,

    TiffParserExtension::tiff_parser_extension_create,
    TiffParserExtension::tiff_parser_extension_destroy
};
// clang-format on

nvimgcodecStatus_t get_tiff_parser_extension_desc(nvimgcodecExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->struct_type != NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = tiff_parser_extension;
    return NVIMGCODEC_STATUS_SUCCESS;
}

} // namespace nvimgcodec
