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

#include "libtiff_decoder.h"
#include <tiffio.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <mutex>

#define NOMINMAX
#include <nvtx3/nvtx3.hpp>
#include "error_handling.h"
#include "log.h"
#include "nvimgcodec.h"

#include "imgproc/color_space_conversion_impl.h"
#include "imgproc/convert.h"
#include "imgproc/out_of_bound_roi_fill.h"
#include "imgproc/region_orientation.h"
#include "imgproc/image_info_checks.h"
#include "imgproc/safe_arithmetic.h"

namespace libtiff {

class DecoderHelper
{
  public:
    explicit DecoderHelper(nvimgcodecIoStreamDesc_t* io_stream)
        : io_stream_(io_stream)
    {
        if (io_stream_->size(io_stream_->instance, &size_) != NVIMGCODEC_STATUS_SUCCESS ||
            size_ > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()))
            throw std::runtime_error("Unable to determine TIFF stream size");
        if (io_stream_->map(io_stream_->instance, &mapped_, 0, size_) != NVIMGCODEC_STATUS_SUCCESS)
            mapped_ = nullptr;
    }

    ~DecoderHelper()
    {
        if (mapped_)
            io_stream_->unmap(io_stream_->instance, mapped_, size_);
    }

    static tmsize_t read(thandle_t handle, void* buffer, tmsize_t n)
    {
        DecoderHelper* helper = reinterpret_cast<DecoderHelper*>(handle);
        if (n <= 0)
            return 0;
        const size_t count = std::min(static_cast<size_t>(n), helper->size_ - helper->position_);
        if (!count)
            return 0;
        size_t read_nbytes = 0;
        if (helper->mapped_) {
            std::memcpy(buffer, static_cast<const uint8_t*>(helper->mapped_) + helper->position_, count);
            read_nbytes = count;
        } else {
            // Unmappable sources still have a shared cursor. Serialize only the
            // seek/read transaction across libtiff readers, not image decoding.
            static std::mutex read_mutex;
            std::lock_guard<std::mutex> lock(read_mutex);
            auto* io = helper->io_stream_;
            if (io->seek(io->instance, helper->position_, SEEK_SET) != NVIMGCODEC_STATUS_SUCCESS ||
                io->read(io->instance, &read_nbytes, buffer, count) != NVIMGCODEC_STATUS_SUCCESS)
                return 0;
        }
        helper->position_ += read_nbytes;
        return read_nbytes;
    }

    static tmsize_t write(thandle_t, void*, tmsize_t)
    {
        // Not used for decoding.
        return 0;
    }

    static toff_t seek(thandle_t handle, toff_t offset, int whence)
    {
        DecoderHelper* helper = reinterpret_cast<DecoderHelper*>(handle);
        size_t base;
        switch (whence) {
        case SEEK_SET: base = 0; break;
        case SEEK_CUR: base = helper->position_; break;
        case SEEK_END: base = helper->size_; break;
        default: return static_cast<toff_t>(-1);
        }
        // libtiff represents negative relative offsets in unsigned toff_t.
        if (whence != SEEK_SET && offset > std::numeric_limits<toff_t>::max() / 2) {
            const toff_t backwards = toff_t{0} - offset;
            if (backwards > base)
                return static_cast<toff_t>(-1);
            helper->position_ = base - backwards;
        } else {
            if (offset > helper->size_ - base)
                return static_cast<toff_t>(-1);
            helper->position_ = base + offset;
        }
        return helper->position_;
    }

    static int map(thandle_t handle, void** base, toff_t* size)
    {
        DecoderHelper* helper = reinterpret_cast<DecoderHelper*>(handle);
        *base = helper->mapped_;
        *size = helper->size_;
        return helper->mapped_ != nullptr; // libtiff expects nonzero on success.
    }

    static void unmap(thandle_t, void*, toff_t)
    {
        // DecoderHelper owns the mapping, including during TIFFClientOpen.
    }

    static toff_t size(thandle_t handle)
    {
        DecoderHelper* helper = reinterpret_cast<DecoderHelper*>(handle);
        return helper->size_;
    }

    static int close(thandle_t handle)
    {
        DecoderHelper* helper = reinterpret_cast<DecoderHelper*>(handle);
        delete helper;
        return 0;
    }

  private:
    nvimgcodecIoStreamDesc_t* io_stream_;
    void* mapped_ = nullptr;
    size_t size_ = 0;
    size_t position_ = 0; // Independent of every other IFD view of this source.
};

std::unique_ptr<TIFF, void (*)(TIFF*)> OpenTiff(nvimgcodecIoStreamDesc_t* io_stream)
{
    DecoderHelper* helper = new DecoderHelper(io_stream);
    TIFF* tiffptr = TIFFClientOpen("", "r", reinterpret_cast<thandle_t>(helper), &DecoderHelper::read, &DecoderHelper::write,
        &DecoderHelper::seek, &DecoderHelper::close, &DecoderHelper::size, &DecoderHelper::map, &DecoderHelper::unmap);
    if (!tiffptr)
        delete helper;
    if (tiffptr == nullptr)
        throw std::runtime_error("Unable to open TIFF image");
    return {tiffptr, &TIFFClose};
}

struct TiffInfo
{
    uint32_t image_width, image_height;
    uint16_t channels;

    uint32_t rows_per_strip;
    uint16_t bit_depth;
    uint16_t key_bit_depth;
    uint16_t orientation;
    uint16_t compression;
    uint16_t photometric_interpretation;
    uint16_t fill_order;

    bool is_tiled;
    bool is_palette;
    bool is_planar;
    bool is_fp;
    bool is_signed;
    struct
    {
        uint16_t *red, *green, *blue;
    } palette;

    uint32_t tile_width, tile_height;
};

TiffInfo GetTiffInfo(TIFF* tiffptr)
{
    TiffInfo info = {};

    LIBTIFF_CALL(TIFFGetField(tiffptr, TIFFTAG_IMAGEWIDTH, &info.image_width));
    LIBTIFF_CALL(TIFFGetField(tiffptr, TIFFTAG_IMAGELENGTH, &info.image_height));
    LIBTIFF_CALL(TIFFGetFieldDefaulted(tiffptr, TIFFTAG_SAMPLESPERPIXEL, &info.channels));
    LIBTIFF_CALL(TIFFGetFieldDefaulted(tiffptr, TIFFTAG_BITSPERSAMPLE, &info.bit_depth));
    LIBTIFF_CALL(TIFFGetFieldDefaulted(tiffptr, TIFFTAG_ORIENTATION, &info.orientation));
    LIBTIFF_CALL(TIFFGetFieldDefaulted(tiffptr, TIFFTAG_COMPRESSION, &info.compression));
    LIBTIFF_CALL(TIFFGetFieldDefaulted(tiffptr, TIFFTAG_ROWSPERSTRIP, &info.rows_per_strip));
    LIBTIFF_CALL(TIFFGetFieldDefaulted(tiffptr, TIFFTAG_FILLORDER, &info.fill_order));

    uint16_t sample_format;
    LIBTIFF_CALL(TIFFGetFieldDefaulted(tiffptr, TIFFTAG_SAMPLEFORMAT, &sample_format));
    info.is_fp = false;
    info.is_signed = false;
    if (sample_format == SAMPLEFORMAT_IEEEFP) {
        info.is_fp = true;
    } else if (sample_format == SAMPLEFORMAT_INT) {
        info.is_signed = true;
    } else if (sample_format != SAMPLEFORMAT_UINT) {
        throw std::runtime_error("Unsupported sample format: " + std::to_string(sample_format));
    }

    info.is_tiled = TIFFIsTiled(tiffptr);
    if (info.is_tiled) {
        LIBTIFF_CALL(TIFFGetField(tiffptr, TIFFTAG_TILEWIDTH, &info.tile_width));
        LIBTIFF_CALL(TIFFGetField(tiffptr, TIFFTAG_TILELENGTH, &info.tile_height));
    } else {
        // We will be reading data line-by-line and pretend that lines are tiles
        info.tile_width = info.image_width;
        info.tile_height = 1;
    }

    if (TIFFGetField(tiffptr, TIFFTAG_PHOTOMETRIC, &info.photometric_interpretation)) {
        info.is_palette = (info.photometric_interpretation == PHOTOMETRIC_PALETTE);
    } else {
        info.photometric_interpretation = PHOTOMETRIC_MINISBLACK;
    }

    uint16_t planar_config;
    LIBTIFF_CALL(TIFFGetFieldDefaulted(tiffptr, TIFFTAG_PLANARCONFIG, &planar_config));
    info.is_planar = (planar_config == PLANARCONFIG_SEPARATE);

    if (info.is_palette) {
        LIBTIFF_CALL(TIFFGetField(tiffptr, TIFFTAG_COLORMAP, &info.palette.red, &info.palette.green, &info.palette.blue));
        info.channels = 3; // Palette is always RGB
        info.key_bit_depth = info.bit_depth;
        info.bit_depth = 16; // Palette colors are always 16 bits
        // Palette pixels store an index into the colormap, not a color sample, so
        // TIFFTAG_SAMPLEFORMAT (which describes the color values) doesn't apply to
        // the index width. Forcing this keeps TiffConvert's unpacked index in an
        // unsigned type; treating it as signed would let a crafted TIFF turn a
        // large index negative and index the colormap arrays out of bounds.
        info.is_signed = false;
    }

    return info;
}

bool IsJpegYcbcr(const TiffInfo& info)
{
    return info.photometric_interpretation == PHOTOMETRIC_YCBCR && info.compression == COMPRESSION_JPEG;
}

bool IsSupportedPhotometric(const TiffInfo& info)
{
    return info.photometric_interpretation == PHOTOMETRIC_RGB ||
           info.photometric_interpretation == PHOTOMETRIC_MINISBLACK ||
           info.photometric_interpretation == PHOTOMETRIC_PALETTE ||
           IsJpegYcbcr(info);
}

template <int depth>
struct depth2type;

template <>
struct depth2type<8>
{
    using type = uint8_t;
};

template <>
struct depth2type<16>
{
    using type = uint16_t;
};
template <>
struct depth2type<32>
{
    using type = uint32_t;
};

/**
 * @brief Unpacks packed bits and/or converts palette data to RGB.
 *
 * @tparam OutputType Required output type
 * @tparam normalize If true, values will be upscaled to OutputType's dynamic range
 * @param nbits Number of bits per value
 * @param out Output array
 * @param in Pointer to the bits to unpack
 * @param n Number of input values to unpack
 */
template <typename OutputType, bool normalize = true>
void TiffConvert(const TiffInfo& info, OutputType* out, const void* in, size_t n)
{
    // We don't care about endianness here, because we read byte-by-byte and:
    // 1) "The library attempts to hide bit- and byte-ordering differences between the image and the
    //    native machine by converting data to the native machine order."
    //    http://www.libtiff.org/man/TIFFReadScanline.3t.html
    // 2) We only support FILL_ORDER=1 (i.e. big endian), which is TIFF's default and the only fill
    //    order required in Baseline TIFF readers.
    //    https://www.awaresystems.be/imaging/tiff/tifftags/fillorder.html

    size_t nbits = info.is_palette ? info.key_bit_depth : info.bit_depth;
    size_t out_type_bits = 8 * sizeof(OutputType);
    if (out_type_bits < nbits)
        throw std::logic_error("Unpacking bits failed: OutputType too small");
    if (n == 0)
        return;

    auto in_ptr = static_cast<const uint8_t*>(in);

    if (nbits == out_type_bits) {
        // The sample width exactly matches the output container width (e.g. a 16-bit
        // palette index unpacked into a 16-bit buffer). TIFFReadScanline already converts
        // such whole, byte-aligned samples to the host's native byte order, so read them
        // directly instead of running them through the MSB-first bit-unpacking below. That
        // path is meant for sub-byte-packed data (where FILLORDER_MSB2LSB genuinely governs
        // bit order) and would otherwise re-order the bytes of a full-width multi-byte
        // sample on a little-endian host.
        for (size_t i = 0; i < n; i++) {
            OutputType result = 0;
            std::memcpy(&result, in_ptr, sizeof(OutputType));
            in_ptr += sizeof(OutputType);
            if (info.is_palette) {
                using nvimgcodec::ConvertNorm;
                out[3 * i + 0] = ConvertNorm<OutputType>(info.palette.red[result]);
                out[3 * i + 1] = ConvertNorm<OutputType>(info.palette.green[result]);
                out[3 * i + 2] = ConvertNorm<OutputType>(info.palette.blue[result]);
            } else {
                out[i] = result; // widths match exactly, so no normalization scaling is needed
            }
        }
        return;
    }

    uint8_t buffer = *(in_ptr++);
    constexpr size_t buffer_capacity = 8 * sizeof(buffer);
    size_t bits_in_buffer = buffer_capacity;

    for (size_t i = 0; i < n; i++) {
        OutputType result = 0;
        size_t bits_to_read = nbits;
        while (bits_to_read > 0) {
            if (bits_in_buffer >= bits_to_read) {
                // If we have enough bits in the buffer, we store them and finish
                if (31 < nbits)
                    throw std::logic_error("Unpacking bits failed: Too many bits to read");
                result <<= bits_to_read;
                result |= buffer >> (buffer_capacity - bits_to_read);
                bits_in_buffer -= bits_to_read;
                buffer <<= bits_to_read;
                bits_to_read = 0;
            } else {
                // If we don't have enough bits, we store what we have and refill the buffer
                result <<= bits_in_buffer;
                result |= buffer >> (buffer_capacity - bits_in_buffer);
                bits_to_read -= bits_in_buffer;
                buffer = *(in_ptr++);
                bits_in_buffer = buffer_capacity;
            }
        }
        if (info.is_palette) {
            using nvimgcodec::ConvertNorm;
            out[3 * i + 0] = ConvertNorm<OutputType>(info.palette.red[result]);
            out[3 * i + 1] = ConvertNorm<OutputType>(info.palette.green[result]);
            out[3 * i + 2] = ConvertNorm<OutputType>(info.palette.blue[result]);
        } else {
            if (normalize) {
                double coeff = static_cast<double>((1ull << out_type_bits) - 1) / ((1ull << nbits) - 1);
                result *= coeff;
            }
            out[i] = result;
        }
    }
}

struct DecoderImpl
{
    DecoderImpl(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params);
    ~DecoderImpl();

    static nvimgcodecStatus_t static_destroy(nvimgcodecDecoder_t decoder);

    nvimgcodecStatus_t getMetadata(const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count) const;
    static nvimgcodecStatus_t static_get_metadata(nvimgcodecDecoder_t decoder, const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count)
    {
        try {
            XM_CHECK_NULL(decoder);
            auto handle = reinterpret_cast<DecoderImpl*>(decoder);
            return handle->getMetadata(code_stream, metadata, metadata_count);
        } catch (const std::exception& e) {
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
        }
    }

    nvimgcodecProcessingStatus_t canDecode(
        const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx);
    static nvimgcodecProcessingStatus_t static_can_decode(nvimgcodecDecoder_t decoder, const nvimgcodecImageDesc_t* image,
        const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx)
    {
        try {
            XM_CHECK_NULL(decoder);
            auto handle = reinterpret_cast<DecoderImpl*>(decoder);
            return handle->canDecode(image, code_stream, params, thread_idx);
        } catch (const std::exception& e) {
            return NVIMGCODEC_PROCESSING_STATUS_FAIL;
        }
    }

    nvimgcodecStatus_t decode(
        const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx);
    static nvimgcodecStatus_t static_decode_sample(nvimgcodecDecoder_t decoder, const nvimgcodecImageDesc_t* image,
        const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx)
    {
        try {
            XM_CHECK_NULL(decoder);
            auto handle = reinterpret_cast<DecoderImpl*>(decoder);
            return handle->decode(image, code_stream, params, thread_idx);
        } catch (const std::exception& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

    const char* plugin_id_;
    const nvimgcodecFrameworkDesc_t* framework_;
    const nvimgcodecExecutionParams_t* exec_params_;
};

LibtiffDecoderPlugin::LibtiffDecoderPlugin(const nvimgcodecFrameworkDesc_t* framework)
    : decoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC, sizeof(nvimgcodecDecoderDesc_t), NULL, this, plugin_id_, "tiff",
          NVIMGCODEC_BACKEND_KIND_CPU_ONLY, static_create, DecoderImpl::static_destroy, DecoderImpl::static_get_metadata, DecoderImpl::static_can_decode,
          DecoderImpl::static_decode_sample, nullptr}
    , framework_(framework)
{
}

nvimgcodecDecoderDesc_t* LibtiffDecoderPlugin::getDecoderDesc()
{
    return &decoder_desc_;
}

nvimgcodecStatus_t DecoderImpl::getMetadata(const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count) const
{
    return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
}

nvimgcodecProcessingStatus_t DecoderImpl::canDecode(
    const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx)
{
    (void) thread_idx;
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "can_decode ");
        XM_CHECK_NULL(code_stream);
        XM_CHECK_NULL(params);

        XM_CHECK_NULL(image);
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        auto ret = image->getImageInfo(image->instance, &image_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS)
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;

        nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        code_stream->getImageInfo(code_stream->instance, &cs_image_info);

        bool is_tiff = strcmp(cs_image_info.codec_name, "tiff") == 0;
        if (!is_tiff) {
            return NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
        }

        nvimgcodecCodeStreamInfo_t codestream_info{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr};
        ret = code_stream->getCodeStreamInfo(code_stream->instance, &codestream_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS)
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;


        nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
        if (codestream_info.code_stream_view) {
            // image_idx is supported via TIFFSetDirectory
            // bitstream_offset is supported via TIFFSetSubDirectory

            const auto& region = codestream_info.code_stream_view->region;
            if (region.ndim == 0) {
                // no ROI, okay
            } else if (region.ndim == 2) {
                if (nvimgcodec::is_region_out_of_bounds_effective(region, image_info.orientation,
                        cs_image_info.plane_info[0].width, cs_image_info.plane_info[0].height,
                        params->apply_exif_orientation)) {
                    NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "Out of bounds region is not supported.");
                    status |= NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED;
                }
            } else {
                status |= NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED;
                NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "Region decoding is supported only for 2 dimensions.");
            }
        }

        switch (image_info.sample_format) {
        case NVIMGCODEC_SAMPLEFORMAT_P_YUV:
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
            break;
        case NVIMGCODEC_SAMPLEFORMAT_I_BGR:
        case NVIMGCODEC_SAMPLEFORMAT_I_RGB:
        case NVIMGCODEC_SAMPLEFORMAT_P_BGR:
        case NVIMGCODEC_SAMPLEFORMAT_P_RGB:
        case NVIMGCODEC_SAMPLEFORMAT_P_Y:
        case NVIMGCODEC_SAMPLEFORMAT_I_Y:
        case NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED:
        case NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED:
        default:
            break; // supported
        }

        if (image_info.plane_info[0].precision != (image_info.plane_info[0].sample_type >> 8) &&
            image_info.plane_info[0].precision != 0 &&
            image_info.plane_info[0].precision != 1)
        {
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
            NVIMGCODEC_LOG_INFO(
                framework_,
                plugin_id_,
                "libTIFF extension can only decode to a type that has full precision (like 8 bits, not 6)."
            );
        }

        if (!nvimgcodec::check_planes_consistency(framework_, plugin_id_, image_info)) {
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
        }

        if (image_info.num_planes == 1) {
            if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_BGR || image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_RGB)
                status |= NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
        } else if (image_info.num_planes > 1) {
            if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_BGR || image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_RGB ||
                image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED)
                status |= NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
        }
        if (image_info.num_planes != 1 && image_info.num_planes != 3 && image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED)
            status |= NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;

        if (image_info.plane_info[0].num_channels == 1) {
            if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_BGR || image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_RGB)
                status |= NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
        } else if (image_info.plane_info[0].num_channels > 1) {
            if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_BGR || image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_RGB ||
                image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED)
                status |= NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
        }
        if (image_info.plane_info[0].num_channels != 1 && image_info.plane_info[0].num_channels != 3 &&
            image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED)
            status |= NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;

        // This codec doesn't apply EXIF orientation
        if (params->apply_exif_orientation &&
            (image_info.orientation.flip_x || image_info.orientation.flip_y || image_info.orientation.rotated != 0)) {
            status |= NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED;
        }
        return status;
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if libtiff can decode - " << e.what());
        return NVIMGCODEC_PROCESSING_STATUS_FAIL;
    }
}

DecoderImpl::DecoderImpl(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params)
    : plugin_id_(plugin_id)
    , framework_(framework)
    , exec_params_(exec_params)
{
}

nvimgcodecStatus_t LibtiffDecoderPlugin::create(
    nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "libtiff_create");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(exec_params);
        *decoder = reinterpret_cast<nvimgcodecDecoder_t>(new DecoderImpl(plugin_id_, framework_, exec_params));
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create libtiff decoder - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t LibtiffDecoderPlugin::static_create(
    void* instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    if (!instance) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }

    auto handle = reinterpret_cast<LibtiffDecoderPlugin*>(instance);
    return handle->create(decoder, exec_params, options);
}

DecoderImpl::~DecoderImpl()
{
    NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "libtiff_destroy");
}

nvimgcodecStatus_t DecoderImpl::static_destroy(nvimgcodecDecoder_t decoder)
{
    try {
        auto handle = reinterpret_cast<DecoderImpl*>(decoder);
        delete handle;
    } catch (const std::exception& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }

    return NVIMGCODEC_STATUS_SUCCESS;
}

template <typename Output, typename Input>
nvimgcodecProcessingStatus_t decodeImplTyped2(
    const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework,  const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecImageInfo_t& image_info, const nvimgcodecDecodeParams_t* params, TIFF* tiff, const TiffInfo& info)
{
    if (!IsSupportedPhotometric(info)) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id,
            "Unsupported photometric interpretation: " << info.photometric_interpretation <<
            ", compression: " << info.compression);
        return NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED;
    }

    if (IsJpegYcbcr(info)) {
        if (!TIFFSetField(tiff, TIFFTAG_JPEGCOLORMODE, JPEGCOLORMODE_RGB)) {
            NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Could not configure libtiff JPEG YCbCr RGB conversion");
            return NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED;
        }
    }

    if (info.is_planar) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Planar TIFFs are not supported");
        return NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED;
    }

    if (info.bit_depth > 32) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unsupported bit depth: " << info.bit_depth);
        return NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED;
    }

    if (info.is_signed && info.bit_depth != 8 && info.bit_depth != 16 && info.bit_depth != 32) {
        // TiffConvert's bit-unpacking/normalization path assumes unsigned samples, so
        // sub-byte-aligned or otherwise non-native signed bit depths (e.g. 12-bit signed)
        // can't be decoded correctly. Only full-width signed samples take the direct
        // (no-unpack) path below.
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unsupported bit depth for signed-integer TIFF: " << info.bit_depth);
        return NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED;
    }

    if (info.is_tiled && (info.tile_width % 16 != 0 || info.tile_height % 16 != 0)) {
        // http://www.libtiff.org/libtiff.html
        // (...) tile width and length must each be a multiple of 16 pixels
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "TIFF tile dimensions must be a multiple of 16");
        return NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED;
    }

    if (info.is_tiled && (info.bit_depth != 8 && info.bit_depth != 16 && info.bit_depth != 32)) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unsupported bit depth in tiled TIFF: " << info.bit_depth);
        return NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED;
    }

    // Other fill orders are rare and discouraged by TIFF specification, but can happen
    if (info.fill_order != FILLORDER_MSB2LSB) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Only FILL_ORDER=1 is supported");
        return NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED;
    }

    size_t buf_nbytes;
    if (!info.is_tiled) {
        buf_nbytes = TIFFScanlineSize(tiff);
    } else {
        buf_nbytes = TIFFTileSize(tiff);
    }

    std::unique_ptr<void, void (*)(void*)> buf{_TIFFmalloc(buf_nbytes), _TIFFfree};
    if (buf.get() == nullptr)
        throw std::runtime_error("Could not allocate memory");

    int num_channels;
    bool planar;
    switch (image_info.sample_format) {
    case NVIMGCODEC_SAMPLEFORMAT_I_RGB:
    case NVIMGCODEC_SAMPLEFORMAT_I_BGR:
        num_channels = 3;
        planar = false;
        break;
    case NVIMGCODEC_SAMPLEFORMAT_P_RGB:
    case NVIMGCODEC_SAMPLEFORMAT_P_BGR:
        num_channels = 3;
        planar = true;
        break;
    case NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED:
        num_channels = info.channels;
        planar = false;
        break;
    case NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED:
        num_channels = info.channels;
        planar = true;
        break;
    case NVIMGCODEC_SAMPLEFORMAT_P_Y:
    case NVIMGCODEC_SAMPLEFORMAT_I_Y:
        num_channels = 1;
        planar = true;
        break;
    default:
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unsupported sample_format: " << image_info.sample_format);
        return NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
    }

    if (image_info.num_planes > info.channels && !(info.channels == 1 && image_info.num_planes == 3)) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Invalid number of planes: " << image_info.num_planes);
        return NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
    }
    // The interleaved copy loops read plane_info[0].num_channels samples out of each source pixel,
    // which only holds info.channels of them. num_planes alone does not bound this (interleaved
    // images have num_planes == 1), and image_info comes from the caller while info comes from the
    // selected IFD, so the two can disagree - e.g. a multi-IFD TIFF whose IFD 0 advertises 3
    // channels while the IFD actually decoded has 1.
    if (info.channels != 1 && image_info.plane_info[0].num_channels > info.channels) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id,
            "Requested " << image_info.plane_info[0].num_channels << " channels but the TIFF directory has only " << info.channels);
        return NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
    }
    nvimgcodecCodeStreamInfo_t codestream_info{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr, nullptr};
    auto ret = code_stream->getCodeStreamInfo(code_stream->instance, &codestream_info);
    if (ret != NVIMGCODEC_STATUS_SUCCESS) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Could not retrieve code stream information");
        return NVIMGCODEC_PROCESSING_STATUS_FAIL;
    }

    int64_t region_start_y = 0;
    int64_t region_start_x = 0;
    int64_t region_end_y = image_info.plane_info[0].height;
    int64_t region_end_x = image_info.plane_info[0].width;

    if (codestream_info.code_stream_view) {
        auto region = codestream_info.code_stream_view->region;
        if (region.ndim != 0) {
  
            region_start_y = region.start[0];
            region_start_x = region.start[1];
            region_end_y = region.end[0];
            region_end_x = region.end[1];
        }
    }

    int64_t stride_y = image_info.plane_info[0].row_stride / sizeof(Output);
    int64_t stride_x = planar ? 1 : num_channels;
    int64_t tile_stride_y = int64_t(info.tile_width) * info.channels;
    int64_t tile_stride_x = info.channels;

    const bool allow_random_row_access = (info.compression == COMPRESSION_NONE || info.rows_per_strip == 1);
    // If random access is not allowed, need to read sequentially all previous rows
    // From: http://www.libtiff.org/man/TIFFReadScanline.3t.html
    // Compression algorithm does not support random access. Data was requested in a non-sequential
    // order from a file that uses a compression algorithm and that has RowsPerStrip greater than
    // one. That is, data in the image is stored in a compressed form, and with multiple rows packed
    // into a strip. In this case, the library does not support random access to the data. The data
    // should either be accessed sequentially, or the file should be converted so that each strip is
    // made up of one row of data.
    if (!info.is_tiled && !allow_random_row_access) {
        // Need to read sequentially since not all the images support random access
        // If random access is not allowed, need to read sequentially all previous rows
        for (int64_t y = 0; y < region_start_y; y++) {
            LIBTIFF_CALL(TIFFReadScanline(tiff, buf.get(), y, 0));
        }
    }

    bool convert_needed = info.bit_depth != (sizeof(Input) * 8) || info.is_palette;

    Input* in;
    std::vector<uint8_t> scratch;
    size_t tile_element_count = 0;
    if (!convert_needed) {
        in = static_cast<Input*>(buf.get());
    } else {
        // tile_height * tile_width * channels was originally computed in uint32
        // arithmetic and could wrap when fed crafted TIFF metadata, producing an
        // undersized scratch buffer that later writes overrun. Compute it in
        // size_t with overflow checks so the wrap can no longer occur. Both this
        // count and the byte-size derived from it are only consumed below in the
        // conversion branch, so the checks live here.
        if (!nvimgcodec::SafeMul3SizeT(info.tile_height, info.tile_width, info.channels, tile_element_count)) {
            NVIMGCODEC_LOG_ERROR(framework, plugin_id,
                "Overflow computing TIFF tile element count from tile_height=" << info.tile_height
                << ", tile_width=" << info.tile_width << ", channels=" << info.channels);
            return NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED;
        }
        size_t scratch_nbytes = 0;
        if (!nvimgcodec::SafeMulSizeT(tile_element_count, sizeof(Input), scratch_nbytes)) {
            NVIMGCODEC_LOG_ERROR(framework, plugin_id,
                "Overflow computing TIFF tile scratch byte count from element_count="
                << tile_element_count << ", elem_size=" << sizeof(Input));
            return NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED;
        }
        scratch.resize(scratch_nbytes);
        in = reinterpret_cast<Input*>(scratch.data());
    }

    Output* img_out = reinterpret_cast<Output*>(image_info.buffer);

    // For non-tiled TIFFs first_tile_x is always 0, because the scanline spans the whole image.
    int64_t first_tile_y = region_start_y - region_start_y % info.tile_height;
    int64_t first_tile_x = region_start_x - region_start_x % info.tile_width;

    for (int64_t tile_y = first_tile_y; tile_y < region_end_y; tile_y += info.tile_height) {
        for (int64_t tile_x = first_tile_x; tile_x < region_end_x; tile_x += info.tile_width) {
            int64_t tile_begin_y = std::max(tile_y, region_start_y);
            int64_t tile_begin_x = std::max(tile_x, region_start_x);
            int64_t tile_end_y = std::min(tile_y + info.tile_height, region_end_y);
            int64_t tile_end_x = std::min(tile_x + info.tile_width, region_end_x);
            int64_t tile_size_y = tile_end_y - tile_begin_y;
            int64_t tile_size_x = tile_end_x - tile_begin_x;

            if (info.is_tiled) {
                auto ret = TIFFReadTile(tiff, buf.get(), tile_x, tile_y, 0, 0);
                if (ret <= 0) {
                    throw std::runtime_error("TIFFReadTile failed");
                }
            } else {
                LIBTIFF_CALL(TIFFReadScanline(tiff, buf.get(), tile_y, 0));
            }

            if (convert_needed) {
                if constexpr (std::is_same_v<Input, float>) {
                    NVIMGCODEC_LOG_ERROR(
                        framework, plugin_id, "Conversion is not supported for FP32 image.");
                    return NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED;
                } else {
                    // tile_element_count was computed once above with overflow
                    // checking; reuse it here so the decode and convert paths
                    // can never disagree on the buffer footprint.
                    size_t input_values = tile_element_count;
                    if (info.is_palette)
                        input_values /= info.channels;
                    TiffConvert(info, in, buf.get(), input_values);
                }
            }

            Output* dst = img_out + (tile_begin_y - region_start_y) * stride_y + (tile_begin_x - region_start_x) * stride_x;
            const Input* src = in + (tile_begin_y - tile_y) * tile_stride_y + (tile_begin_x - tile_x) * tile_stride_x;

            using nvimgcodec::ConvertSatNorm;
            using nvimgcodec::rgb_to_gray;
            using nvimgcodec::vec;
            switch (image_info.sample_format) {
            case NVIMGCODEC_SAMPLEFORMAT_P_Y:
            case NVIMGCODEC_SAMPLEFORMAT_I_Y:
                if (info.channels == 1) {
                    auto* plane = dst;
                    for (uint32_t i = 0; i < tile_size_y; i++) {
                        auto* row = plane + i * stride_y;
                        auto* tile_row = src + i * tile_stride_y;
                        for (uint32_t j = 0; j < tile_size_x; j++) {
                            *(row + j * stride_x) = ConvertSatNorm<Output>(*(tile_row + j * tile_stride_x));
                        }
                    }
                } else if (info.channels >= 3) {
                    // Per-plane offset / stride. The C image_info contract
                    // allows per-plane dimensions (e.g. subsampled chroma),
                    // and an external reuse buffer may report per-plane row
                    // strides; build dst offsets from each plane's own
                    // row_stride * height (overflow checked).
                    size_t plane_offsets_bytes[NVIMGCODEC_MAX_NUM_PLANES];
                    if (!nvimgcodec::SafePlaneByteOffsets(image_info, plane_offsets_bytes)) {
                        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Plane byte offsets overflow.");
                        return NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED;
                    }
                    for (uint32_t c = 0; c < image_info.num_planes; c++) {
                        const auto& plane_info_c = image_info.plane_info[c];
                        const int64_t plane_stride_y =
                            static_cast<int64_t>(plane_info_c.row_stride) / sizeof(Output);
                        // Position the tile inside plane c using that plane's own
                        // row stride (not plane 0's, which is baked into `dst`).
                        // Grayscale output is single-plane today, but compute it
                        // the multi-plane-correct way so this stays right if the
                        // plane count ever grows.
                        auto* plane = reinterpret_cast<Output*>(
                                          reinterpret_cast<uint8_t*>(img_out) + plane_offsets_bytes[c])
                                      + (tile_begin_y - region_start_y) * plane_stride_y
                                      + (tile_begin_x - region_start_x) * stride_x;
                        for (uint32_t i = 0; i < tile_size_y; i++) {
                            auto* row = plane + i * plane_stride_y;
                            auto* tile_row = src + i * tile_stride_y;
                            for (uint32_t j = 0; j < tile_size_x; j++) {
                                auto* pixel = tile_row + j * tile_stride_x;
                                auto* out_pixel = row + j * stride_x;
                                auto r = *(pixel + 0);
                                auto g = *(pixel + 1);
                                auto b = *(pixel + 2);
                                *(out_pixel) = rgb_to_gray<Output>(vec<3, Input>(r, g, b));
                            }
                        }
                    }
                } else {
                    NVIMGCODEC_LOG_ERROR(
                        framework, plugin_id, "Unexpected number of channels for conversion to grayscale: " << info.channels);
                    return NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
                }
                break;
            case NVIMGCODEC_SAMPLEFORMAT_P_RGB:
            case NVIMGCODEC_SAMPLEFORMAT_P_BGR:
            case NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED: {
                // Per-plane destination offsets and row strides. The running
                // sum of row_stride * height keeps the layout correct for any
                // per-plane dimensions the C image_info contract allows
                // (overflow checked).
                size_t plane_offsets_bytes[NVIMGCODEC_MAX_NUM_PLANES];
                if (!nvimgcodec::SafePlaneByteOffsets(image_info, plane_offsets_bytes)) {
                    NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Plane byte offsets overflow.");
                    return NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED;
                }
                for (uint32_t c = 0; c < image_info.num_planes; c++) {
                    uint32_t dst_p = c;
                    if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_BGR)
                        dst_p = c == 2 ? 0 : c == 0 ? 2 : c;
                    uint32_t source_channel = info.channels == 1 ? 0 : c;
                    const int64_t plane_stride_y =
                        static_cast<int64_t>(image_info.plane_info[dst_p].row_stride) / sizeof(Output);
                    // Position the tile inside plane dst_p using that plane's own
                    // row stride. `dst` bakes in the vertical offset scaled by
                    // plane 0's stride_y, which is wrong here when planes have
                    // different row strides, so compute the plane pointer from
                    // img_out: plane start + per-plane vertical offset + the
                    // horizontal offset (stride_x == 1 for planar).
                    auto* plane = reinterpret_cast<Output*>(
                                      reinterpret_cast<uint8_t*>(img_out) + plane_offsets_bytes[dst_p])
                                  + (tile_begin_y - region_start_y) * plane_stride_y
                                  + (tile_begin_x - region_start_x) * stride_x;
                    for (uint32_t i = 0; i < tile_size_y; i++) {
                        auto* row = plane + i * plane_stride_y;
                        auto* tile_row = src + i * tile_stride_y;
                        for (uint32_t j = 0; j < tile_size_x; j++) {
                            *(row + j * stride_x) = ConvertSatNorm<Output>(*(tile_row + j * tile_stride_x + source_channel));
                        }
                    }
                }
            } break;

            case NVIMGCODEC_SAMPLEFORMAT_I_RGB:
            case NVIMGCODEC_SAMPLEFORMAT_I_BGR:
            case NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED: {
                for (uint32_t i = 0; i < tile_size_y; i++) {
                    auto* row = dst + i * stride_y;
                    auto* tile_row = src + i * tile_stride_y;
                    for (uint32_t j = 0; j < tile_size_x; j++) {
                        auto* pixel = row + j * stride_x;
                        auto* tile_pixel = tile_row + j * tile_stride_x;
                        if (info.channels == 1) {
                            for (uint32_t c = 0; c < image_info.plane_info[0].num_channels; c++) {
                                *(pixel + c) = ConvertSatNorm<Output>(*tile_pixel);
                            }
                        } else {
                            assert(info.channels >= image_info.plane_info[0].num_channels);
                            for (uint32_t c = 0; c < image_info.plane_info[0].num_channels; c++) {
                                uint32_t out_c = c;
                                if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_BGR)
                                    out_c = c == 2 ? 0 : c == 0 ? 2 : c;
                                *(pixel + out_c) = ConvertSatNorm<Output>(*(tile_pixel + c));
                            }
                        }
                    }
                }
            } break;

            case NVIMGCODEC_SAMPLEFORMAT_P_YUV:
            default:
                NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unsupported sample_format: " << image_info.sample_format);
                return NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
            }
        }
    }
    return NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
}

template <typename Output>
nvimgcodecProcessingStatus_t decodeImplTyped(
    const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecImageInfo_t& image_info, const nvimgcodecDecodeParams_t* params, TIFF* tiff, const TiffInfo& info)
{
    if (info.is_fp) {
        if (info.bit_depth == 32) {
            return decodeImplTyped2<Output, float>(plugin_id, framework, code_stream, image_info, params, tiff, info);
        }

        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unsupported bit depth for FP image: " << info.bit_depth);
        return NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
    }
    else if (info.is_signed) {
        if (info.bit_depth <= 8) {
            return decodeImplTyped2<Output, int8_t>(plugin_id, framework, code_stream, image_info, params, tiff, info);
        } else if (info.bit_depth <= 16) {
            return decodeImplTyped2<Output, int16_t>(plugin_id, framework, code_stream, image_info, params, tiff, info);
        } else if (info.bit_depth <= 32) {
            return decodeImplTyped2<Output, int32_t>(plugin_id, framework, code_stream, image_info, params, tiff, info);
        } else {
            NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unsupported bit depth: " << info.bit_depth);
            return NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
        }
    }
    else if (info.bit_depth <= 8) {
        return decodeImplTyped2<Output, uint8_t>(plugin_id, framework, code_stream, image_info, params, tiff, info);
    } else if (info.bit_depth <= 16) {
        return decodeImplTyped2<Output, uint16_t>(plugin_id, framework, code_stream, image_info, params, tiff, info);
    } else if (info.bit_depth <= 32) {
        return decodeImplTyped2<Output, uint32_t>(plugin_id, framework, code_stream, image_info, params, tiff, info);
    } else {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unsupported bit depth: " << info.bit_depth);
        return NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
    }
}

nvimgcodecStatus_t DecoderImpl::decode(
    const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx)
{
    (void) thread_idx;
    try {
        XM_CHECK_NULL(image);
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        auto ret = image->getImageInfo(image->instance, &image_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not retrieve image information");
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
            return ret;
        }
        XM_CHECK_NULL(code_stream);
        nvimgcodecCodeStreamInfoTiffExt_t tiff_ext{
            NVIMGCODEC_STRUCTURE_TYPE_TIFF_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfoTiffExt_t), nullptr};
        nvimgcodecCodeStreamInfo_t codestream_info{
            NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), &tiff_ext};
        ret = code_stream->getCodeStreamInfo(code_stream->instance, &codestream_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not retrieve code stream information");
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
            return ret;
        }

        if (image_info.buffer_kind != NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected buffer kind");
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }

        auto tiff = OpenTiff(code_stream->io_stream);

        // Extract image_idx and bitstream_offset from code_stream_view
        size_t image_idx = 0;
        size_t bitstream_offset = 0;
        if (codestream_info.code_stream_view) {
            image_idx = codestream_info.code_stream_view->image_idx;
            bitstream_offset = codestream_info.code_stream_view->bitstream_offset;
        }

        if (bitstream_offset != 0 && image_idx != 0) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "image_idx cannot be combined with bitstream_offset for TIFF IFD views");
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }

        // Navigate to the selected IFD. The TIFF parser resolves image_idx and
        // bitstream_offset views to an absolute IFD offset when the extension is requested.
        if (codestream_info.code_stream_view && tiff_ext.ifd_offset != 0) {
            if (!TIFFSetSubDirectory(tiff.get(), tiff_ext.ifd_offset)) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Failed to set TIFF subdirectory at offset " << tiff_ext.ifd_offset);
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }
        } else if (bitstream_offset != 0) {
            if (!TIFFSetSubDirectory(tiff.get(), bitstream_offset)) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Failed to set TIFF subdirectory at offset " << bitstream_offset);
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }
        } else if (image_idx != 0) {
            if (image_idx >= codestream_info.num_images) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Image index " << image_idx << " out of range (0-" << (codestream_info.num_images - 1) << ")");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_NUM_IMAGES_UNSUPPORTED);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }
            if (!TIFFSetDirectory(tiff.get(), static_cast<tdir_t>(image_idx))) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Failed to set TIFF directory " << image_idx);
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }
        }

        auto info = GetTiffInfo(tiff.get());
        nvimgcodecProcessingStatus_t res;
        switch (image_info.plane_info[0].sample_type) {
        case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8:
            res = decodeImplTyped<uint8_t>(plugin_id_, framework_, code_stream, image_info, params, tiff.get(), info);
            break;
        case NVIMGCODEC_SAMPLE_DATA_TYPE_INT8:
            res = decodeImplTyped<int8_t>(plugin_id_, framework_, code_stream, image_info, params, tiff.get(), info);
            break;
        case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16:
            res = decodeImplTyped<uint16_t>(plugin_id_, framework_, code_stream, image_info, params,tiff.get(), info);
            break;
        case NVIMGCODEC_SAMPLE_DATA_TYPE_INT16:
            res = decodeImplTyped<int16_t>(plugin_id_, framework_, code_stream, image_info, params, tiff.get(), info);
            break;
        case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT32:
            res = decodeImplTyped<uint32_t>(plugin_id_, framework_, code_stream, image_info, params,tiff.get(), info);
            break;
        case NVIMGCODEC_SAMPLE_DATA_TYPE_INT32:
            res = decodeImplTyped<int32_t>(plugin_id_, framework_, code_stream, image_info, params, tiff.get(), info);
            break;
        case NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32:
            res = decodeImplTyped<float>(plugin_id_, framework_, code_stream, image_info, params, tiff.get(), info);
            break;
        default:
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Invalid data type: " << image_info.plane_info[0].sample_type);
            res = NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
            break;
        }
        image->imageReady(image->instance, res);
        return res == NVIMGCODEC_PROCESSING_STATUS_SUCCESS ? NVIMGCODEC_STATUS_SUCCESS : NVIMGCODEC_STATUS_EXECUTION_FAILED;
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode tiff code stream - " << e.what());
        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
        return NVIMGCODEC_STATUS_EXECUTION_FAILED;
    }
}

} // namespace libtiff
