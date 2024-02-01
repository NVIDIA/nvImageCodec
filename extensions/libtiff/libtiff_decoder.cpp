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
#include <cstring>
#include <future>

#define NOMINMAX
#include <nvtx3/nvtx3.hpp>
#include "convert.h"
#include "error_handling.h"
#include "log.h"
#include "nvimgcodec.h"

namespace libtiff {

class DecoderHelper
{
  public:
    explicit DecoderHelper(nvimgcodecIoStreamDesc_t* io_stream)
        : io_stream_(io_stream)
    {}

    static tmsize_t read(thandle_t handle, void* buffer, tmsize_t n)
    {
        DecoderHelper* helper = reinterpret_cast<DecoderHelper*>(handle);
        size_t read_nbytes = 0;
        if (helper->io_stream_->read(helper->io_stream_->instance, &read_nbytes, buffer, n) != NVIMGCODEC_STATUS_SUCCESS)
            return 0;
        else
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
        if (helper->io_stream_->seek(helper->io_stream_->instance, offset, whence) != NVIMGCODEC_STATUS_SUCCESS)
            return -1;
        ptrdiff_t curr_offset = 0;
        if (helper->io_stream_->tell(helper->io_stream_->instance, &curr_offset) != NVIMGCODEC_STATUS_SUCCESS)
            return -1;
        return curr_offset;
    }

    static int map(thandle_t handle, void** base, toff_t* size)
    {
        // This function will be used by LibTIFF only if input is InputKind::HostMemory.
        DecoderHelper* helper = reinterpret_cast<DecoderHelper*>(handle);
        size_t data_size = 0;
        if (helper->io_stream_->size(helper->io_stream_->instance, &data_size) != NVIMGCODEC_STATUS_SUCCESS)
            return -1;
        void* addr = nullptr;
        if (helper->io_stream_->map(helper->io_stream_->instance, &addr, 0, data_size) != NVIMGCODEC_STATUS_SUCCESS)
            return -1;
        if (addr == nullptr)
            return -1;

        *base = const_cast<void*>(addr);
        *size = data_size;
        return 0;
    }

    static void unmap(thandle_t handle, void* base, toff_t size)
    {
        DecoderHelper* helper = reinterpret_cast<DecoderHelper*>(handle);
        helper->io_stream_->unmap(helper->io_stream_->instance, base, size);
    }

    static toff_t size(thandle_t handle)
    {
        DecoderHelper* helper = reinterpret_cast<DecoderHelper*>(handle);
        size_t data_size = 0;
        if (helper->io_stream_->size(helper->io_stream_->instance, &data_size) != NVIMGCODEC_STATUS_SUCCESS)
            return 0;
        return data_size;
    }

    static int close(thandle_t handle)
    {
        DecoderHelper* helper = reinterpret_cast<DecoderHelper*>(handle);
        delete helper;
        return 0;
    }

  private:
    nvimgcodecIoStreamDesc_t* io_stream_;
};

std::unique_ptr<TIFF, void (*)(TIFF*)> OpenTiff(nvimgcodecIoStreamDesc_t* io_stream)
{
    TIFF* tiffptr;
    TIFFMapFileProc mapproc = nullptr;
    TIFFUnmapFileProc unmapproc = nullptr;

    void* addr = nullptr;
    if (io_stream->map(io_stream->instance, &addr, 0, 1) == NVIMGCODEC_STATUS_SUCCESS && addr != nullptr) {
        mapproc = &DecoderHelper::map;
        unmapproc = &DecoderHelper::unmap;
        io_stream->unmap(io_stream->instance, &addr, 1);
    }

    DecoderHelper* helper = new DecoderHelper(io_stream);
    tiffptr = TIFFClientOpen("", "r", reinterpret_cast<thandle_t>(helper), &DecoderHelper::read, &DecoderHelper::write,
        &DecoderHelper::seek, &DecoderHelper::close, &DecoderHelper::size, mapproc, unmapproc);
    if (!tiffptr)
        delete helper;
    if (tiffptr == nullptr)
        std::runtime_error("Unable to open TIFF image");
    return {tiffptr, &TIFFClose};
}

struct TiffInfo
{
    uint32_t image_width, image_height;
    uint16_t channels;

    uint32_t rows_per_strip;
    uint16_t bit_depth;
    uint16_t orientation;
    uint16_t compression;
    uint16_t photometric_interpretation;
    uint16_t fill_order;

    bool is_tiled;
    bool is_palette;
    bool is_planar;
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
    }

    return info;
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

    size_t nbits = info.bit_depth;
    size_t out_type_bits = 8 * sizeof(OutputType);
    if (out_type_bits < nbits)
        throw std::logic_error("Unpacking bits failed: OutputType too small");
    if (n == 0)
        return;

    auto in_ptr = static_cast<const uint8_t*>(in);
    uint8_t buffer = *(in_ptr++);
    constexpr size_t buffer_capacity = 8 * sizeof(buffer);
    size_t bits_in_buffer = buffer_capacity;

    for (size_t i = 0; i < n; i++) {
        OutputType result = 0;
        size_t bits_to_read = nbits;
        while (bits_to_read > 0) {
            if (bits_in_buffer >= bits_to_read) {
                // If we have enough bits in the buffer, we store them and finish
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

struct DecodeState
{
    DecodeState(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, int num_threads)
        : plugin_id_(plugin_id)
        , framework_(framework)
        , per_thread_(num_threads)
    {}
    ~DecodeState() = default;

    struct PerThreadResources
    {
        std::vector<uint8_t> buffer;
    };

    struct Sample
    {
        nvimgcodecCodeStreamDesc_t* code_stream;
        nvimgcodecImageDesc_t* image;
        const nvimgcodecDecodeParams_t* params;
    };
    const char* plugin_id_;
    const nvimgcodecFrameworkDesc_t* framework_;
    std::vector<PerThreadResources> per_thread_;
    std::vector<Sample> samples_;
};

struct DecoderImpl
{
    DecoderImpl(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params);
    ~DecoderImpl();

    nvimgcodecStatus_t canDecode(nvimgcodecProcessingStatus_t* status, nvimgcodecCodeStreamDesc_t* code_stream,
        nvimgcodecImageDesc_t* image, const nvimgcodecDecodeParams_t* params);
    nvimgcodecStatus_t canDecode(nvimgcodecProcessingStatus_t* status, nvimgcodecCodeStreamDesc_t** code_streams,
        nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params);
    static nvimgcodecProcessingStatus_t decode(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework,
        nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecImageDesc_t* image, const nvimgcodecDecodeParams_t* params,
        std::vector<uint8_t>& buffer);
    nvimgcodecStatus_t decodeBatch(
        nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params);

    static nvimgcodecStatus_t static_destroy(nvimgcodecDecoder_t decoder);
    static nvimgcodecStatus_t static_can_decode(nvimgcodecDecoder_t decoder, nvimgcodecProcessingStatus_t* status,
        nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params);
    static nvimgcodecStatus_t static_decode_batch(nvimgcodecDecoder_t decoder, nvimgcodecCodeStreamDesc_t** code_streams,
        nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params);

    const char* plugin_id_;
    const nvimgcodecFrameworkDesc_t* framework_;
    const nvimgcodecExecutionParams_t* exec_params_;
    std::unique_ptr<DecodeState> decode_state_batch_;

    struct CanDecodeCtx
    {
        DecoderImpl* this_ptr;
        nvimgcodecProcessingStatus_t* status;
        nvimgcodecCodeStreamDesc_t** code_streams;
        nvimgcodecImageDesc_t** images;
        const nvimgcodecDecodeParams_t* params;
        int num_samples;
        int num_blocks;
        std::vector<std::promise<void>> promise;
    };
};

LibtiffDecoderPlugin::LibtiffDecoderPlugin(const nvimgcodecFrameworkDesc_t* framework)
    : decoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC, sizeof(nvimgcodecDecoderDesc_t), NULL, this, plugin_id_, "tiff", NVIMGCODEC_BACKEND_KIND_CPU_ONLY, static_create,
          DecoderImpl::static_destroy, DecoderImpl::static_can_decode, DecoderImpl::static_decode_batch}
    , framework_(framework)
{}

nvimgcodecDecoderDesc_t* LibtiffDecoderPlugin::getDecoderDesc()
{
    return &decoder_desc_;
}

nvimgcodecStatus_t DecoderImpl::canDecode(nvimgcodecProcessingStatus_t* status, nvimgcodecCodeStreamDesc_t* code_stream,
    nvimgcodecImageDesc_t* image, const nvimgcodecDecodeParams_t* params)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "libtiff_can_decode");
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(code_stream);
        XM_CHECK_NULL(image);
        XM_CHECK_NULL(params);
        *status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
        nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        code_stream->getImageInfo(code_stream->instance, &cs_image_info);
        if (strcmp(cs_image_info.codec_name, "tiff") != 0) {
            *status = NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
            return NVIMGCODEC_STATUS_SUCCESS;
        }
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        image->getImageInfo(image->instance, &image_info);
        switch (image_info.sample_format) {
        case NVIMGCODEC_SAMPLEFORMAT_P_YUV:
            *status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
            break;
        case NVIMGCODEC_SAMPLEFORMAT_I_BGR:
        case NVIMGCODEC_SAMPLEFORMAT_I_RGB:
        case NVIMGCODEC_SAMPLEFORMAT_P_BGR:
        case NVIMGCODEC_SAMPLEFORMAT_P_RGB:
        case NVIMGCODEC_SAMPLEFORMAT_P_Y:
        case NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED:
        case NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED:
        default:
            break; // supported
        }

        if (image_info.num_planes == 1) {
            if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_BGR || image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_RGB)
                *status |= NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
        } else if (image_info.num_planes > 1) {
            if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_BGR || image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_RGB ||
                image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED)
                *status |= NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
        }
        if (image_info.num_planes != 1 && image_info.num_planes != 3 && image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED)
            *status |= NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;

        if (image_info.plane_info[0].num_channels == 1) {
            if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_BGR || image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_RGB)
                *status |= NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
        } else if (image_info.plane_info[0].num_channels > 1) {
            if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_BGR || image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_RGB ||
                image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED)
                *status |= NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
        }
        if (image_info.plane_info[0].num_channels != 1 && image_info.plane_info[0].num_channels != 3 &&
            image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED)
            *status |= NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;

        // This codec doesn't apply EXIF orientation
        if (params->apply_exif_orientation &&
            (image_info.orientation.flip_x || image_info.orientation.flip_y || image_info.orientation.rotated != 0)) {
            *status |= NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED;
        }
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if libtiff can decode - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t DecoderImpl::canDecode(nvimgcodecProcessingStatus_t* status, nvimgcodecCodeStreamDesc_t** code_streams,
    nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "libtiff_can_decode");
        nvtx3::scoped_range marker{"libtiff_can_decode"};
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images);
        XM_CHECK_NULL(params);

        auto executor = exec_params_->executor;
        int num_threads = executor->getNumThreads(executor->instance);

        if (batch_size < (num_threads + 1)) {  // not worth parallelizing
            for (int i = 0; i < batch_size; i++)
                canDecode(&status[i], code_streams[i], images[i], params);
        } else {
            int num_blocks = num_threads + 1;  // the last block is processed in the current thread
            CanDecodeCtx canDecodeCtx{this, status, code_streams, images, params, batch_size, num_blocks};
            canDecodeCtx.promise.resize(num_threads);
            std::vector<std::future<void>> fut;
            fut.reserve(num_threads);
            for (auto& pr : canDecodeCtx.promise)
                fut.push_back(pr.get_future());
            auto task = [](int tid, int block_idx, void* context) -> void {
                auto* ctx = reinterpret_cast<CanDecodeCtx*>(context);
                int64_t i_start = ctx->num_samples * block_idx / ctx->num_blocks;
                int64_t i_end = ctx->num_samples * (block_idx + 1) / ctx->num_blocks;
                for (int i = i_start; i < i_end; i++) {
                    ctx->this_ptr->canDecode(&ctx->status[i], ctx->code_streams[i], ctx->images[i], ctx->params);
                }
                if (block_idx < static_cast<int>(ctx->promise.size()))
                    ctx->promise[block_idx].set_value();
            };
            int block_idx = 0;
            for (; block_idx < num_threads; ++block_idx) {
                executor->launch(executor->instance, exec_params_->device_id, block_idx, &canDecodeCtx, task);
            }
            task(-1, block_idx, &canDecodeCtx);

            // wait for it to finish
            for (auto& f : fut)
                f.wait();
        }
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if opencv can decode - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t DecoderImpl::static_can_decode(nvimgcodecDecoder_t decoder, nvimgcodecProcessingStatus_t* status,
    nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    try {
        XM_CHECK_NULL(decoder);
        auto handle = reinterpret_cast<DecoderImpl*>(decoder);
        return handle->canDecode(status, code_streams, images, batch_size, params);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
}

DecoderImpl::DecoderImpl(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params)
    : plugin_id_(plugin_id)
    , framework_(framework)
    , exec_params_(exec_params)
{
    auto executor = exec_params_->executor;
    int num_threads = executor->getNumThreads(executor->instance);
    decode_state_batch_ = std::make_unique<DecodeState>(plugin_id_, framework_, num_threads);
}

nvimgcodecStatus_t LibtiffDecoderPlugin::create(
    nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "libtiff_create");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(exec_params);
        *decoder = reinterpret_cast<nvimgcodecDecoder_t>(new DecoderImpl(plugin_id_, framework_, exec_params));
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create libtiff decoder - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t LibtiffDecoderPlugin::static_create(
    void* instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    try {
        XM_CHECK_NULL(instance);
        auto handle = reinterpret_cast<LibtiffDecoderPlugin*>(instance);
        handle->create(decoder, exec_params, options);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
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
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }

    return NVIMGCODEC_STATUS_SUCCESS;
}

template <typename Output, typename Input>
nvimgcodecProcessingStatus_t decodeImplTyped2(
    const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, nvimgcodecImageInfo_t& image_info, TIFF* tiff, const TiffInfo& info)
{
    if (info.photometric_interpretation != PHOTOMETRIC_RGB && info.photometric_interpretation != PHOTOMETRIC_MINISBLACK &&
        info.photometric_interpretation != PHOTOMETRIC_PALETTE) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unsupported photometric interpretation: " << info.photometric_interpretation);
        return NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED;
    }

    if (info.is_planar) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Planar TIFFs are not supported");
        return NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED;
    }

    if (info.bit_depth > 32) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unsupported bit depth: " << info.bit_depth);
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
        num_channels = 1;
        planar = true;
        break;
    default:
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unsupported sample_format: " << image_info.sample_format);
        return NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
    }

    if (image_info.num_planes > info.channels) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Invalid number of planes: " << image_info.num_planes);
        return NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
    }

    int64_t region_start_y = image_info.region.ndim == 2 ? image_info.region.start[0] : 0;
    int64_t region_start_x = image_info.region.ndim == 2 ? image_info.region.start[1] : 0;
    int64_t region_end_y = image_info.region.ndim == 2 ? image_info.region.end[0] : image_info.plane_info[0].height;
    int64_t region_end_x = image_info.region.ndim == 2 ? image_info.region.end[1] : image_info.plane_info[0].width;
    int64_t region_size_x = region_end_x - region_start_x;
    int64_t stride_y = planar ? region_size_x : region_size_x * num_channels;
    int64_t stride_x = planar ? 1 : num_channels;
    int64_t tile_stride_y = info.tile_width * info.channels;
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
    if (!convert_needed) {
        in = static_cast<Input*>(buf.get());
    } else {
        scratch.resize(info.tile_height * info.tile_width * info.channels * sizeof(Input));
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
                size_t input_values = info.tile_height * info.tile_width * info.channels;
                if (info.is_palette)
                    input_values /= info.channels;
                TiffConvert(info, in, buf.get(), input_values);
            }

            Output* dst = img_out + (tile_begin_y - region_start_y) * stride_y + (tile_begin_x - region_start_x) * stride_x;
            const Input* src = in + (tile_begin_y - tile_y) * tile_stride_y + (tile_begin_x - tile_x) * tile_stride_x;

            switch (image_info.sample_format) {
            case NVIMGCODEC_SAMPLEFORMAT_P_Y:
                if (info.channels == 1) {
                    auto* plane = dst;
                    for (uint32_t i = 0; i < tile_size_y; i++) {
                        auto* row = plane + i * stride_y;
                        auto* tile_row = src + i * tile_stride_y;
                        for (uint32_t j = 0; j < tile_size_x; j++) {
                            *(row + j * stride_x) = *(tile_row + j * tile_stride_x);
                        }
                    }
                } else if (info.channels >= 3) {
                    uint32_t plane_stride = image_info.plane_info[0].height * image_info.plane_info[0].row_stride;
                    for (uint32_t c = 0; c < image_info.num_planes; c++) {
                        auto* plane = dst + c * plane_stride;
                        for (uint32_t i = 0; i < tile_size_y; i++) {
                            auto* row = plane + i * stride_y;
                            auto* tile_row = src + i * tile_stride_y;
                            for (uint32_t j = 0; j < tile_size_x; j++) {
                                auto* pixel = tile_row + j * tile_stride_x;
                                auto* out_pixel = row + j * stride_x;
                                auto r = *(pixel + 0);
                                auto g = *(pixel + 1);
                                auto b = *(pixel + 2);
                                *(out_pixel) = 0.299f * r + 0.587f * g + 0.114f * b;
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
                uint32_t plane_stride = image_info.plane_info[0].height * image_info.plane_info[0].row_stride;
                for (uint32_t c = 0; c < image_info.num_planes; c++) {
                    uint32_t dst_p = c;
                    if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_BGR)
                        dst_p = c == 2 ? 0 : c == 0 ? 2 : c;
                    auto* plane = dst + dst_p * plane_stride;
                    for (uint32_t i = 0; i < tile_size_y; i++) {
                        auto* row = plane + i * stride_y;
                        auto* tile_row = src + i * tile_stride_y;
                        for (uint32_t j = 0; j < tile_size_x; j++) {
                            *(row + j * stride_x) = *(tile_row + j * tile_stride_x + c);
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
                        for (uint32_t c = 0; c < image_info.plane_info[0].num_channels; c++) {
                            uint32_t out_c = c;
                            if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_BGR)
                                out_c = c == 2 ? 0 : c == 0 ? 2 : c;
                            *(pixel + out_c) = *(tile_pixel + c);
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
    const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, nvimgcodecImageInfo_t& image_info, TIFF* tiff, const TiffInfo& info)
{
    if (info.bit_depth <= 8) {
        return decodeImplTyped2<Output, uint8_t>(plugin_id, framework, image_info, tiff, info);
    } else if (info.bit_depth <= 16) {
        return decodeImplTyped2<Output, uint16_t>(plugin_id, framework, image_info, tiff, info);
    } else if (info.bit_depth <= 32) {
        return decodeImplTyped2<Output, uint32_t>(plugin_id, framework, image_info, tiff, info);
    } else {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unsupported bit depth: " << info.bit_depth);
        return NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
    }
}

nvimgcodecProcessingStatus_t DecoderImpl::decode(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework,
    nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecImageDesc_t* image, const nvimgcodecDecodeParams_t* params,
    std::vector<uint8_t>& buffer)
{
    try {
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        auto ret = image->getImageInfo(image->instance, &image_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS)
            return NVIMGCODEC_PROCESSING_STATUS_FAIL;

        if (image_info.region.ndim != 0 && image_info.region.ndim != 2) {
            NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Invalid region of interest");
            return NVIMGCODEC_PROCESSING_STATUS_FAIL;
        }

        if (image_info.buffer_kind != NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST) {
            NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unexpected buffer kind");
            return NVIMGCODEC_PROCESSING_STATUS_FAIL;
        }

        auto io_stream = code_stream->io_stream;
        io_stream->seek(io_stream->instance, 0, SEEK_SET);
        auto tiff = OpenTiff(io_stream);
        auto info = GetTiffInfo(tiff.get());
        switch (image_info.plane_info[0].sample_type) {
        case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8:
            return decodeImplTyped<uint8_t>(plugin_id, framework, image_info, tiff.get(), info);
        case NVIMGCODEC_SAMPLE_DATA_TYPE_INT8:
            return decodeImplTyped<uint8_t>(plugin_id, framework, image_info, tiff.get(), info);
        case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16:
            return decodeImplTyped<uint8_t>(plugin_id, framework, image_info, tiff.get(), info);
        case NVIMGCODEC_SAMPLE_DATA_TYPE_INT16:
            return decodeImplTyped<uint8_t>(plugin_id, framework, image_info, tiff.get(), info);
        case NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32:
            return decodeImplTyped<uint8_t>(plugin_id, framework, image_info, tiff.get(), info);
        default:
            NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Invalid data type: " << image_info.plane_info[0].sample_type);
            return NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
        }
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Could not decode jpeg code stream - " << e.what());
        return NVIMGCODEC_PROCESSING_STATUS_FAIL;
    }
    return NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
}

nvimgcodecStatus_t DecoderImpl::decodeBatch(
    nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "libtiff_decode_batch");
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images)
        XM_CHECK_NULL(params)
        if (batch_size < 1) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Batch size lower than 1");
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }

        decode_state_batch_->samples_.resize(batch_size);
        for (int i = 0; i < batch_size; i++) {
            decode_state_batch_->samples_[i].code_stream = code_streams[i];
            decode_state_batch_->samples_[i].image = images[i];
            decode_state_batch_->samples_[i].params = params;
        }

        auto task = [](int tid, int sample_idx, void* context) -> void {
            nvtx3::scoped_range marker{"libtiff decode " + std::to_string(sample_idx)};
            auto* decode_state = reinterpret_cast<DecodeState*>(context);
            auto& sample = decode_state->samples_[sample_idx];
            auto& thread_resources = decode_state->per_thread_[tid];
            auto& plugin_id = decode_state->plugin_id_;
            auto& framework = decode_state->framework_;

            auto result = decode(plugin_id, framework, sample.code_stream, sample.image, sample.params, thread_resources.buffer);
            sample.image->imageReady(sample.image->instance, result);
        };
        if (batch_size == 1) {
            task(0, 0, decode_state_batch_.get());
        } else {
            auto executor = exec_params_->executor;
            for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
                executor->launch(executor->instance, NVIMGCODEC_DEVICE_CPU_ONLY, sample_idx, decode_state_batch_.get(), task);
            }
        }
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode tiff batch - " << e.what());
        for (int i = 0; i < batch_size; ++i) {
            images[i]->imageReady(images[i]->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
        }
        return NVIMGCODEC_STATUS_INTERNAL_ERROR;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t DecoderImpl::static_decode_batch(nvimgcodecDecoder_t decoder, nvimgcodecCodeStreamDesc_t** code_streams,
    nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    try {
        XM_CHECK_NULL(decoder);
        auto handle = reinterpret_cast<DecoderImpl*>(decoder);
        return handle->decodeBatch(code_streams, images, batch_size, params);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
}

} // namespace libtiff
