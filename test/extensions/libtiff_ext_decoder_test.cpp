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

#include <extensions/libtiff/libtiff_ext.h>
#include "common_ext_decoder_test.h"
#include <gtest/gtest.h>
#include <nvimgcodec.h>
#include <parsers/tiff.h>
#include <parsers/parser_test_utils.h>
#include <test_utils.h>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <atomic>
#include <future>
#include <mutex>
#include <thread>
#include <tiffio.h>
#include "nvimgcodec_tests.h"

namespace libtiff {
// The test target links libtiff_ext_static; OpenTiff is not exported by the shared extension.
std::unique_ptr<TIFF, void (*)(TIFF*)> OpenTiff(nvimgcodecIoStreamDesc_t* io_stream);
}

using ::testing::Combine;
using ::testing::Values;

namespace nvimgcodec { namespace test {

// Individual callbacks are synchronized, but a seek followed by a read is not
// atomic: this models the shared-cursor hazard without a data race in the test.
struct SharedTiffInput
{
    std::vector<uint8_t> bytes;
    size_t position = 0;
    bool mappable = false;
    std::mutex mutex;
    std::atomic<int> mappings{0}, unmappings{0}, reads{0};
    std::atomic<bool> bad_unmap{false};

    nvimgcodecIoStreamDesc_t descriptor()
    {
        nvimgcodecIoStreamDesc_t io{NVIMGCODEC_STRUCTURE_TYPE_IO_STREAM_DESC, sizeof(io), nullptr};
        io.instance = this;
        io.read = [](void* ptr, size_t* count, void* dst, size_t requested) {
            auto& s = *static_cast<SharedTiffInput*>(ptr);
            std::lock_guard<std::mutex> lock(s.mutex);
            *count = std::min(requested, s.bytes.size() - s.position);
            std::memcpy(dst, s.bytes.data() + s.position, *count);
            s.position += *count;
            ++s.reads;
            return NVIMGCODEC_STATUS_SUCCESS;
        };
        io.seek = [](void* ptr, ptrdiff_t offset, int whence) {
            auto& s = *static_cast<SharedTiffInput*>(ptr);
            {
                std::lock_guard<std::mutex> lock(s.mutex);
                size_t base = whence == SEEK_CUR ? s.position : whence == SEEK_END ? s.bytes.size() : 0;
                size_t next = base + offset;
                if (next > s.bytes.size())
                    return NVIMGCODEC_STATUS_INVALID_PARAMETER;
                s.position = next;
            }
            std::this_thread::yield(); // Encourage interference between seek and read.
            return NVIMGCODEC_STATUS_SUCCESS;
        };
        io.tell = [](void* ptr, ptrdiff_t* offset) {
            auto& s = *static_cast<SharedTiffInput*>(ptr);
            std::lock_guard<std::mutex> lock(s.mutex);
            *offset = s.position;
            return NVIMGCODEC_STATUS_SUCCESS;
        };
        io.size = [](void* ptr, size_t* size) {
            *size = static_cast<SharedTiffInput*>(ptr)->bytes.size();
            return NVIMGCODEC_STATUS_SUCCESS;
        };
        io.map = [](void* ptr, void** data, size_t offset, size_t) {
            auto& s = *static_cast<SharedTiffInput*>(ptr);
            *data = s.mappable ? s.bytes.data() + offset : nullptr;
            if (*data)
                ++s.mappings;
            return NVIMGCODEC_STATUS_SUCCESS;
        };
        io.unmap = [](void* ptr, void* data, size_t size) {
            auto& s = *static_cast<SharedTiffInput*>(ptr);
            if (data != s.bytes.data() || size != s.bytes.size())
                s.bad_unmap = true;
            ++s.unmappings;
            return NVIMGCODEC_STATUS_SUCCESS;
        };
        return io;
    }
};

TEST(LibtiffReader, ConcurrentReadersHaveIndependentCursors)
{
    const auto path = resources_dir + "/tiff/cat-300572_640_no_compression.tiff";
    SharedTiffInput input;
    input.bytes = read_file(path);
    ASSERT_FALSE(input.bytes.empty());
    auto reference_tiff = std::unique_ptr<TIFF, decltype(&TIFFClose)>(TIFFOpen(path.c_str(), "r"), TIFFClose);
    ASSERT_NE(nullptr, reference_tiff);
    uint32_t width = 0, height = 0;
    ASSERT_EQ(1, TIFFGetField(reference_tiff.get(), TIFFTAG_IMAGEWIDTH, &width));
    ASSERT_EQ(1, TIFFGetField(reference_tiff.get(), TIFFTAG_IMAGELENGTH, &height));
    std::vector<uint32_t> reference(static_cast<size_t>(width) * height);
    ASSERT_EQ(1, TIFFReadRGBAImageOriented(reference_tiff.get(), width, height, reference.data(), ORIENTATION_TOPLEFT));
    for (bool mapped : {true, false, true}) {
        SCOPED_TRACE(mapped ? "mapped" : "unmapped");
        input.mappable = mapped;
        input.position = input.bytes.size() / 2; // Opening must not depend on the shared cursor.
        input.reads = 0;
        input.mappings = 0;
        input.unmappings = 0;
        input.bad_unmap = false;
        auto io = input.descriptor();
        std::vector<std::future<bool>> workers;
        for (int worker = 0; worker < 4; ++worker) {
            workers.push_back(std::async(std::launch::async, [&] {
                for (int iteration = 0; iteration < 10; ++iteration) {
                    auto tiff = libtiff::OpenTiff(&io);
                    std::vector<uint32_t> pixels(reference.size());
                    if (!TIFFReadRGBAImageOriented(tiff.get(), width, height, pixels.data(), ORIENTATION_TOPLEFT) || pixels != reference)
                        return false;
                }
                return true;
            }));
        }
        for (auto& worker : workers)
            EXPECT_TRUE(worker.get());
        if (mapped) {
            EXPECT_EQ(0, input.reads);
            EXPECT_EQ(input.bytes.size() / 2, input.position);
            EXPECT_EQ(40, input.mappings);
            EXPECT_EQ(input.mappings.load(), input.unmappings.load());
            EXPECT_FALSE(input.bad_unmap);
        } else {
            EXPECT_GT(input.reads, 0);
        }
    }
}

class LibtiffExtDecoderTest : public CommonExtDecoderTestWithPathAndFormat
{
  public:
    void SetUp() override
    {
        CommonExtDecoderTestWithPathAndFormat::SetUp();

        nvimgcodecExtensionDesc_t tiff_parser_extension_desc{
            NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_tiff_parser_extension_desc(&tiff_parser_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &tiff_parser_extension_desc));

        nvimgcodecExtensionDesc_t libtiff_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_libtiff_extension_desc(&libtiff_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &libtiff_extension_desc));

        CommonExtDecoderTestWithPathAndFormat::CreateDecoder();
    }
};

DEFINE_SINGLE_IMAGE_STRIDE_TESTS(LibtiffExtDecoderTest)

INSTANTIATE_TEST_SUITE_P(LIBTIFF_DECODE,
    LibtiffExtDecoderTest,
    Combine(
        Values(
            "tiff/cat-1245673_640.tiff",
            "tiff/cat-1245673_300572.tiff",
            "tiff/cat-300572_640_palette.tiff",
            "tiff/cat-300572_640_grayscale.tiff",
            "tiff/cat-300572_640_no_compression.tiff",
            "tiff/cat-300572_640_ycbcr.tiff",
            "tiff/cat-300572_640_fp32.tiff"
        ), Values (
            NVIMGCODEC_SAMPLEFORMAT_I_RGB,
            NVIMGCODEC_SAMPLEFORMAT_I_BGR,
            NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED,
            NVIMGCODEC_SAMPLEFORMAT_P_RGB,
            NVIMGCODEC_SAMPLEFORMAT_P_BGR,
            NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED,
            NVIMGCODEC_SAMPLEFORMAT_P_Y
        )
    )
);

// Higher-precision (uint16 / fp32) reference decode tests: same extension setup
// as LibtiffExtDecoderTest, driven through the path+format+type fixture.
class LibtiffExtDecoderTestHighPrec : public CommonExtDecoderTestWithPathFormatAndType
{
  public:
    void SetUp() override
    {
        CommonExtDecoderTestWithPathFormatAndType::SetUp();

        nvimgcodecExtensionDesc_t tiff_parser_extension_desc{
            NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_tiff_parser_extension_desc(&tiff_parser_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &tiff_parser_extension_desc));

        nvimgcodecExtensionDesc_t libtiff_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_libtiff_extension_desc(&libtiff_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &libtiff_extension_desc));

        CommonExtDecoderTestWithPathFormatAndType::CreateDecoder();
    }
};

DEFINE_SINGLE_IMAGE_HIGHPREC_STRIDE_TESTS(LibtiffExtDecoderTestHighPrec)

INSTANTIATE_TEST_SUITE_P(LIBTIFF_DECODE_HIGHPREC_UINT16,
    LibtiffExtDecoderTestHighPrec,
    Combine(
        Values("tiff/cat-300572_640_uint16.tiff"),
        Values(NVIMGCODEC_SAMPLEFORMAT_I_RGB, NVIMGCODEC_SAMPLEFORMAT_I_BGR,
               NVIMGCODEC_SAMPLEFORMAT_P_RGB, NVIMGCODEC_SAMPLEFORMAT_P_BGR),
        Values(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16)
    )
);

INSTANTIATE_TEST_SUITE_P(LIBTIFF_DECODE_HIGHPREC_FP32,
    LibtiffExtDecoderTestHighPrec,
    Combine(
        Values("tiff/cat-300572_640_fp32.tiff"),
        Values(NVIMGCODEC_SAMPLEFORMAT_I_RGB, NVIMGCODEC_SAMPLEFORMAT_I_BGR,
               NVIMGCODEC_SAMPLEFORMAT_P_RGB, NVIMGCODEC_SAMPLEFORMAT_P_BGR),
        Values(NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32)
    )
);

// The LibTIFF (CPU) decoder used to reject SAMPLEFORMAT_INT outright
// ("Unsupported sample format: 2"). These fixtures reuse the signed int16 TIFFs also used
// by the nvTIFF signed-decode tests, but here we decode straight to a signed INT16 output
// (no unsigned midpoint conversion), so the raw ramp values must come back unmodified.
// The fixtures carry the column ramp {-32767, -16384, 0, 6554, 32767} repeated over 4 rows.
class LibtiffExtDecoderSignedTest : public CommonExtDecoderTest
{
  public:
    void SetUp() override
    {
        CommonExtDecoderTest::SetUp();

        nvimgcodecExtensionDesc_t tiff_parser_extension_desc{
            NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_tiff_parser_extension_desc(&tiff_parser_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &tiff_parser_extension_desc));

        nvimgcodecExtensionDesc_t libtiff_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_libtiff_extension_desc(&libtiff_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &libtiff_extension_desc));

        CommonExtDecoderTest::CreateDecoder();
    }

    // Decodes rel_path to a signed output of type T (matching sample_type/precision) with the
    // given sample_format/num_channels.
    template <typename T>
    nvimgcodecProcessingStatus_t DecodeToSigned(
        const std::string& rel_path, nvimgcodecSampleFormat_t sample_format, int num_channels, nvimgcodecSampleDataType_t sample_type)
    {
        LoadImageFromFilename(instance_, in_code_stream_, resources_dir + "/" + rel_path);
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(in_code_stream_, &image_info_));
        image_info_.sample_format = sample_format;
        image_info_.color_spec = num_channels == 1 ? NVIMGCODEC_COLORSPEC_GRAY : NVIMGCODEC_COLORSPEC_SRGB;
        image_info_.chroma_subsampling = num_channels == 1 ? NVIMGCODEC_SAMPLING_GRAY : NVIMGCODEC_SAMPLING_444;
        image_info_.num_planes = 1;
        image_info_.plane_info[0].num_channels = num_channels;
        image_info_.plane_info[0].sample_type = sample_type;
        image_info_.plane_info[0].precision = 8 * sizeof(T);
        image_info_.plane_info[0].row_stride = image_info_.plane_info[0].width * num_channels * sizeof(T);
        out_buffer_.resize(GetBufferSize(image_info_));
        image_info_.buffer = out_buffer_.data();
        image_info_.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &image_, &image_info_));
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS,
            nvimgcodecDecoderDecode(decoder_, &in_code_stream_, &image_, 1, &params_, &future_));
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future_));
        nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_FAIL;
        size_t status_size = 0;
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future_, &status, &status_size));
        return status;
    }

    nvimgcodecProcessingStatus_t DecodeToInt16(const std::string& rel_path, nvimgcodecSampleFormat_t sample_format, int num_channels)
    {
        return DecodeToSigned<int16_t>(rel_path, sample_format, num_channels, NVIMGCODEC_SAMPLE_DATA_TYPE_INT16);
    }

    // Checks a single-channel plane of type T against expected, indexed by column mod 5 and
    // repeated over every row (the shape all synth-4x5_* fixtures share).
    template <typename T>
    void ExpectGrayRamp(const T (&expected)[5])
    {
        const T* pixels = reinterpret_cast<const T*>(out_buffer_.data());
        for (uint32_t y = 0; y < image_info_.plane_info[0].height; y++) {
            const T* row = pixels + y * (image_info_.plane_info[0].row_stride / sizeof(T));
            for (uint32_t x = 0; x < image_info_.plane_info[0].width; x++) {
                EXPECT_EQ(expected[x % 5], row[x]) << " at y=" << y << " x=" << x;
            }
        }
    }
};

TEST_F(LibtiffExtDecoderSignedTest, SignedGrayscaleDecodesToInt16Exactly)
{
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, DecodeToInt16("tiff/synth-4x5_int16_gray.tiff", NVIMGCODEC_SAMPLEFORMAT_P_Y, 1));
    const int16_t expected[5] = {-32767, -16384, 0, 6554, 32767};
    ExpectGrayRamp(expected);
}

TEST_F(LibtiffExtDecoderSignedTest, SignedRgbDecodesToInt16Exactly)
{
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, DecodeToInt16("tiff/synth-4x5_int16_rgb.tiff", NVIMGCODEC_SAMPLEFORMAT_I_RGB, 3));
    const int16_t expected[5] = {-32767, -16384, 0, 6554, 32767};
    const int16_t* pixels = reinterpret_cast<const int16_t*>(out_buffer_.data());
    for (uint32_t y = 0; y < image_info_.plane_info[0].height; y++) {
        const int16_t* row = pixels + y * (image_info_.plane_info[0].row_stride / sizeof(int16_t));
        for (uint32_t x = 0; x < image_info_.plane_info[0].width; x++) {
            for (int c = 0; c < 3; c++) {
                EXPECT_EQ(expected[x % 5], row[x * 3 + c]) << " at y=" << y << " x=" << x << " c=" << c;
            }
        }
    }
}

// Exercises the int8_t dispatch branch of decodeImplTyped, which the INT16-only fixtures above
// don't reach.
TEST_F(LibtiffExtDecoderSignedTest, SignedGrayscaleDecodesToInt8Exactly)
{
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS,
        (DecodeToSigned<int8_t>("tiff/synth-4x5_int8_gray.tiff", NVIMGCODEC_SAMPLEFORMAT_P_Y, 1, NVIMGCODEC_SAMPLE_DATA_TYPE_INT8)));
    const int8_t expected[5] = {-127, -64, 0, 25, 127};
    ExpectGrayRamp(expected);
}

// Exercises the int32_t dispatch branch of decodeImplTyped, which the INT16-only fixtures above
// don't reach.
TEST_F(LibtiffExtDecoderSignedTest, SignedGrayscaleDecodesToInt32Exactly)
{
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS,
        (DecodeToSigned<int32_t>("tiff/synth-4x5_int32_gray.tiff", NVIMGCODEC_SAMPLEFORMAT_P_Y, 1, NVIMGCODEC_SAMPLE_DATA_TYPE_INT32)));
    const int32_t expected[5] = {-2147483647, -1073774593, 0, 429536052, 2147483647};
    ExpectGrayRamp(expected);
}

// A palette TIFF with 16-bit indices (BitsPerSample=16, so 65536 possible palette entries) whose
// TIFFTAG_SAMPLEFORMAT is (incorrectly, but crafted files can do this) SAMPLEFORMAT_INT. Indices
// >= 0x8000 (here 40000, 60000, 65535) would become negative if GetTiffInfo let is_signed apply
// to the palette index width - that negative value would then be used directly to index the
// color-map arrays. This only reproduces at a 16-bit index width: any narrower palette index
// never fills enough bits of the (always 16-bit-wide) unpack buffer to set its sign bit.
TEST_F(LibtiffExtDecoderSignedTest, SignedFormatPaletteIndexIsTreatedAsUnsigned)
{
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS,
        (DecodeToSigned<uint16_t>(
            "tiff/synth-4x5_int16_palette.tiff", NVIMGCODEC_SAMPLEFORMAT_I_RGB, 3, NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16)));
    const int index[5] = {0, 50, 40000, 60000, 65535};
    const uint16_t* pixels = reinterpret_cast<const uint16_t*>(out_buffer_.data());
    for (uint32_t y = 0; y < image_info_.plane_info[0].height; y++) {
        const uint16_t* row = pixels + y * (image_info_.plane_info[0].row_stride / sizeof(uint16_t));
        for (uint32_t x = 0; x < image_info_.plane_info[0].width; x++) {
            const int i = index[x % 5];
            EXPECT_EQ(static_cast<uint16_t>(i), row[x * 3 + 0]) << " R at y=" << y << " x=" << x;
            EXPECT_EQ(static_cast<uint16_t>(65535 - i), row[x * 3 + 1]) << " G at y=" << y << " x=" << x;
            EXPECT_EQ(0, row[x * 3 + 2]) << " B at y=" << y << " x=" << x;
        }
    }
}

// Same fixture as SignedFormatPaletteIndexIsTreatedAsUnsigned, but decodes using the image info
// as reported by nvimgcodecCodeStreamGetImageInfo unchanged (planar P_RGB, one UINT16 channel per
// plane), instead of overriding sample_type/sample_format to an interleaved UINT16 RGB output.
// This is what a caller gets by default (e.g. the Python decode() without an explicit output
// dtype). The parser must report palette planes as UINT16 -- if it instead reported INT16
// (following the file's declared, but decoder-ignored, signed SampleFormat), the unsigned palette
// colors produced by the decoder would be reinterpreted as signed and corrupted by the
// unsigned-to-signed conversion.
TEST_F(LibtiffExtDecoderSignedTest, SignedFormatPaletteImageInfoReportsUnsignedIndices)
{
    LoadImageFromFilename(instance_, in_code_stream_, resources_dir + "/tiff/synth-4x5_int16_palette.tiff");
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(in_code_stream_, &image_info_));
    ASSERT_EQ(NVIMGCODEC_SAMPLEFORMAT_P_RGB, image_info_.sample_format);
    ASSERT_EQ(3u, image_info_.num_planes);
    for (uint32_t p = 0; p < image_info_.num_planes; p++) {
        EXPECT_EQ(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16, image_info_.plane_info[p].sample_type) << " plane " << p;
        // nvimgcodecCodeStreamGetImageInfo does not fill row_stride -- that describes the
        // caller-provided output buffer, not the source. Set the natural (unpadded) stride here.
        image_info_.plane_info[p].row_stride = image_info_.plane_info[p].width * image_info_.plane_info[p].num_channels * sizeof(uint16_t);
    }

    out_buffer_.resize(GetBufferSize(image_info_));
    image_info_.buffer = out_buffer_.data();
    image_info_.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &image_, &image_info_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDecode(decoder_, &in_code_stream_, &image_, 1, &params_, &future_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future_));
    nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_FAIL;
    size_t status_size = 0;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future_, &status, &status_size));
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, status);

    const uint16_t expected_plane[3][5] = {
        {0, 50, 40000, 60000, 65535},
        {65535, 65485, 25535, 5535, 0},
        {0, 0, 0, 0, 0},
    };
    uint8_t* plane_base = out_buffer_.data();
    for (uint32_t p = 0; p < image_info_.num_planes; p++) {
        const auto& plane = image_info_.plane_info[p];
        const uint16_t* pixels = reinterpret_cast<const uint16_t*>(plane_base);
        for (uint32_t y = 0; y < plane.height; y++) {
            const uint16_t* row = pixels + y * (plane.row_stride / sizeof(uint16_t));
            for (uint32_t x = 0; x < plane.width; x++) {
                EXPECT_EQ(expected_plane[p][x % 5], row[x]) << " plane " << p << " at y=" << y << " x=" << x;
            }
        }
        plane_base += plane.row_stride * plane.height;
    }
}

}} // namespace nvimgcodec::test
