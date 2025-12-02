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

#include <extensions/nvjpeg/nvjpeg_ext.h>
#include <gtest/gtest.h>
#include <nvimgcodec.h>
#include <parsers/parser_test_utils.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "nvimgcodec_tests.h"
#include "nvjpeg_ext_test_common.h"

using ::testing::Bool;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

#define NV_DEVELOPER_DUMP_OUTPUT_CODE_STREAM 0
#define NV_DEVELOPER_DEBUG_DUMP_DECODE_OUTPUT 0

namespace nvimgcodec { namespace test {

class NvJpegExtEncoderTestBase : public NvJpegExtTestBase
{
  public:
    NvJpegExtEncoderTestBase() {}

    virtual void SetUp()
    {
        NvJpegExtTestBase::SetUp();
        const char* options = nullptr;
        nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
        exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;
        exec_params.num_backends = 1;
        exec_params.backends = &backend_;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderCreate(instance_, &encoder_, &exec_params, options));
        backend_.kind = NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY;
        is_hw_encoder_available_ = nvimgcodecEncoderCreate(instance_, &hw_encoder_, &exec_params, options) == NVIMGCODEC_STATUS_SUCCESS;

        jpeg_enc_params_ = {NVIMGCODEC_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS, sizeof(nvimgcodecJpegEncodeParams_t), 0};
        jpeg_enc_params_.optimized_huffman = 0;
        params_ = {NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(nvimgcodecEncodeParams_t), 0};
        params_.struct_next = &jpeg_enc_params_;
        out_jpeg_image_info_ = {NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO, sizeof(nvimgcodecJpegImageInfo_t), 0};
    }

    virtual void TearDown()
    {
        if (encoder_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderDestroy(encoder_));
        }
        if (hw_encoder_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderDestroy(hw_encoder_));
        }
        NvJpegExtTestBase::TearDown();
    }

    nvimgcodecEncoder_t encoder_ = nullptr;
    nvimgcodecEncoder_t hw_encoder_ = nullptr;
    bool is_hw_encoder_available_ = false;
    nvimgcodecJpegEncodeParams_t jpeg_enc_params_;
    nvimgcodecEncodeParams_t params_;
    nvimgcodecJpegImageInfo_t out_jpeg_image_info_;
    nvimgcodecBackend_t backend_ 
    {
        NVIMGCODEC_STRUCTURE_TYPE_BACKEND, 
        sizeof(nvimgcodecBackend_t), 
        nullptr,
        NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU, 
        {
            NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, 
            sizeof(nvimgcodecBackendParams_t), 
            nullptr, 
            1.0f, 
            NVIMGCODEC_LOAD_HINT_POLICY_FIXED
        }
    };
    nvimgcodecBackendKind_t backend_kind_;
};

class NvJpegExtEncoderTestSingleImage : public NvJpegExtEncoderTestBase,
                                        public NvJpegTestBase,
                                        public TestWithParam<std::tuple<nvimgcodecBackendKind_t, const char*, nvimgcodecColorSpec_t, nvimgcodecSampleFormat_t,
                                            nvimgcodecChromaSubsampling_t, nvimgcodecChromaSubsampling_t, nvimgcodecJpegEncoding_t,
                                            nvimgcodecQualityType_t, float /*quality value*/, nvimgcodecProcessingStatus>>
{
  public:
    using NvJpegTestBase::SetUpTestSuite;
    using NvJpegTestBase::TearDownTestSuite;
    virtual ~NvJpegExtEncoderTestSingleImage() = default;

  protected:
    void SetUp() override
    {
        NvJpegExtEncoderTestBase::SetUp();
        NvJpegTestBase::SetUp();

        backend_kind_ = std::get<0>(GetParam());
        image_file_ = std::get<1>(GetParam());
        color_spec_ = std::get<2>(GetParam());
        sample_format_ = std::get<3>(GetParam());
        chroma_subsampling_ = std::get<4>(GetParam());
        encoded_chroma_subsampling_ = std::get<5>(GetParam());
        out_jpeg_image_info_.encoding = std::get<6>(GetParam());
        image_info_.struct_next = &out_jpeg_image_info_;
        params_.quality_type = std::get<7>(GetParam());
        params_.quality_value = std::get<8>(GetParam());
        expected_encode_status = std::get<9>(GetParam());
    }

    void TearDown() override
    {
        NvJpegTestBase::TearDown();
        NvJpegExtEncoderTestBase::TearDown();
    }

    nvimgcodecChromaSubsampling_t encoded_chroma_subsampling_;
    nvimgcodecProcessingStatus expected_encode_status;
};

TEST_P(NvJpegExtEncoderTestSingleImage, ValidFormatAndParameters)
{
    if (backend_kind_ == NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY && !is_hw_encoder_available_) {
        return;
    }

    nvimgcodecImageInfo_t ref_cs_image_info;
    DecodeReference(resources_dir, image_file_, sample_format_, true, &ref_cs_image_info);
    image_info_.plane_info[0] = ref_cs_image_info.plane_info[0];
    image_info_.plane_info[0].precision = 8;
    image_info_.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
    PrepareImageForFormat();
    memcpy(image_buffer_.data(), reinterpret_cast<void*>(ref_buffer_.data()), ref_buffer_.size());

    nvimgcodecImageInfo_t cs_image_info(image_info_);
    cs_image_info.chroma_subsampling = encoded_chroma_subsampling_;
    strcpy(cs_image_info.codec_name,"jpeg");
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &in_image_, &image_info_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,    
        nvimgcodecCodeStreamCreateToHostMem(instance_, &out_code_stream_, (void*)this,
            &NvJpegExtEncoderTestSingleImage::ResizeBufferStatic<NvJpegExtEncoderTestSingleImage>, &cs_image_info));
    images_.push_back(in_image_);
    streams_.push_back(out_code_stream_);
    if (backend_kind_ == NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU) {
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderEncode(encoder_, images_.data(), streams_.data(), 1, &params_, &future_));
    } else if (backend_kind_ == NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY) {
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderEncode(hw_encoder_, images_.data(), streams_.data(), 1, &params_, &future_));
    }
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future_));
    nvimgcodecProcessingStatus_t status;
    size_t status_size;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future_, &status, &status_size));
    ASSERT_EQ(expected_encode_status, status);
    ASSERT_EQ(status_size, 1);
    if (expected_encode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
        return;
    }

    if (NV_DEVELOPER_DUMP_OUTPUT_CODE_STREAM) {
        std::ofstream b_stream("./encoded_out.jpg", std::fstream::out | std::fstream::binary);
        b_stream.write(reinterpret_cast<char*>(code_stream_buffer_.data()), code_stream_buffer_.size());
    }

    LoadImageFromHostMemory(instance_, in_code_stream_, code_stream_buffer_.data(), code_stream_buffer_.size());
    nvimgcodecImageInfo_t load_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    nvimgcodecJpegImageInfo_t load_jpeg_info{NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO, sizeof(nvimgcodecJpegImageInfo_t), 0};
    load_info.struct_next = &load_jpeg_info;

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(in_code_stream_, &load_info));
    EXPECT_EQ(out_jpeg_image_info_.encoding, load_jpeg_info.encoding);
    EXPECT_EQ(cs_image_info.chroma_subsampling, load_info.chroma_subsampling);

    std::vector<unsigned char> ref_out_buffer;
#if NVJPEG_HW_ENCODER_SUPPORTED
    if (backend_kind_ == NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU)
        EncodeReference(image_info_, params_, jpeg_enc_params_, cs_image_info, out_jpeg_image_info_, &ref_out_buffer, NVJPEG_ENC_BACKEND_GPU);
    else if (backend_kind_ == NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY)
        EncodeReference(image_info_, params_, jpeg_enc_params_, cs_image_info, out_jpeg_image_info_, &ref_out_buffer, NVJPEG_ENC_BACKEND_HARDWARE);
#else
        EncodeReference(image_info_, params_, jpeg_enc_params_, cs_image_info, out_jpeg_image_info_, &ref_out_buffer);
#endif
    ASSERT_EQ(0,
        memcmp(reinterpret_cast<void*>(ref_out_buffer.data()), reinterpret_cast<void*>(code_stream_buffer_.data()), ref_out_buffer.size()));
}

// clang-format off

static const char* css_filenames[] = {"/jpeg/padlock-406986_640_410.jpg", "/jpeg/padlock-406986_640_411.jpg",
    "/jpeg/padlock-406986_640_420.jpg", "/jpeg/padlock-406986_640_422.jpg", "/jpeg/padlock-406986_640_440.jpg",
    "/jpeg/padlock-406986_640_444.jpg", "/jpeg/padlock-406986_640_gray.jpg"};

static const char* hw_filenames[] = {"/jpeg/padlock-406986_640_420.jpg"};

// This is a workaround for the NVJPEG_STATUS_ALLOCATOR_FAILURE (error 6) on first call DecodeReference
class DummyTestSuite : public NvJpegTestBase, public ::testing::Test {
public:
    using NvJpegTestBase::SetUpTestSuite;
    using NvJpegTestBase::TearDownTestSuite;
    virtual ~DummyTestSuite() = default;
protected:
    void SetUp() override {
        NvJpegTestBase::SetUp();
    }

    void TearDown() override {
        NvJpegTestBase::TearDown();
    }

};

TEST_F(DummyTestSuite, DummyDecodeReferenceNoAssert) {
    nvimgcodecImageInfo_t ref_cs_image_info;
    DecodeReference<false>(resources_dir, "/jpeg/padlock-406986_640_410.jpg", NVIMGCODEC_SAMPLEFORMAT_P_RGB, true, &ref_cs_image_info);
}

INSTANTIATE_TEST_SUITE_P(NVJPEG_HW_ENCODE_VALID, NvJpegExtEncoderTestSingleImage,
    Combine(
        Values(NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY),
        ValuesIn(hw_filenames),
        Values(NVIMGCODEC_COLORSPEC_SRGB),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_RGB, NVIMGCODEC_SAMPLEFORMAT_I_RGB, NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_420),
        Values(NVIMGCODEC_SAMPLING_420),
        Values(NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT),
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_VALID_SRGB_INPUT_FORMATS_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImage,
    Combine(
        Values(NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU),
        ValuesIn(css_filenames),
        Values(NVIMGCODEC_COLORSPEC_SRGB),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_RGB, NVIMGCODEC_SAMPLEFORMAT_I_RGB, NVIMGCODEC_SAMPLEFORMAT_P_BGR, NVIMGCODEC_SAMPLEFORMAT_I_BGR),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_440, NVIMGCODEC_SAMPLING_422 , NVIMGCODEC_SAMPLING_420, 
            NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410, NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT, NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN),
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_VALID_SYCC_CSS444_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImage,
    Combine(
        Values(NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU),
        Values("/jpeg/padlock-406986_640_444.jpg"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_440, NVIMGCODEC_SAMPLING_422 , NVIMGCODEC_SAMPLING_420, 
            NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410,  NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT, NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN),
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_VALID_SYCC_CSS410_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImage,
    Combine(
        Values(NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU),
        Values("/jpeg/padlock-406986_640_410.jpg"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_410),
        Values(NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_440, NVIMGCODEC_SAMPLING_422 , NVIMGCODEC_SAMPLING_420, 
            NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410, NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT, NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN),
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_VALID_SYCC_CSS411_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImage,
    Combine(
        Values(NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU),
        Values("/jpeg/padlock-406986_640_411.jpg"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_411),
        Values(NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_440, NVIMGCODEC_SAMPLING_422 , NVIMGCODEC_SAMPLING_420, 
            NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410, NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT, NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN),
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_VALID_SYCC_CSS420_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImage,
    Combine(
        Values(NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU),
        Values("/jpeg/padlock-406986_640_420.jpg"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_420),
        Values(NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_440, NVIMGCODEC_SAMPLING_422 , NVIMGCODEC_SAMPLING_420, 
            NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410, NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT, NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN),
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_VALID_SYCC_CSS422_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImage,
    Combine(
        Values(NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU),
        Values("/jpeg/padlock-406986_640_422.jpg"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_422),
        Values(NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_440, NVIMGCODEC_SAMPLING_422 , NVIMGCODEC_SAMPLING_420, 
            NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410, NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT, NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN),
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_VALID_SYCC_CSS440_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImage,
    Combine(
        Values(NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU),
        Values("/jpeg/padlock-406986_640_440.jpg"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_440),
        Values(NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_440, NVIMGCODEC_SAMPLING_422 , NVIMGCODEC_SAMPLING_420, 
            NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410, NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT, NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN),
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_VALID_GRAY_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImage,
    Combine(
        Values(NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU),
        ValuesIn(css_filenames),
        Values(NVIMGCODEC_COLORSPEC_GRAY, NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_Y),
        Values(NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT, NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN),
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_VALID_QUALITY_VALUES, NvJpegExtEncoderTestSingleImage,
    Combine(
        Values(NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU),
        Values("/jpeg/padlock-406986_640_444.jpg"),
        Values(NVIMGCODEC_COLORSPEC_SRGB),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_RGB),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT, NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN),
        Values(NVIMGCODEC_QUALITY_TYPE_QUALITY),
        Values(1, 10, 30, 50, 75, 95, 100, 85.1f, 85.6f, 85.5f), // quality value
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

/*-------------negative tests below-------------*/
INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_INVALID_OUTPUT_CHROMA_FOR_P_Y_FORMAT_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImage,
    Combine(
        Values(NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU),
        Values("/jpeg/padlock-406986_640_444.jpg"),
        Values(NVIMGCODEC_COLORSPEC_GRAY, NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_Y),
        Values(NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_SAMPLING_440, 
                NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410, NVIMGCODEC_SAMPLING_410V, NVIMGCODEC_SAMPLING_422, NVIMGCODEC_SAMPLING_420),
        Values(NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT, NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN),
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED | NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_INVALID_INPUT_CHROMA_FOR_P_Y_FORMAT_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImage,
    Combine(
        Values(NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU),
        Values("/jpeg/padlock-406986_640_444.jpg"),
        Values(NVIMGCODEC_COLORSPEC_GRAY, NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_Y),
        Values(NVIMGCODEC_SAMPLING_440, 
                NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410,  NVIMGCODEC_SAMPLING_410V,  NVIMGCODEC_SAMPLING_422, NVIMGCODEC_SAMPLING_420),
        Values(NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT, NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN),
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED | NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_INVALID_COLOR_SPEC_FOR_P_Y_FORMAT, NvJpegExtEncoderTestSingleImage,
    Combine(
        Values(NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU),
        Values("/jpeg/padlock-406986_640_444.jpg"),
        Values(NVIMGCODEC_COLORSPEC_SRGB, NVIMGCODEC_COLORSPEC_YCCK, NVIMGCODEC_COLORSPEC_CMYK),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_Y),
        Values(NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT),
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED | NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_INVALID_QUALITY_VALUES, NvJpegExtEncoderTestSingleImage,
    Combine(
        Values(NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU),
        Values("/jpeg/padlock-406986_640_444.jpg"),
        Values(NVIMGCODEC_COLORSPEC_SRGB),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_RGB),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT, NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN),
        Values(NVIMGCODEC_QUALITY_TYPE_QUALITY),
        Values(0.f, 101.f), // quality value
        Values(NVIMGCODEC_PROCESSING_STATUS_QUALITY_VALUE_UNSUPPORTED)
    )
);
INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_INVALID_QUALITY_TYPES, NvJpegExtEncoderTestSingleImage,
    Combine(
        Values(NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU),
        Values("/jpeg/padlock-406986_640_444.jpg"),
        Values(NVIMGCODEC_COLORSPEC_SRGB),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_RGB),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT, NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN),
        Values(
            NVIMGCODEC_QUALITY_TYPE_LOSSLESS,
            NVIMGCODEC_QUALITY_TYPE_QUANTIZATION_STEP,
            NVIMGCODEC_QUALITY_TYPE_PSNR,
            NVIMGCODEC_QUALITY_TYPE_SIZE_RATIO
        ),
        Values(0), // quality value
        Values(NVIMGCODEC_PROCESSING_STATUS_QUALITY_TYPE_UNSUPPORTED)
    )
);

// clang-format on

}} // namespace nvimgcodec::test
