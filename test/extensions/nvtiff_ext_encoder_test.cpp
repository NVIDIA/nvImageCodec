#include <extensions/nvtiff/nvtiff_ext.h>
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
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "nvimgcodec_tests.h"
#include "common_ext_encoder_test.h"

namespace nvimgcodec { namespace test {

class NvTiffExtEncoderTestBase : public CommonExtEncoderTest
{
public:
    void SetUp() override
    {
        CommonExtEncoderTest::SetUp();

        nvimgcodecExtensionDesc_t tiff_parser_extension_desc{
            NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC,
            sizeof(nvimgcodecExtensionDesc_t),
            0
        };
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
            get_tiff_parser_extension_desc(&tiff_parser_extension_desc));
        extensions_.emplace_back();
        nvimgcodecExtensionCreate(instance_, &extensions_.back(), &tiff_parser_extension_desc);

        nvimgcodecExtensionDesc_t nvtiff_extension_desc{
            NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC,
            sizeof(nvimgcodecExtensionDesc_t),
            0
        };
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_nvtiff_extension_desc(&nvtiff_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &nvtiff_extension_desc));

        CommonExtEncoderTest::CreateDecoderAndEncoder();
    }

    void TearDown() override
    {
        CommonExtEncoderTest::TearDown();
    }
};

class NvTiffExtEncoderRandomTest:
    public NvTiffExtEncoderTestBase,
    public ::testing::TestWithParam<std::tuple<nvimgcodecSampleFormat_t, 
                                               nvimgcodecChromaSubsampling_t, 
                                               nvimgcodecQualityType_t,
                                               nvimgcodecProcessingStatus_t>>
{
public:
    void SetUp() override
    {
        NvTiffExtEncoderTestBase::SetUp();
    }

    void TearDown() override
    {
        NvTiffExtEncoderTestBase::TearDown();
    }

    void TestEncodeSingleImage(nvimgcodecSampleFormat_t sample_format,
                               nvimgcodecChromaSubsampling_t subsampling,
                               nvimgcodecQualityType_t quality_type,
                               nvimgcodecProcessingStatus_t expected_status)
    {
        image_info_.plane_info[0].width = image_width_;
        image_info_.plane_info[0].height = image_height_;
        image_info_.plane_info[0].precision = 8;
        image_info_.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        sample_format_ = sample_format;
        chroma_subsampling_ = subsampling;
        color_spec_ = sample_format == NVIMGCODEC_SAMPLEFORMAT_P_Y ? NVIMGCODEC_COLORSPEC_GRAY : NVIMGCODEC_COLORSPEC_SRGB;
        PrepareImageForFormat();
        genRandomImage();

        image_info_.buffer = image_buffer_.data();
        strcpy(image_info_.codec_name, "tiff");

        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &in_image_, &image_info_));

        nvimgcodecImageInfo_t encoded_image_info(image_info_);
        strcpy(encoded_image_info.codec_name, "tiff");
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamCreateToHostMem(instance_, &out_code_stream_, (void*)this, &ResizeBufferStatic<CommonExtEncoderTest>, &encoded_image_info));
        
        encode_params_.quality_type = quality_type;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderEncode(encoder_, &in_image_, &out_code_stream_, 1, &encode_params_, &future_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future_));

        size_t status_size;
        nvimgcodecProcessingStatus_t encode_status;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future_, &encode_status, &status_size));
        ASSERT_EQ(encode_status, expected_status);
    }
};

class NvTiffExtEncoderTest :
    public NvTiffExtEncoderTestBase,
    public ::testing::TestWithParam<std::tuple<const char*, // file name
                                               nvimgcodecQualityType_t,
                                               float>> // quality value
{
    void SetUp() override
    {
        NvTiffExtEncoderTestBase::SetUp();
    }

    void TearDown() override
    {
        NvTiffExtEncoderTestBase::TearDown();
    }
};

TEST_P(NvTiffExtEncoderRandomTest, RandomImage)
{
    nvimgcodecSampleFormat_t sample_format = std::get<0>(GetParam());
    nvimgcodecChromaSubsampling_t subsampling = std::get<1>(GetParam());
    nvimgcodecQualityType_t quality_type = std::get<2>(GetParam());
    nvimgcodecProcessingStatus_t expected_status = std::get<3>(GetParam());
    TestEncodeSingleImage(sample_format, subsampling, quality_type, expected_status);
}

TEST_P(NvTiffExtEncoderTest, SingleImage)
{
    image_file_ = resources_dir + "/" + std::get<0>(GetParam());
    nvimgcodecQualityType_t quality_type = std::get<1>(GetParam());
    float quality_value = std::get<2>(GetParam());
    LoadImageFromFilename(instance_, in_code_stream_, image_file_);
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(in_code_stream_, &image_info_));
    int num_channels = 0;
    if (image_info_.num_planes == 1) {
        image_info_.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_Y;
        image_info_.color_spec = NVIMGCODEC_COLORSPEC_GRAY;
        image_info_.chroma_subsampling = NVIMGCODEC_SAMPLING_GRAY;
        num_channels = 1;
    } else if (image_info_.num_planes == 3) {
        image_info_.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        image_info_.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        image_info_.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        num_channels = 3;
    }
    uint32_t& width = image_info_.plane_info[0].width;
    uint32_t& height = image_info_.plane_info[0].height;


    image_info_.num_planes = 1;
    image_info_.plane_info[0].width = width;
    image_info_.plane_info[0].height = height;
    image_info_.plane_info[0].row_stride = width * num_channels * (image_info_.plane_info[0].sample_type >> 11);
    image_info_.plane_info[0].num_channels = num_channels;
    image_info_.plane_info[0].precision = 0; // Compute precision based on sample_type.
    image_info_.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&image_info_.buffer, GetBufferSize(image_info_)));
    image_buffer_.resize(GetBufferSize(image_info_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &in_image_, &image_info_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDecode(decoder_, &in_code_stream_, &in_image_, 1, &decode_params_, &future_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future_));
    nvimgcodecProcessingStatus_t status;
    size_t status_size;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future_, &status, &status_size));
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, status);
    
    ASSERT_EQ(cudaSuccess, cudaMemcpy(image_buffer_.data(), image_info_.buffer, GetBufferSize(image_info_), cudaMemcpyDeviceToHost));

    // Encode
    nvimgcodecImageInfo_t encoded_image_info(image_info_);
    strcpy(encoded_image_info.codec_name, "tiff");
    // For single-channel images, ensure chroma_subsampling is set to GRAY
    if (num_channels == 1) {
        encoded_image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_GRAY;
    }
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &out_image_, &encoded_image_info));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamCreateToHostMem(instance_, &out_code_stream_, (void*)this, &ResizeBufferStatic<CommonExtEncoderTest>, &encoded_image_info));
    encode_params_.quality_type = quality_type;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderEncode(encoder_, &out_image_, &out_code_stream_, 1, &encode_params_, &future_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future_));
    nvimgcodecProcessingStatus_t encode_status;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future_, &encode_status, &status_size));
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, status);

    ASSERT_EQ(cudaSuccess, cudaFree(image_info_.buffer));

    // Decode back and compare
    LoadImageFromHostMemory(instance_, in_code_stream_, code_stream_buffer_.data(), code_stream_buffer_.size());
    std::vector<unsigned char> out_buf = image_buffer_;
    image_info_.buffer = out_buf.data();
    image_info_.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &in_image_, &image_info_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDecode(decoder_, &in_code_stream_, &in_image_, 1, &decode_params_, &future_));
    
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future_, &status, &status_size));
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, status);

    for (size_t i = 0; i < image_buffer_.size(); ++i) {
        ASSERT_EQ(image_buffer_[i], out_buf[i]);
    }

    // Make sure the compression setting based on quality type works as expected
    // Maybe there is a bug in nvtiff LZW, we need a very smooth image for the compressed image to be smaller than the raw image.
    if (strcmp(std::get<0>(GetParam()), "tiff/ycck_colorspace.tiff") == 0) {
        size_t compressed_size = code_stream_buffer_.size();
        size_t raw_size = GetBufferSize(image_info_);
        if (quality_type == NVIMGCODEC_QUALITY_TYPE_DEFAULT) { // LZW
            ASSERT_LT(compressed_size, raw_size); 
        } else if (quality_type == NVIMGCODEC_QUALITY_TYPE_LOSSLESS) {
            if (quality_value == 0) { // none compression
                ASSERT_LE(raw_size, compressed_size); // the compressed_size also includes tiff header
            } else if (quality_value == 1) { // LZW
                ASSERT_LT(compressed_size, raw_size); 
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(NVTIFF_ENCODE,
    NvTiffExtEncoderTest,
    ::testing::Combine(
        ::testing::Values(
            "tiff/cat-300572_640_no_compression.tiff",
            "tiff/ycck_colorspace.tiff"
        ),
        ::testing::Values(
            NVIMGCODEC_QUALITY_TYPE_DEFAULT,
            NVIMGCODEC_QUALITY_TYPE_LOSSLESS
        ),
        ::testing::Values(
            0.0f,
            0.1f
        )
    )
);

#if !SKIP_NVTIFF_WITH_NVCOMP_TESTS_ENABLED
    INSTANTIATE_TEST_SUITE_P(NVTIFF_ENCODE_MORE_DATA_TYPES,
        NvTiffExtEncoderTest,
        ::testing::Combine(
            ::testing::Values(
                "tiff/cat-300572_640_uint16.tiff",
                "tiff/cat-300572_640_fp32.tiff",
                "tiff/cat-300572_640_uint32.tiff",
                "tiff/cat-300572_640_grayscale.tiff"
            ),
            ::testing::Values(
                NVIMGCODEC_QUALITY_TYPE_DEFAULT,
                NVIMGCODEC_QUALITY_TYPE_LOSSLESS
            ),
            ::testing::Values(
                0.0f,
                0.1f
            )
        )
);
#endif

INSTANTIATE_TEST_SUITE_P(NVTIFF_ENCODE_RANDOM_GOOD,
    NvTiffExtEncoderRandomTest,
    ::testing::Combine(
        ::testing::Values(
            NVIMGCODEC_SAMPLEFORMAT_I_RGB,
            NVIMGCODEC_SAMPLEFORMAT_I_BGR,
            NVIMGCODEC_SAMPLEFORMAT_P_Y
        ),
        ::testing::Values(
            NVIMGCODEC_SAMPLING_NONE
        ),
        ::testing::Values(
            NVIMGCODEC_QUALITY_TYPE_DEFAULT,
            NVIMGCODEC_QUALITY_TYPE_LOSSLESS
        ),
        ::testing::Values(
            NVIMGCODEC_PROCESSING_STATUS_SUCCESS
        )
    )
);


INSTANTIATE_TEST_SUITE_P(NVTIFF_ENCODE_RANDOM_UNSUPPORTED_FORMATS,
    NvTiffExtEncoderRandomTest,
    ::testing::Combine(
        ::testing::Values(
            NVIMGCODEC_SAMPLEFORMAT_P_RGB,
            NVIMGCODEC_SAMPLEFORMAT_P_BGR,
            NVIMGCODEC_SAMPLEFORMAT_P_YUV
        ),
        ::testing::Values(
            NVIMGCODEC_SAMPLING_NONE
        ),
        ::testing::Values(
            NVIMGCODEC_QUALITY_TYPE_DEFAULT
        ),
        ::testing::Values(
            NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED |
            NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED
        )
    )
);

INSTANTIATE_TEST_SUITE_P(NVTIFF_ENCODE_RANDOM_UNSUPPORTED_SAMPLING,
    NvTiffExtEncoderRandomTest,
    ::testing::Combine(
        ::testing::Values(
            NVIMGCODEC_SAMPLEFORMAT_I_RGB 
        ),
        ::testing::Values(
            NVIMGCODEC_SAMPLING_422,
            NVIMGCODEC_SAMPLING_420,
            NVIMGCODEC_SAMPLING_440,
            NVIMGCODEC_SAMPLING_411,
            NVIMGCODEC_SAMPLING_410,
            NVIMGCODEC_SAMPLING_GRAY,
            NVIMGCODEC_SAMPLING_410V
        ),
        ::testing::Values(
            NVIMGCODEC_QUALITY_TYPE_DEFAULT
        ),
        ::testing::Values(
            NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED
        )
    )
);

INSTANTIATE_TEST_SUITE_P(NVTIFF_ENCODE_RANDOM_UNSUPPORTED_QUALITY_TYPE,
    NvTiffExtEncoderRandomTest,
    ::testing::Combine(
        ::testing::Values(
            NVIMGCODEC_SAMPLEFORMAT_I_RGB
        ),
        ::testing::Values(
            NVIMGCODEC_SAMPLING_NONE
        ),
        ::testing::Values(
            NVIMGCODEC_QUALITY_TYPE_QUALITY,
            NVIMGCODEC_QUALITY_TYPE_QUANTIZATION_STEP,
            NVIMGCODEC_QUALITY_TYPE_PSNR,
            NVIMGCODEC_QUALITY_TYPE_SIZE_RATIO
        ),
        ::testing::Values(
            NVIMGCODEC_PROCESSING_STATUS_QUALITY_TYPE_UNSUPPORTED
        )
    )
);

}} // namespace nvimgcodec::test