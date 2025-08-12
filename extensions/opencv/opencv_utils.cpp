/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "opencv_utils.h"
#include "imgproc/sample_format_utils.h"

namespace opencv {

void colorConvert(cv::Mat& img, cv::ColorConversionCodes conversion)
{
    if (img.data == nullptr || img.rows == 0 || img.cols == 0)
        throw std::runtime_error("Invalid input image");
    cv::cvtColor(img, img, conversion);
}

template <typename DestType, typename SrcType>
nvimgcodecStatus_t convertPlanarFromCvMat(DestType* destinationBuffer, uint32_t plane_stride, uint32_t row_stride_bytes, const cv::Mat& source_image)
{
    using nvimgcodec::ConvertSatNorm;
    std::vector<cv::Mat> planes;
    cv::split(source_image, planes);
    size_t height = source_image.size[0];
    size_t width = source_image.size[1];
    for (size_t ch = 0; ch < planes.size(); ++ch) {
        const cv::Mat& srcPlane = planes[ch];
        const SrcType* srcPlanePtr = srcPlane.ptr<SrcType>();
        DestType* destPlanePtr = destinationBuffer + ch * plane_stride;
        for (size_t i = 0; i < height; ++i) {
            const SrcType* srcRow = srcPlanePtr + i * width;
            DestType* destRow = reinterpret_cast<DestType*>(reinterpret_cast<uint8_t*>(destPlanePtr) + i * row_stride_bytes);
            for (size_t j = 0; j < width; ++j) {
                destRow[j] = ConvertSatNorm<DestType>(srcRow[j]);
            }
        }
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

template <typename DestType, typename SrcType>
nvimgcodecStatus_t convertInterleavedFromCvMat(DestType* destinationBuffer, uint32_t row_stride_bytes, const cv::Mat& source_image)
{
    using nvimgcodec::ConvertSatNorm;
    size_t height = source_image.size[0];
    size_t width = source_image.size[1];
    size_t channels = source_image.channels();
    for (size_t i = 0; i < height; ++i) {
        const SrcType* srcRow = source_image.ptr<SrcType>() + i * width * channels;
        DestType* destRow = reinterpret_cast<DestType*>(reinterpret_cast<uint8_t*>(destinationBuffer) + i * row_stride_bytes);
        for (size_t j = 0; j < width; ++j) {
            for (size_t c = 0; c < channels; c++) {
                destRow[j * channels + c] = ConvertSatNorm<DestType>(srcRow[j * channels + c]);
            }
        }
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t convertPlanarFromCvMat(const nvimgcodecImageInfo_t& destination, const cv::Mat& source)
{

#define CaseConvertPlanar(OUT_SAMPLE_TYPE, OutType)                                                  \
    case OUT_SAMPLE_TYPE:                                                                            \
        switch (source.depth()) {                                                                    \
        case CV_8U:                                                                                  \
            return convertPlanarFromCvMat<OutType, uint8_t>(                                         \
                reinterpret_cast<OutType*>(destination.buffer),                                      \
                destination.plane_info[0].row_stride * destination.plane_info[0].height,             \
                destination.plane_info[0].row_stride,                                                \
                source);                                                                             \
        case CV_16U:                                                                                 \
            return convertPlanarFromCvMat<OutType, uint16_t>(                                        \
                reinterpret_cast<OutType*>(destination.buffer),                                      \
                destination.plane_info[0].row_stride * destination.plane_info[0].height,             \
                destination.plane_info[0].row_stride,                                                \
                source);                                                                             \
        default:                                                                                     \
            return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;                                     \
        }                                                                                            \
        break;

    switch (destination.plane_info[0].sample_type) {
        CaseConvertPlanar(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, uint8_t);
        CaseConvertPlanar(NVIMGCODEC_SAMPLE_DATA_TYPE_INT8, int8_t);
        CaseConvertPlanar(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16, uint16_t);
        CaseConvertPlanar(NVIMGCODEC_SAMPLE_DATA_TYPE_INT16, int16_t);
        CaseConvertPlanar(NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32, float);
    default:
        return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
    }
#undef CaseConvertPlanar
}

nvimgcodecStatus_t convertInterleavedFromCvMat(const nvimgcodecImageInfo_t& destination, const cv::Mat& source)
{

#define CaseConvertInterleaved(OUT_SAMPLE_TYPE, OutType)                                                       \
    case OUT_SAMPLE_TYPE:                                                                                      \
        switch (source.depth()) {                                                                              \
        case CV_8U:                                                                                            \
            return convertInterleavedFromCvMat<OutType, uint8_t>(                                              \
                reinterpret_cast<OutType*>(destination.buffer),destination.plane_info[0].row_stride, source);  \
        case CV_16U:                                                                                           \
            return convertInterleavedFromCvMat<OutType, uint16_t>(                                             \
                reinterpret_cast<OutType*>(destination.buffer), destination.plane_info[0].row_stride, source); \
        default:                                                                                               \
            return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;                                               \
        }                                                                                                      \
        break;

    switch (destination.plane_info[0].sample_type) {
        CaseConvertInterleaved(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, uint8_t);
        CaseConvertInterleaved(NVIMGCODEC_SAMPLE_DATA_TYPE_INT8, int8_t);
        CaseConvertInterleaved(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16, uint16_t);
        CaseConvertInterleaved(NVIMGCODEC_SAMPLE_DATA_TYPE_INT16, int16_t);
        CaseConvertInterleaved(NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32, float);
    default:
        return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
    }

#undef CaseConvertInterleaved
}

nvimgcodecStatus_t convertFromCvMat(nvimgcodecImageInfo_t& destination, const cv::Mat& source)
{
    if (destination.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_RGB || destination.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_BGR ||
        destination.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED) {
        return convertPlanarFromCvMat(destination, source);
    } else {
        return convertInterleavedFromCvMat(destination, source);
    }
}

template <typename SrcType, typename DstType>
nvimgcodecStatus_t convertPlanarToCvMat(const nvimgcodecImageInfo_t& source, cv::Mat& destination_image)
{
    SrcType* sourceBuffer = static_cast<SrcType*>(source.buffer);
    uint32_t plane_stride = source.plane_info[0].row_stride * source.plane_info[0].height;
    uint32_t row_stride_bytes = source.plane_info[0].row_stride;

    using nvimgcodec::ConvertSatNorm;
    size_t height = destination_image.size[0];
    size_t width = destination_image.size[1];
    size_t channels = destination_image.channels();
    for (size_t ch = 0; ch < channels; ++ch) {
        const SrcType* srcPlanePtr = sourceBuffer + ch * plane_stride;
        for (size_t i = 0; i < height; ++i) {
            DstType* destRow = destination_image.ptr<DstType>() + i * width * channels;
            const SrcType* srcRow = reinterpret_cast<const SrcType*>(reinterpret_cast<const uint8_t*>(srcPlanePtr) + i * row_stride_bytes);
            for (size_t j = 0; j < width; ++j) {
                destRow[j * channels + ch] = ConvertSatNorm<DstType>(srcRow[j]);
            }
        }
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t convertPlanarToCvMat(const nvimgcodecImageInfo_t& source, cv::Mat& destination)
{

#define CaseConvertPlanar(OUT_SAMPLE_TYPE, OutType)                                                     \
    case OUT_SAMPLE_TYPE:                                                                               \
        switch (destination.depth()) {                                                                  \
        case CV_8U:                                                                                     \
            return convertPlanarToCvMat<uint8_t, OutType>(source, destination);                         \
        case CV_16U:                                                                                    \
            return convertPlanarToCvMat<uint16_t, OutType>(source, destination);                        \
        case CV_32F:                                                                                    \
            return convertPlanarToCvMat<float, OutType>(source, destination);                           \
        default:                                                                                        \
            return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;                                        \
        }                                                                                               \
        break;

    switch (source.plane_info[0].sample_type) {
        CaseConvertPlanar(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, uint8_t);
        CaseConvertPlanar(NVIMGCODEC_SAMPLE_DATA_TYPE_INT8, int8_t);
        CaseConvertPlanar(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16, uint16_t);
        CaseConvertPlanar(NVIMGCODEC_SAMPLE_DATA_TYPE_INT16, int16_t);
        CaseConvertPlanar(NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32, float);
    default:
        return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
    }

#undef CaseConvertInterleaved
}

template <typename SrcType, typename DstType>
nvimgcodecStatus_t convertKeepInterleaved(const nvimgcodecImageInfo_t& source, cv::Mat& destination_image)
{
    SrcType* sourceBuffer = static_cast<SrcType*>(source.buffer);
    uint32_t row_stride_bytes = source.plane_info[0].row_stride;

    using nvimgcodec::ConvertSatNorm;
    size_t height = destination_image.size[0];
    size_t width = destination_image.size[1];
    size_t channels = destination_image.channels();
    for (size_t i = 0; i < height; ++i) {
        DstType* destRow = destination_image.ptr<DstType>() + i * width * channels;
        const SrcType* srcRow = reinterpret_cast<const SrcType*>(reinterpret_cast<const uint8_t*>(sourceBuffer) + i * row_stride_bytes);
        for (size_t j = 0; j < width * channels; ++j) {
            destRow[j] = ConvertSatNorm<DstType>(srcRow[j]);
        }
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t convertInterleavedToCvMat(const nvimgcodecImageInfo_t& source, cv::Mat& destination)
{

#define CaseConvertInterleaved(OUT_SAMPLE_TYPE, OutType)                               \
    case OUT_SAMPLE_TYPE:                                                              \
        switch (destination.depth()) {                                                 \
        case CV_8U:                                                                    \
            return convertKeepInterleaved<OutType, uint8_t>(source, destination);      \
        case CV_16U:                                                                   \
            return convertKeepInterleaved<OutType, uint16_t>(source, destination);     \
        case CV_32F:                                                                   \
            return convertKeepInterleaved<OutType, float>(source, destination);        \
        default:                                                                       \
            return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;                       \
        }                                                                              \
        break;

    switch (source.plane_info[0].sample_type) {
        CaseConvertInterleaved(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, uint8_t);
        CaseConvertInterleaved(NVIMGCODEC_SAMPLE_DATA_TYPE_INT8, int8_t);
        CaseConvertInterleaved(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16, uint16_t);
        CaseConvertInterleaved(NVIMGCODEC_SAMPLE_DATA_TYPE_INT16, int16_t);
        CaseConvertInterleaved(NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32, float);
    default:
        return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
    }

#undef CaseConvertInterleaved
}

nvimgcodecStatus_t convertToCvMat(const nvimgcodecImageInfo_t& source, cv::Mat& destination)
{
    if (nvimgcodec::IsPlanar(source.sample_format)) {
        return convertPlanarToCvMat(source, destination);
    } else {
        return convertInterleavedToCvMat(source, destination);
    }
}

nvimgcodecStatus_t getOpencvDataType(int *type, const nvimgcodecImageInfo_t& info) {

#define CaseBitSize(type_opencv, bits_opencv, type_nvimgcodec, bits_nvimgcodec) \
    case NVIMGCODEC_SAMPLE_DATA_TYPE_##type_nvimgcodec##bits_nvimgcodec:        \
        switch(num_channels) {                                                  \
            case 1:                                                             \
                *type = CV_##bits_opencv##type_opencv##C1;                      \
                break;                                                          \
            case 2:                                                             \
                *type = CV_##bits_opencv##type_opencv##C2;                      \
                break;                                                          \
            case 3:                                                             \
                *type = CV_##bits_opencv##type_opencv##C3;                      \
                break;                                                          \
            case 4:                                                             \
                *type = CV_##bits_opencv##type_opencv##C4;                      \
                break;                                                          \
            default:                                                            \
                return NVIMGCODEC_STATUS_INVALID_PARAMETER;                     \
        }                                                                       \
        break;

    int num_channels = std::max(info.num_planes, info.plane_info[0].num_channels);

    switch(info.plane_info[0].sample_type) {
        CaseBitSize(U, 8, UINT, 8)
        CaseBitSize(S, 8, INT, 8)
        CaseBitSize(U, 16, UINT, 16)
        CaseBitSize(S, 16, INT, 16)
        CaseBitSize(S, 32, INT, 32)
        CaseBitSize(F, 32, FLOAT, 32)
        CaseBitSize(F, 64, FLOAT, 64)
    default:
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

const nvimgcodecJpegEncodeParams_t *getJpegEncodeParams(const nvimgcodecEncodeParams_t *encode_params) {
    for(
            const nvimgcodecJpegEncodeParams_t *jpeg_encode_params = reinterpret_cast<const nvimgcodecJpegEncodeParams_t*>(encode_params) ;
            jpeg_encode_params ;
            jpeg_encode_params=reinterpret_cast<nvimgcodecJpegEncodeParams_t*>(jpeg_encode_params->struct_next)) {
        if(jpeg_encode_params->struct_type == NVIMGCODEC_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS) {
            return jpeg_encode_params;
        }
    }
    
    return nullptr;
}

const nvimgcodecJpeg2kEncodeParams_t *getJpeg2kEncodeParams(const nvimgcodecEncodeParams_t *encode_params) {
    for(
            const nvimgcodecJpeg2kEncodeParams_t *jpeg2k_encode_params = reinterpret_cast<const nvimgcodecJpeg2kEncodeParams_t*>(encode_params) ;
            jpeg2k_encode_params ;
            jpeg2k_encode_params=reinterpret_cast<nvimgcodecJpeg2kEncodeParams_t*>(jpeg2k_encode_params->struct_next)) {
        if(jpeg2k_encode_params->struct_type == NVIMGCODEC_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS) {
            return jpeg2k_encode_params;
        }
    }
    
    return nullptr;
}

}
