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

#pragma once

#include "nvimgcodec.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "imgproc/convert.h"

namespace opencv {

nvimgcodecStatus_t convertToCvMat(const nvimgcodecImageInfo_t& info, cv::Mat& decoded);
nvimgcodecStatus_t convertFromCvMat(nvimgcodecImageInfo_t& info, const cv::Mat& decoded);

void colorConvert(cv::Mat& img, cv::ColorConversionCodes conversion);
nvimgcodecStatus_t getOpencvDataType(int *type, nvimgcodecImageInfo_t info);

/**
 * @brief Get the nvimgcodecJpegEncodeParams_t structure from the nvimgcodecEncodeParams_t chain
 *
 * @return Pointer to the first node that is a nvimgcodecJpegEncodeParams_t if it exists, else nullptr
 */
const nvimgcodecJpegEncodeParams_t *getJpegEncodeParams(const nvimgcodecEncodeParams_t *encode_params);
/**
 * @brief Get the nvimgcodecJpeg2kEncodeParams_t structure from the nvimgcodecEncodeParams_t chain
 *
 * @return Pointer to the first node that is a nvimgcodecJpeg2kEncodeParams_t if it exists, else nullptr
 */
const nvimgcodecJpeg2kEncodeParams_t *getJpeg2kEncodeParams(const nvimgcodecEncodeParams_t *encode_params);

}