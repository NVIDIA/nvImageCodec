
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
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace nvimgcodec { namespace test {

cv::Mat rgb2bgr(const cv::Mat& img);
cv::Mat bgr2rgb(const cv::Mat& img);

int write_bmp(const char* filename, const unsigned char* d_chanR, int pitchR, const unsigned char* d_chanG, int pitchG,
    const unsigned char* d_chanB, int pitchB, int width, int height);

}} // namespace nvimgcodec::test
