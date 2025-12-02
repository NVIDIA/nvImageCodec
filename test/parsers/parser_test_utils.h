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

#include <fstream>
#include <string>
#include <vector>

namespace nvimgcodec {
namespace test {

inline std::vector<uint8_t> read_file(const std::string &filename) {
    std::vector<uint8_t> buffer;
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return buffer;
    }
    std::streamsize size = file.tellg();
    if (size == 0) {
        return buffer;
    }
    file.seekg(0, std::ios::beg);
    buffer.resize(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        buffer.resize(0);
        return buffer;
    }
  return buffer;
}

inline std::vector<uint8_t> replace(const std::vector<uint8_t>& data, const std::vector<uint8_t>& old_value, const std::vector<uint8_t>& new_value)
{
    std::vector<uint8_t> result;
    result.reserve(data.size());
    auto it = data.begin();
    size_t n = old_value.size();
    while (it != data.end()) {
        if (it + n <= data.end() && std::equal(it, it + n, old_value.begin(), old_value.end())) {
            result.insert(result.end(), new_value.begin(), new_value.end());
            it += n;
        } else {
            result.push_back(*(it++));
        }
    }
    return result;
}

inline void expect_eq(nvimgcodecImageInfo_t expected, nvimgcodecImageInfo_t actual) {
    EXPECT_EQ(expected.struct_type, actual.struct_type);
    EXPECT_EQ(expected.struct_size, actual.struct_size);
    EXPECT_EQ(expected.struct_next, actual.struct_next);
    EXPECT_EQ(expected.sample_format, actual.sample_format);
    EXPECT_EQ(expected.num_planes, actual.num_planes);
    EXPECT_EQ(expected.color_spec, actual.color_spec);
    EXPECT_EQ(expected.chroma_subsampling, actual.chroma_subsampling);
    EXPECT_EQ(expected.orientation.rotated, actual.orientation.rotated);
    EXPECT_EQ(expected.orientation.flip_x, actual.orientation.flip_x);
    EXPECT_EQ(expected.orientation.flip_y, actual.orientation.flip_y);
    for (uint32_t p = 0; p < expected.num_planes; p++) {
        EXPECT_EQ(expected.plane_info[p].height, actual.plane_info[p].height);
        EXPECT_EQ(expected.plane_info[p].width, actual.plane_info[p].width);
        EXPECT_EQ(expected.plane_info[p].num_channels, actual.plane_info[p].num_channels);
        EXPECT_EQ(expected.plane_info[p].sample_type, actual.plane_info[p].sample_type);
        EXPECT_EQ(expected.plane_info[p].precision, actual.plane_info[p].precision);
    }
}

inline void LoadImageFromFilename(nvimgcodecInstance_t instance, nvimgcodecCodeStream_t& stream_handle, const std::string& filename)
{
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamCreateFromFile(instance, &stream_handle, filename.c_str()));
}

inline void LoadImageFromHostMemory(nvimgcodecInstance_t instance, nvimgcodecCodeStream_t& stream_handle, const uint8_t* data, size_t data_size)
{
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamCreateFromHostMem(instance, &stream_handle, data, data_size));
}

}  // namespace test
}  // namespace nvimgcodec