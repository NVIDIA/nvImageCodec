/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <nvimgcodec.h>
#include <nvtiff.h>
#include <cstddef>
#include <cstdint>
#include "utils/library_version.h"

namespace nvtiff {

// Identify Hamamatsu JPEG IFDs by their MCU-start tag.
// This is a format hint; nvTIFF remains responsible for decode validation.
template <typename HasTag>
inline bool isHamamatsu(nvtiffCompression_t compression, HasTag has_tag)
{
    constexpr uint16_t ndpi_mcu_starts = 65426;
    return compression == NVTIFF_COMPRESSION_JPEG && has_tag(ndpi_mcu_starts);
}

using NvtiffVersion = nvimgcodec::LibraryVersion;

// Get the current nvTIFF library version
NvtiffVersion get_nvtiff_version();

// Resolve the (image_idx | bitstream_offset) view selection to a concrete IFD byte offset.
size_t resolve_ifd_selection(nvtiffStream_t nvtiff_stream, const nvimgcodecCodeStreamInfo_t& codestream_info);
size_t resolve_ifd_selection(nvtiffStream_t nvtiff_stream, const nvimgcodecCodeStreamDesc_t* code_stream);

}
