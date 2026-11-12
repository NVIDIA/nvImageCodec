/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <tiffio.h>

#define XM_CHECK_NULL(ptr)                            \
    {                                                 \
        if (!ptr)                                     \
            throw std::runtime_error("null pointer"); \
    }

#define LIBTIFF_CALL_SUCCESS 1
#define LIBTIFF_CALL(call)                                                                                     \
    do {                                                                                                       \
        int retcode = (call);                                                                                  \
        if (LIBTIFF_CALL_SUCCESS != retcode)                                                                   \
            throw std::runtime_error("libtiff call failed with code " + std::to_string(retcode) + ": " #call); \
    } while (0)

// GeoTIFF/GDAL tag IDs that libtiff does not recognise natively.
// Decoding is correct; the tags carry only geographic metadata.
static constexpr uint32_t kGeoTIFFTags[] = {
    33550,  // ModelPixelScaleTag
    33922,  // ModelTiepointTag
    34264,  // ModelTransformationTag
    34735,  // GeoKeyDirectoryTag
    34736,  // GeoDoubleParamsTag
    34737,  // GeoAsciiParamsTag
    42112,  // GDAL_METADATA
    42113,  // GDAL_NODATA
};

// TIFFWarningHandler that silently drops "Unknown field with tag X" warnings for
// the known GeoTIFF/GDAL tags and forwards all other libtiff warnings to stderr.
inline void SuppressGeoTIFFTagWarnings(const char* module, const char* fmt, va_list ap)
{
    if (strstr(fmt, "Unknown field with tag") != nullptr) {
        va_list ap_copy;
        va_copy(ap_copy, ap);
        unsigned int tag = va_arg(ap_copy, unsigned int);
        va_end(ap_copy);
        for (auto geotiff_tag : kGeoTIFFTags) {
            if (tag == geotiff_tag)
                return;
        }
    }
    char buf[1024];
    vsnprintf(buf, sizeof(buf), fmt, ap);
    // libtiff permits a null module name in some code paths
    if (module)
        fprintf(stderr, "%s: %s\n", module, buf);
    else
        fprintf(stderr, "%s\n", buf);
}

// Declared inline so the static once_flag is shared across all translation units,
// guaranteeing TIFFSetWarningHandler is called exactly once per process.
inline void InstallGeoTIFFWarningFilter()
{
    static std::once_flag flag;
    std::call_once(flag, [] { TIFFSetWarningHandler(SuppressGeoTIFFTagWarnings); });
}
