/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace nvtiff {

class MetadataExtractor {
public:
    class IMetadataSetExtractor {
    public:
        virtual ~IMetadataSetExtractor() = default;
        virtual bool extract(const nvimgcodecCodeStreamDesc_t* code_stream) = 0;
        virtual nvimgcodecStatus_t getMetadata(const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t* metadata, int index) = 0;
    };

    MetadataExtractor(const nvimgcodecFrameworkDesc_t* framework, const char* plugin_id, nvtiffStream_t* nvtiff_stream);
    MetadataExtractor(const MetadataExtractor& other) = delete;
    MetadataExtractor& operator=(const MetadataExtractor& other) = delete;
    MetadataExtractor(MetadataExtractor&& other) noexcept;
    MetadataExtractor& operator=(MetadataExtractor&& other) noexcept;
    ~MetadataExtractor() = default;

    size_t getMetadataCount(const nvimgcodecCodeStreamDesc_t* code_stream);
    nvimgcodecStatus_t getMetadata(const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t* metadata, int index);
    std::string getMetadata(const nvimgcodecCodeStreamDesc_t* code_stream, int index);
private:
    const nvimgcodecFrameworkDesc_t* framework_;
    const char* plugin_id_;
    nvtiffStream_t* nvtiff_stream_;
    std::vector<std::unique_ptr<IMetadataSetExtractor>> metadata_set_extractors_;
    std::vector<IMetadataSetExtractor*> active_extractors_;
};

} // namespace nvtiff 