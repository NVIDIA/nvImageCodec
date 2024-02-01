/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <vector>
#include <array>

namespace nvimgcodec {

class JPEG2KParserPlugin
{
  public:
    explicit JPEG2KParserPlugin(const nvimgcodecFrameworkDesc_t* framework);
    nvimgcodecParserDesc_t* getParserDesc();

  private:
    struct Parser
    {
        Parser(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework);

        
        nvimgcodecStatus_t parseJP2(nvimgcodecIoStreamDesc_t* io_stream);
        nvimgcodecStatus_t parseCodeStream(nvimgcodecIoStreamDesc_t* io_stream);
        nvimgcodecStatus_t getImageInfo(
            nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t* code_stream);

        static nvimgcodecStatus_t static_destroy(nvimgcodecParser_t parser);

        static nvimgcodecStatus_t static_get_image_info(nvimgcodecParser_t parser,
            nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t* code_stream);

        const char* plugin_id_;
        const nvimgcodecFrameworkDesc_t* framework_;
        uint16_t num_components, CSiz;
        uint32_t height = 0, width;
        uint8_t bits_per_component;
        nvimgcodecColorSpec_t color_spec;
        uint32_t XSiz, YSiz, XOSiz, YOSiz;
        std::array<uint8_t, NVIMGCODEC_MAX_NUM_PLANES> XRSiz, YRSiz, Ssiz;
    };

    nvimgcodecStatus_t canParse(int* result, nvimgcodecCodeStreamDesc_t* code_stream);
    nvimgcodecStatus_t create(nvimgcodecParser_t* parser);

    static nvimgcodecStatus_t static_can_parse(
        void* instance, int* result, nvimgcodecCodeStreamDesc_t* code_stream);
    static nvimgcodecStatus_t static_create(void* instance, nvimgcodecParser_t* parser);

    static constexpr const char* plugin_id_ = "jpeg2k_parser";
    const nvimgcodecFrameworkDesc_t* framework_;
    nvimgcodecParserDesc_t parser_desc_;
};

nvimgcodecStatus_t get_jpeg2k_parser_extension_desc(nvimgcodecExtensionDesc_t* ext_desc);

}  // namespace nvimgcodec