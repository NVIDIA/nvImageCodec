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
#include <memory>
#include <vector>
#include <string>

namespace opencv {

class OpenCVEncoderPlugin
{
  public:
    explicit OpenCVEncoderPlugin(const std::string& codec_name, const nvimgcodecFrameworkDesc_t* framework);
    const nvimgcodecEncoderDesc_t* getEncoderDesc() const;

  private:
    struct Encoder;

    nvimgcodecStatus_t create(nvimgcodecEncoder_t* encoder, const nvimgcodecExecutionParams_t* exec_params, const char* options);
    static nvimgcodecStatus_t static_create(void* instance, nvimgcodecEncoder_t* encoder, const nvimgcodecExecutionParams_t* exec_params, const char* options);

    std::string codec_name_;
    std::string plugin_id_;
    nvimgcodecEncoderDesc_t encoder_desc_;
    
    const nvimgcodecFrameworkDesc_t* framework_;
};

} // namespace opencv
