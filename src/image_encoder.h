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
#include <memory>
#include <string>
#include "iimage_encoder.h"

namespace nvimgcodec {

class IEncodeState;
class IImage;
class ICodeStream;

class ImageEncoder : public IImageEncoder
{
  public:
    ImageEncoder(const nvimgcodecEncoderDesc_t* desc, const nvimgcodecExecutionParams_t* exec_params, const char* options);
    ~ImageEncoder() override;
    nvimgcodecBackendKind_t getBackendKind() const override;
    std::unique_ptr<IEncodeState> createEncodeStateBatch() const override;
    bool canEncode(const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecImageDesc_t* image, const nvimgcodecEncodeParams_t* params,
        nvimgcodecProcessingStatus_t* status, int thread_idx) const override;
    bool encode(const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecImageDesc_t* image,
        const nvimgcodecEncodeParams_t* params, nvimgcodecProcessingStatus_t* status, int thread_idx) override;
    const char* encoderId() const override;

  private:
    const nvimgcodecEncoderDesc_t* encoder_desc_;
    nvimgcodecEncoder_t encoder_;
};

} // namespace nvimgcodec