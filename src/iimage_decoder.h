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
#include <nvimgcodec.h>
#include <memory>
#include <string>
#include <vector>
#include "processing_results.h"

namespace nvimgcodec {
class IDecodeState;
class IImage;
class ICodeStream;

class IImageDecoder
{
  public:
    virtual ~IImageDecoder() = default;
    virtual nvimgcodecBackendKind_t getBackendKind() const = 0;
    virtual std::unique_ptr<IDecodeState> createDecodeStateBatch() const = 0;
    virtual void canDecode(const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images,
        const nvimgcodecDecodeParams_t* params, std::vector<bool>* result, std::vector<nvimgcodecProcessingStatus_t>* status) const = 0;
    virtual std::unique_ptr<ProcessingResultsFuture> decode(IDecodeState* decode_state_batch,
        const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images,
        const nvimgcodecDecodeParams_t* params) = 0;
    virtual const char* decoderId() const = 0;
};

} // namespace nvimgcodec