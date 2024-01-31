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
#include "image_decoder.h"
#include <algorithm>
#include <cassert>
#include "decode_state_batch.h"
#include "exception.h"
#include "icode_stream.h"
#include "iimage.h"
#include "processing_results.h"
#include <nvtx3/nvtx3.hpp>

namespace nvimgcodec {

ImageDecoder::ImageDecoder(const nvimgcodecDecoderDesc_t* desc, const nvimgcodecExecutionParams_t* exec_params, const char* options)
    : decoder_desc_(desc)
{
    auto ret = decoder_desc_->create(decoder_desc_->instance, &decoder_, exec_params, options);
    if (NVIMGCODEC_STATUS_SUCCESS != ret) {
        decoder_ = nullptr;
    }
}

ImageDecoder::~ImageDecoder()
{
    if (decoder_)
        decoder_desc_->destroy(decoder_);
}

nvimgcodecBackendKind_t ImageDecoder::getBackendKind() const
{
    return decoder_desc_->backend_kind;
}


std::unique_ptr<IDecodeState> ImageDecoder::createDecodeStateBatch() const
{
    return std::make_unique<DecodeStateBatch>();
}

void ImageDecoder::canDecode(const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images,
    const nvimgcodecDecodeParams_t* params, std::vector<bool>* result, std::vector<nvimgcodecProcessingStatus_t>* status) const
{
    nvtx3::scoped_range marker{"ImageDecoder::canDecode"};
    assert(result->size() == code_streams.size());
    assert(status->size() == code_streams.size());

    // in case the decoder couldn't be created for some reason
    if (!decoder_) {
        for (size_t i = 0; i < code_streams.size(); ++i) {
           (*result)[i] = false;
        }
        return;
    }

    std::vector<nvimgcodecCodeStreamDesc_t*> cs_descs(code_streams.size());
    std::vector<nvimgcodecImageDesc_t*> im_descs(code_streams.size());
    for (size_t i = 0; i < code_streams.size(); ++i) {
        cs_descs[i] = code_streams[i]->getCodeStreamDesc();
        im_descs[i] = images[i]->getImageDesc();
    }
    decoder_desc_->canDecode(decoder_, &(*status)[0], &cs_descs[0], &im_descs[0], code_streams.size(), params);
    for (size_t i = 0; i < code_streams.size(); ++i) {
        (*result)[i] = (*status)[i] == NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
    }
}

std::unique_ptr<ProcessingResultsFuture> ImageDecoder::decode(IDecodeState* decode_state_batch,
    const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images, const nvimgcodecDecodeParams_t* params)
{
    nvtx3::scoped_range marker{"ImageDecoder::decode"};
    assert(code_streams.size() == images.size());

    int N = images.size();
    assert(static_cast<int>(code_streams.size()) == N);

    ProcessingResultsPromise results(N);
    auto future = results.getFuture();
    decode_state_batch->setPromise(std::move(results));
    std::vector<nvimgcodecCodeStreamDesc_t*> code_stream_descs(code_streams.size());
    std::vector<nvimgcodecImageDesc_t*> image_descs(code_streams.size());

    for (size_t i = 0; i < code_streams.size(); ++i) {
        code_stream_descs[i] = code_streams[i]->getCodeStreamDesc();
        image_descs[i] = images[i]->getImageDesc();
        images[i]->setIndex(i);
        images[i]->setPromise(decode_state_batch->getPromise());
    }

    decoder_desc_->decode(
        decoder_, code_stream_descs.data(), image_descs.data(), code_streams.size(), params);

    return future;
}

const char* ImageDecoder::decoderId() const {
    return decoder_desc_->id;
}

} // namespace nvimgcodec
