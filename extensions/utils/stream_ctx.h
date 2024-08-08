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

#include <cassert>
#include <map>
#include <memory>
#include <nvtx3/nvtx3.hpp>
#include <vector>
#include <future>
#include "nvimgcodec.h"
#include <iostream>

struct CodeStreamCtx;

/**
 * @brief Context for a single sample in a batch
 */
struct BatchItemCtx {
    CodeStreamCtx* code_stream_ctx;
    int index;
    nvimgcodecImageDesc_t* image;
    nvimgcodecProcessingStatus_t processing_status;
    const nvimgcodecDecodeParams_t* params;
};

/** 
 * @brief Context for a single encoded stream
 */
struct CodeStreamCtx {
    // Code stream pointer;
    nvimgcodecCodeStreamDesc_t* code_stream_;
    // Unique stream id
    uint64_t code_stream_id_;

    // Samples in the batch that are decoded from this code stream
    std::vector<BatchItemCtx*> batch_items_;

    // Pointer and size of encoded stream (could point to buffer if the io stream can't be mapped)
    void* encoded_stream_data_ = nullptr;
    size_t encoded_stream_data_size_ = 0;

    // Local copy of the stream, in case map is not supported
    std::vector<unsigned char> buffer_;

    size_t size() const {
        return batch_items_.size();
    }
    
    void reset() {
        setCodeStream(nullptr);
    }

    void clearBatchItems() {
        batch_items_.clear();
    }

    void setCodeStream(nvimgcodecCodeStreamDesc_t* code_stream) {
        bool same_stream = code_stream == code_stream_ && code_stream->id == code_stream_->id;
        if (!same_stream) {
            clearBatchItems();
            code_stream_ = code_stream;
            code_stream_id_ = code_stream ? code_stream->id : 0;
            encoded_stream_data_ = nullptr;
            encoded_stream_data_size_ = 0;
        
            buffer_.clear();
        }
    }

    bool load() {
        if (encoded_stream_data_ == nullptr) {
            nvimgcodecIoStreamDesc_t* io_stream = code_stream_->io_stream;
            io_stream->size(io_stream->instance, &encoded_stream_data_size_);
            void* mapped_encoded_stream_data = nullptr;
            buffer_.clear();
            io_stream->map(io_stream->instance, &mapped_encoded_stream_data, 0, encoded_stream_data_size_);
            if (!mapped_encoded_stream_data) {
                nvtx3::scoped_range marker{"buffer read"};
                buffer_.resize(encoded_stream_data_size_);
                io_stream->seek(io_stream->instance, 0, SEEK_SET);
                size_t read_nbytes = 0;
                io_stream->read(io_stream->instance, &read_nbytes, &buffer_[0], encoded_stream_data_size_);
                if (read_nbytes != encoded_stream_data_size_)
                    return false;
                encoded_stream_data_ = &buffer_[0];
            } else {
                encoded_stream_data_ = mapped_encoded_stream_data;
            }
        }
        return true;
    }

    void addBatchItem(BatchItemCtx* batch_item)
    {
        batch_items_.push_back(batch_item);
    }
};


struct CodeStreamCtxManager {
    using CodeStreamCtxPtr = std::shared_ptr<CodeStreamCtx>;

    CodeStreamCtxPtr acquireCtx() {
        if (free_ctx_.empty())
            return std::make_shared<CodeStreamCtx>();

        auto ret = std::move(free_ctx_.back());
        free_ctx_.pop_back();
        return ret;
    }

    void releaseCtx(CodeStreamCtxPtr&& ctx) {
        assert(ctx);
        ctx->reset();
        free_ctx_.push_back(std::move(ctx));
    }

    void feedSamples(nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images,
        int batch_size, const nvimgcodecDecodeParams_t* params)
    {
        batch_.clear();
        batch_.resize(batch_size);

        std::map<uint64_t, CodeStreamCtxPtr> new_stream_ctx;
        for (int index = 0; index < batch_size; index++) {
            auto *cs = code_streams[index];
            auto &ctx = new_stream_ctx[cs->id];
            if (!ctx) {  // if not seen in this iteration
                auto it = stream_ctx_.find(cs->id);  // look for it in the last iteration ctx
                if (it != stream_ctx_.end()) {
                    ctx = std::move(it->second);
                } else {
                    ctx = acquireCtx();
                }
                ctx->clearBatchItems();
            }

            // Set stream if needed
            ctx->setCodeStream(cs);

            // Append current sample to the stream context
            auto &batch_item = batch_[index];
            batch_item.code_stream_ctx = ctx.get();
            batch_item.index = index;
            batch_item.image = images[index];
            batch_item.processing_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
            batch_item.params = params;
            ctx->addBatchItem(&batch_item);
        }

        // we used the ones we were interested in, we can release the rest to the pool
        for (auto& [id, ctx] : stream_ctx_) {
            if (ctx)
                releaseCtx(std::move(ctx));
        }

        // only remember stream ctxs for one iteration back
        std::swap(stream_ctx_, new_stream_ctx);

        stream_ctx_view_.clear();
        stream_ctx_view_.reserve(stream_ctx_.size());
        for (auto& [id, ctx] : stream_ctx_)
            stream_ctx_view_.push_back(ctx);
    }

    size_t size() const {
        return stream_ctx_view_.size();
    }

    CodeStreamCtxPtr& operator[](size_t index) {
        return stream_ctx_view_[index];
    }

    CodeStreamCtxPtr& get_by_stream_id(uint64_t stream_id) {
        return stream_ctx_[stream_id];
    }

    BatchItemCtx& get_batch_item(int index) {
        return batch_[index];
    }

  private:
    std::map<uint64_t, CodeStreamCtxPtr> stream_ctx_;
    std::vector<CodeStreamCtxPtr> free_ctx_;
    std::vector<CodeStreamCtxPtr> stream_ctx_view_;
    std::vector<BatchItemCtx> batch_;
};
