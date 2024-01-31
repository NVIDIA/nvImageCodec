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
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "iencode_state.h"
#include "iimage_encoder.h"
#include "iwork_manager.h"
#include "work.h"

namespace nvimgcodec {

class ICodec;
class ILogger;

/**
 * @brief A worker that processes sub-batches of work to be processed by a particular encoder.
 *
 * A Worker waits for incoming Work objects and processes them by running
 * `encoder_->ScheduleEncode` and waiting for partial results, scheduling the failed
 * samples to a fallback encoder, if present.
 *
 * When a sample is successfully encoded, it is marked as a success in the parent
 * EncodeResultsPromise. If it fails, it goes to fallback and only if all fallbacks fail, it is
 * marked in the EncodeResultsPromise as a failure.
 */
class EncoderWorker
{
  public:
    /**
   * @brief Constructs a encoder worker for a given encoder.
   *
   * @param work_manager   - creates and recycles work
   * @param codec   - the factory that constructs the encoder for this worker
   */
    EncoderWorker(ILogger* logger, IWorkManager<nvimgcodecEncodeParams_t>* work_manager, const nvimgcodecExecutionParams_t* exec_params,
        const std::string& options, const ICodec* codec, int index);
    ~EncoderWorker();

    void start();
    void stop();
    void addWork(std::unique_ptr<Work<nvimgcodecEncodeParams_t>> work);

    EncoderWorker* getFallback();
    IImageEncoder* getEncoder();

  private:
    /**
   * @brief Processes a (sub)batch of work.
   *
   * The work is scheduled and the results are waited for. Any failed samples will be added
   * to a fallback work, if a fallback encoder is present.
   */
    void processBatch(std::unique_ptr<Work<nvimgcodecEncodeParams_t>> work) noexcept;

    /**
   * @brief The main loop of the worker thread.
   */
    void run();

    ILogger* logger_;
    IWorkManager<nvimgcodecEncodeParams_t>* work_manager_ = nullptr;
    const ICodec* codec_ = nullptr;
    int index_ = 0;
    const nvimgcodecExecutionParams_t* exec_params_;
    const std::string& options_;

    std::mutex mtx_;
    std::condition_variable cv_;

    std::unique_ptr<Work<nvimgcodecEncodeParams_t>> work_;
    std::thread worker_;
    bool stop_requested_ = false;
    std::once_flag started_;

    std::unique_ptr<IImageEncoder> encoder_;
    bool is_input_expected_in_device_ = false;
    std::unique_ptr<IEncodeState> encode_state_batch_;
    std::unique_ptr<EncoderWorker> fallback_ = nullptr;
};


} // namespace nvimgcodec
