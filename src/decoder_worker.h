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
#include <string>
#include <thread>
#include <vector>
#include <mutex>

#include "work.h"
#include "iimage_decoder.h"
#include "idecode_state.h"
#include "iwork_manager.h"

namespace nvimgcodec {

class ICodec;
class ILogger;

/**
 * @brief A worker that processes sub-batches of work to be processed by a particular decoder.
 *
 * A Worker waits for incoming Work objects and processes them by running
 * `decoder_->ScheduleDecode` and waiting for partial results, scheduling the failed
 * samples to a fallback decoder, if present.
 *
 * When a sample is successfully decoded, it is marked as a success in the parent
 * DecodeResultsPromise. If it fails, it goes to fallback and only if all fallbacks fail, it is
 * marked in the DecodeResultsPromise as a failure.
 */
class DecoderWorker
{
  public:
    /**
   * @brief Constructs a decoder worker for a given decoder.
   *
   * @param work_manager   - creates and recycles work
   * @param codec   - the factory that constructs the decoder for this worker
   */
    DecoderWorker(ILogger* logger, IWorkManager<nvimgcodecDecodeParams_t>* work_manager, const nvimgcodecExecutionParams_t* exec_params,
        const std::string& options, const ICodec* codec, int index);
    ~DecoderWorker();

    void addWork(std::unique_ptr<Work<nvimgcodecDecodeParams_t>> work, bool immediate);

    DecoderWorker* getFallback();
    IImageDecoder* getDecoder();

  private:
    void start();
    void stop();

    /**
   * @brief Processes a (sub)batch of work.
   *
   * The work is scheduled and the results are waited for. Any failed samples will be added
   * to a fallback work, if a fallback decoder is present.
   * 
   * @param work work to execute
   * @param immediate If true, work is not scheduled to a worker thread but executed in the current
   *                  thread instead.
   */
    void processBatch(std::unique_ptr<Work<nvimgcodecDecodeParams_t>> work, bool immediate) noexcept;

  /**
   * @brief Waits for and process current work results
   * 
   * @param curr_work 
   * @param curr_results 
   * @param immediate 
   */
  void processCurrentResults(
    std::unique_ptr<Work<nvimgcodecDecodeParams_t>> curr_work, std::unique_ptr<ProcessingResultsFuture> curr_results, bool immediate);

  /**
   * @brief Set current work future results for processing in the working thread
   * 
   * @param work 
   * @param future 
   */
  void updateCurrentWork(std::unique_ptr<Work<nvimgcodecDecodeParams_t>> work, std::unique_ptr<ProcessingResultsFuture> future);

    /**
   * @brief The main loop of the worker thread.
   */
    void run();

    ILogger* logger_;
    IWorkManager<nvimgcodecDecodeParams_t>* work_manager_ = nullptr;
    const ICodec* codec_ = nullptr;
    int index_ = 0;
    const nvimgcodecExecutionParams_t* exec_params_;
    const std::string& options_;

    std::mutex mtx_;
    std::condition_variable cv_;

    std::unique_ptr<Work<nvimgcodecDecodeParams_t>> work_;  // next iteration
    std::unique_ptr<Work<nvimgcodecDecodeParams_t>> curr_work_;  // current (already scheduled iteration)
    std::unique_ptr<ProcessingResultsFuture> curr_results_;  // future results from current iteration
    std::thread worker_;
    bool stop_requested_ = false;
    std::once_flag started_;

    std::unique_ptr<IImageDecoder> decoder_;
    bool is_device_output_ = false;
    std::unique_ptr<IDecodeState> decode_state_batch_;
    std::unique_ptr<DecoderWorker> fallback_ = nullptr;
};


}
