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
#include <stdexcept>
#include <utility>
#include <vector>

namespace nvimgcodec {

/**
 * @brief Results of a processing operation.
 */
struct ProcessingResult
{
    nvimgcodecProcessingStatus_t status_ = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
    std::exception_ptr exception_ = nullptr;

    bool isSuccess() const {
      return status_ == NVIMGCODEC_PROCESSING_STATUS_SUCCESS && exception_ == nullptr;
    }

    static ProcessingResult success() { return {NVIMGCODEC_PROCESSING_STATUS_SUCCESS, {}}; }
    static ProcessingResult failure(nvimgcodecProcessingStatus_t status) { return {status, {}}; }
    static ProcessingResult failure(std::exception_ptr exception) { return {NVIMGCODEC_PROCESSING_STATUS_FAIL, std::move(exception)}; }
};

class ProcessingResultsSharedState;
class ProcessingResultsFuture;

/**
 * @brief A promise object for processing results.
 *
 * When asynchronous decoding is performed, a promise object and copied among the workers.
 * At exit, a future object is obtained from it by a call to get_future.
 * The promise object is what the workers use to notify the caller about the results.
 * The future object is what the caller uses to wait for and access the results.
 */
class ProcessingResultsPromise
{
  public:
    explicit ProcessingResultsPromise(int num_samples);
    ~ProcessingResultsPromise();

    ProcessingResultsPromise(const ProcessingResultsPromise& other) { *this = other; }
    ProcessingResultsPromise(ProcessingResultsPromise&&) = default;
    ProcessingResultsPromise& operator=(const ProcessingResultsPromise&);
    ProcessingResultsPromise& operator=(ProcessingResultsPromise&&) = default;

    /**
   * @brief Obtains a future object for the caller/consume
   */
    std::unique_ptr<ProcessingResultsFuture> getFuture() const;

    /**
   * @brief The number of samples in this promise
   */
    int getNumSamples() const;

    /**
   * @brief Sets the result for a specific sample
   */
    void set(int index, ProcessingResult res);

    /**
   * @brief Sets all results at once
   */
    void setAll(ProcessingResult* res, size_t size);

    /** 
    * @brief Sets all results at once with the same result
    */
    void setAll(ProcessingResult res);

    /**
   * @brief Checks if two promises point to the same shared state.
   */
    bool operator==(const ProcessingResultsPromise& other) const { return impl_ == other.impl_; }

    /**
   * @brief Checks if two promises point to different shared states.
   */
    bool operator!=(const ProcessingResultsPromise& other) const { return !(*this == other); }

  private:
    std::shared_ptr<ProcessingResultsSharedState> impl_ = nullptr;
};

/**
 * @brief The object returned by asynchronous decoding requests
 *
 * The future object allows the caller of asynchronous decoding APIs to wait for and obtain
 * partial results, so it can react incrementally to the decoding of mulitple samples,
 * perfomed in the background.
 */
class ProcessingResultsFuture
{
  public:
    ProcessingResultsFuture(ProcessingResultsFuture&& other) = default;
    ProcessingResultsFuture(const ProcessingResultsFuture& other) = delete;

    /**
   * @brief Destroys the future object and terminates the program if the results have
   *        not been consumed
   */
    ~ProcessingResultsFuture();

    ProcessingResultsFuture& operator=(const ProcessingResultsFuture&) = delete;
    ProcessingResultsFuture& operator=(ProcessingResultsFuture&& other)
    {
        std::swap(impl_, other.impl_);
        return *this;
    }

    /**
   * @brief Waits for all results to be ready
   */
    void waitForAll() const;

    /**
   * @brief Waits for any results that have appeared since the previous call to wait_new
   *        (or any results, if this is the first call).
   *
   * @return The indices of results that are ready. They can be read with `get_one` without waiting.
   */
    std::pair<int*, size_t> waitForNew() const;

    /**
   * @brief Waits for the result of a  particualr sample
   */
    void waitForOne(int index) const;

    /**
   * @brief The total number of exepcted results.
   */
    int getNumSamples() const;

    /**
   * @brief Waits for all results to become available and returns a span containing the results.
   *
   * @remarks The return value MUST NOT outlive the future object.
   */
    std::pair<ProcessingResult*, size_t> getAllRef() const;

    /**
   * @brief Waits for all results to become available and returns a vector containing the results.
   *
   * @remarks The return value is a copy and can outlive the future object.
   */
    std::vector<ProcessingResult> getAllCopy() const;

    /**
   * @brief Waits for all results to become available and returns a span containing the results.
   *
   * @remarks The return value MUST NOT outlive the future object.
   */
    std::pair<ProcessingResult*, size_t> getAll() const& { return getAllRef(); }

    /**
   * @brief Waits for all results to become available and returns a vector containing the results.
   *
   * @remarks The return value is a copy and can outlive the future object.
   */
    std::vector<ProcessingResult> getAll() && { return getAllCopy(); }

    /**
   * @brief Waits for a result and returns it.
   */
    ProcessingResult getOne(int index) const;

  private:
    explicit ProcessingResultsFuture(std::shared_ptr<ProcessingResultsSharedState> impl);
    friend class ProcessingResultsPromise;
    // friend std::unique_ptr<ProcessingResultsFuture> std::make_unique<ProcessingResultsFuture>(
    //     std::shared_ptr<nvimgcodec::ProcessingResultsSharedState>&);
    // friend std::unique_ptr<ProcessingResultsFuture> std::make_unique<ProcessingResultsFuture>(
    //     std::shared_ptr<nvimgcodec::ProcessingResultsSharedState>&&);
    std::shared_ptr<ProcessingResultsSharedState> impl_ = nullptr;
};

} // namespace nvimgcodec
