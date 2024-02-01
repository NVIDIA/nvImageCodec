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

#include <atomic>
#include <deque>
#include <mutex>
#include <numeric>
#include <condition_variable>
#include "processing_results.h"

namespace nvimgcodec {

/**
 * @brief State object shared between ProcessingResultsPromise and ProcessingResultsFuture
 *
 * When creating a promise, a shared state object is created and then shared when the promise
 * is copied. Finally, it's also shared when a future object is obtained from a promise.
 * This object is where the actual synchronization and notificiation is implemented.
 */
class ProcessingResultsSharedState {
 public:
  static std::shared_ptr<ProcessingResultsSharedState> get() {
    if (free_.empty()) {
      return std::shared_ptr<ProcessingResultsSharedState>(new ProcessingResultsSharedState(), deleter);
    }

    auto ret = std::shared_ptr<ProcessingResultsSharedState>(free_.back().release(), deleter);
    free_.pop_back();
    return ret;
  }

  static void deleter(ProcessingResultsSharedState *ptr) {
    free_.emplace_back(ptr);
  }

  void init(int n) {
    results_.clear();
    results_.resize(n);
    ready_indices_.clear();
    ready_indices_.reserve(n);
    ready_mask_.clear();
    ready_mask_.resize(n);
    last_checked_ = 0;
    has_future_.clear();
  }

  void reset() {
    results_.clear();
    ready_indices_.clear();
    last_checked_ = 0;
    has_future_.clear();
  }

  void wait_all() {
    if (ready_indices_.size() == results_.size())
      return;

    std::unique_lock lock(mtx_);
    cv_any_.wait(lock, [&]() {
      return ready_indices_.size() == results_.size();
    });
  }

  std::pair<int*, size_t> wait_new()
  {
    if (last_checked_ == results_.size())
      return {};

    std::unique_lock lock(mtx_);
    if (last_checked_ == results_.size())
      return {};

    cv_any_.wait(lock, [&]() {
      return ready_indices_.size() > last_checked_;
    });
    size_t last = last_checked_;
    last_checked_ = ready_indices_.size();
    return std::make_pair(&ready_indices_[last], last_checked_ - last);
  }

  void wait_one(int index) {
    if (!ready_mask_[index]) {
      std::unique_lock lock(mtx_);
      cv_any_.wait(lock, [&]() {
        return ready_mask_[index];
      });
    }
  }

  void set(int index, ProcessingResult res) {
    if (static_cast<size_t>(index) >= results_.size())
      throw std::out_of_range("Sample index out of range.");

    std::lock_guard lg(mtx_);
    if (ready_mask_[index])
      throw std::logic_error("Entry already set.");
    results_[index] = std::move(res);
    ready_indices_.push_back(index);
    ready_mask_[index] = true;
    cv_any_.notify_all();
  }

  void set_all(ProcessingResult* res, size_t size) {
    if (static_cast<size_t>(size) == results_.size()) {
      throw std::logic_error("The number of the results doesn't match one specified at "
                             "promise's construction.");
    }

    std::lock_guard lg(mtx_);
    for (int i = 0, n = size; i < n; i++) {
      if (ready_mask_[i])
        throw std::logic_error("Entry already set.");
      results_[i] = std::move(res[i]);
    }
    ready_indices_.resize(size);
    std::iota(ready_indices_.begin(), ready_indices_.end(), 0);

    cv_any_.notify_all();
  }

  std::mutex mtx_;
  std::condition_variable cv_any_;

  std::atomic_flag has_future_ = ATOMIC_FLAG_INIT;
  std::vector<ProcessingResult> results_;
  std::vector<int> ready_indices_;
  std::vector<uint8_t> ready_mask_;  // avoid vector<bool>
  size_t last_checked_ = 0;
  std::atomic_int num_promises_;

  static thread_local std::deque<std::unique_ptr<ProcessingResultsSharedState>> free_;
};

thread_local std::deque<std::unique_ptr<ProcessingResultsSharedState>>
    ProcessingResultsSharedState::free_;

std::unique_ptr<ProcessingResultsFuture> ProcessingResultsPromise::getFuture() const {
  if (impl_->has_future_.test_and_set())
    throw std::logic_error("There's already a future associated with this promise.");
  return std::unique_ptr<ProcessingResultsFuture>(
      new ProcessingResultsFuture(impl_)); //std::make_unique<ProcessingResultsFuture>(impl_); //TODO
}

ProcessingResultsFuture::ProcessingResultsFuture(std::shared_ptr<ProcessingResultsSharedState> impl)
: impl_(std::move(impl)) {}


ProcessingResultsFuture::~ProcessingResultsFuture() {
  if (impl_) {
    #pragma GCC diagnostic push
  #ifdef __clang__
    #pragma GCC diagnostic ignored "-Wexceptions"
  #else
    #pragma GCC diagnostic ignored "-Wterminate"
  #endif
    if (impl_->ready_indices_.size() != impl_->results_.size())
      throw std::logic_error("Deferred results incomplete");
    #pragma GCC diagnostic pop
    impl_.reset();
  }
}

void ProcessingResultsFuture::waitForAll() const {
  impl_->wait_all();
}

std::pair<int*, size_t> ProcessingResultsFuture::waitForNew() const
{
  return impl_->wait_new();
}

void ProcessingResultsFuture::waitForOne(int index) const {
  return impl_->wait_one(index);
}


int ProcessingResultsFuture::getNumSamples() const {
  return impl_->results_.size();
}

std::pair<ProcessingResult*, size_t> ProcessingResultsFuture::getAllRef() const
{
  waitForAll();
  return std::make_pair(impl_->results_.data(), impl_->results_.size());
}

std::vector<ProcessingResult> ProcessingResultsFuture::getAllCopy() const {
  waitForAll();
  return impl_->results_;
}

ProcessingResult ProcessingResultsFuture::getOne(int index) const {
  waitForOne(index);
  return impl_->results_[index];
}

void ProcessingResultsPromise::set(int index, ProcessingResult res) {
  impl_->set(index, std::move(res));
}

void ProcessingResultsPromise::setAll(ProcessingResult* res, size_t size) {
  impl_->set_all(res, size);
}

void ProcessingResultsPromise::setAll(ProcessingResult res)
{
  for (size_t i = 0; i < impl_->results_.size(); ++i) {
    impl_->set(i, res);
  }
}

int ProcessingResultsPromise::getNumSamples() const {
  return impl_->results_.size();
}

ProcessingResultsPromise::ProcessingResultsPromise(int num_samples) {
  impl_ = ProcessingResultsSharedState::get();
  impl_->num_promises_ = 1;
  impl_->init(num_samples);
}

ProcessingResultsPromise &ProcessingResultsPromise::operator=(const ProcessingResultsPromise &other) {
  impl_ = other.impl_;
  if (impl_)
    impl_->num_promises_++;
  return *this;
}

ProcessingResultsPromise::~ProcessingResultsPromise() {
  auto impl = std::move(impl_);
  if (impl) {
    if (--impl->num_promises_ == 0 && impl->ready_indices_.size() != impl->results_.size()) {
    #pragma GCC diagnostic push
  #ifdef __clang__
    #pragma GCC diagnostic ignored "-Wexceptions"
  #else
    #pragma GCC diagnostic ignored "-Wterminate"
  #endif
      std::logic_error("Last promise is dead and the result is incomplete.");
    #pragma GCC diagnostic pop
    }
  }
}

} // namespace nvimgcodec
