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

#include "../src/thread_pool.h"
#include <gtest/gtest.h>
#include <atomic>

namespace nvimgcodec { namespace test {

TEST(ThreadPool, AddWork) {
  ThreadPool tp(16, 0, false, "ThreadPool test");
  std::atomic<int> count{0};
  auto increase = [&count](int thread_id) { count++; };
  for (int i = 0; i < 64; i++) {
    tp.addWork(increase);
  }
  ASSERT_EQ(count, 0);
  tp.run();
  tp.wait();
  ASSERT_EQ(count, 64);
}

TEST(ThreadPool, AddWorkImmediateStart) {
  ThreadPool tp(16, 0, false, "ThreadPool test");
  std::atomic<int> count{0};
  auto increase = [&count](int thread_id) { count++; };
  for (int i = 0; i < 64; i++) {
    tp.addWork(increase);
    tp.run();
  }
  tp.wait();
  ASSERT_EQ(count, 64);
}

}  // namespace test

} // namespace nvimgcodec
