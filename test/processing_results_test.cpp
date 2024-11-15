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

#include <gtest/gtest.h>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <thread>
#include "../src/processing_results.h"
#include "../src/thread_pool.h"

namespace nvimgcodec { namespace test {

TEST(FutureProcessingResultsTest, WaitNew)
{
    ThreadPool tp(4, CPU_ONLY_DEVICE_ID, false, "FutureProcessingResultsTest");

    ProcessingResultsPromise pro(3);
    auto fut = pro.getFuture();
    tp.addWork(
        [&pro](int tidx) mutable {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            pro.set(1, ProcessingResult::success());
            pro.set(0, ProcessingResult::success());
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            pro.set(2, ProcessingResult::success());
        },
        0, true);
    std::pair<int*, size_t> res1 = fut->waitForNew();
    std::pair<int*, size_t> res2 = fut->waitForNew();
    std::pair<int*, size_t> res3 = fut->waitForNew();
    EXPECT_EQ(res1.second + res2.second + res3.second, 3);
    // We can either get all results in one spans, in two spans or in three spans, depending
    // on timing - in any case, the end of the last non-empty span should point to the start of the
    // first span, offset by the number of entries (which is 3).
    ASSERT_TRUE(res1.first + 3 == (res1.first + res1.second) ||
                res1.first + 3 == (res2.first + res2.second) ||
                res1.first + 3 == (res3.first + res3.second));
    // now we know we can access all spans through the first one
    EXPECT_EQ(res1.first[0], 1);
    EXPECT_EQ(res1.first[1], 0);
    EXPECT_EQ(res1.first[2], 2);
}

TEST(FutureProcessingResultsTest, Benchmark)
{
    ThreadPool tp(4, CPU_ONLY_DEVICE_ID, false, "FutureProcessingResultsTest");

    int num_iter = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < num_iter; iter++) {
        ProcessingResultsPromise res(100);
        tp.addWork(
            [&](int tidx) {
                for (int i = 0; i < res.getNumSamples(); i++)
                    res.set(i, ProcessingResult::success());
            },
            0, true);
        auto future = res.getFuture();
        future->waitForAll();
        for (int i = 0; i < res.getNumSamples(); i++)
            future->getOne(i);
        tp.wait();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end - start);
    std::cout << time.count() / num_iter << " us/iter" << std::endl;
}

}} // namespace nvimgcodec::test
