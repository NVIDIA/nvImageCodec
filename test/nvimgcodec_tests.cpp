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

#include "nvimgcodec_tests.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <iostream>

namespace nvimgcodec { namespace test {

std::string resources_dir;
int CC_major;

}} // namespace nvimgcodec::test

namespace {
std::string getCmdOption(char** begin, char** end, const std::string& option)
{
    char** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return {};
}
} // namespace

int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);

    nvimgcodec::test::resources_dir = getCmdOption(argv, argv + argc, "--resources_dir");
    if (nvimgcodec::test::resources_dir.empty()) {
        std::cerr << "Some tests needs a valid resources dir (e.g. --resources_dir path/to/resources)\n";
        nvimgcodec::test::resources_dir = "../../resources";
    }

    cudaDeviceProp props;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&props, dev);
    std::cout << "\nUsing GPU - " << props.name << " with CC " << props.major << "." << props.minor << std::endl;
    nvimgcodec::test::CC_major = props.major;
    int result = RUN_ALL_TESTS();
    cudaDeviceReset();
    return result;
}