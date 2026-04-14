/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <sstream>
#include "../extensions/utils/library_version.h"

namespace nvimgcodec {
namespace test {

TEST(LibraryVersionTest, Constructors) {
    // Default constructor creates invalid version
    LibraryVersion v_invalid;
    EXPECT_FALSE(v_invalid.valid);
    EXPECT_FALSE(v_invalid);
    EXPECT_EQ(v_invalid.flat_version(), -1);
    
    // Parameterized constructor creates valid version
    LibraryVersion v_valid(12, 3, 5);
    EXPECT_TRUE(v_valid.valid);
    EXPECT_TRUE(v_valid);
    EXPECT_EQ(v_valid.major_ver, 12);
    EXPECT_EQ(v_valid.minor_ver, 3);
    EXPECT_EQ(v_valid.patch_ver, 5);
}

TEST(LibraryVersionTest, FlatVersion) {
    EXPECT_EQ(LibraryVersion(12, 3, 5).flat_version(), 12003005);
    EXPECT_EQ(LibraryVersion(1, 0, 0).flat_version(), 1000000);
    EXPECT_EQ(LibraryVersion(0, 1, 0).flat_version(), 1000);
    EXPECT_EQ(LibraryVersion(0, 0, 1).flat_version(), 1);
    EXPECT_EQ(LibraryVersion(0, 0, 0).flat_version(), 0);
    EXPECT_EQ(LibraryVersion().flat_version(), -1);  // Invalid
}

TEST(LibraryVersionTest, EqualityComparison) {
    LibraryVersion v1(12, 3, 5);
    LibraryVersion v2(12, 3, 5);
    LibraryVersion v3(12, 3, 6);
    
    EXPECT_TRUE(v1 == v2);
    EXPECT_FALSE(v1 == v3);
    EXPECT_FALSE(v1 != v2);
    EXPECT_TRUE(v1 != v3);
    
    // Invalid versions equal each other
    EXPECT_TRUE(LibraryVersion() == LibraryVersion());
}

TEST(LibraryVersionTest, OrderingComparison) {
    // Major version differences
    EXPECT_TRUE(LibraryVersion(11, 9, 0) < LibraryVersion(12, 0, 0));
    EXPECT_TRUE(LibraryVersion(12, 0, 0) < LibraryVersion(13, 0, 0));
    
    // Minor version differences
    EXPECT_TRUE(LibraryVersion(12, 1, 0) < LibraryVersion(12, 2, 0));
    EXPECT_FALSE(LibraryVersion(12, 2, 0) < LibraryVersion(12, 1, 0));
    
    // Patch version differences
    EXPECT_TRUE(LibraryVersion(12, 2, 0) < LibraryVersion(12, 2, 1));
    EXPECT_FALSE(LibraryVersion(12, 2, 1) < LibraryVersion(12, 2, 0));
    
    // Equal versions
    EXPECT_FALSE(LibraryVersion(12, 2, 0) < LibraryVersion(12, 2, 0));
    
    // Test >= operator (used in actual code)
    EXPECT_TRUE(LibraryVersion(12, 2, 0) >= LibraryVersion(12, 1, 0));
    EXPECT_TRUE(LibraryVersion(12, 2, 0) >= LibraryVersion(12, 2, 0));
    EXPECT_FALSE(LibraryVersion(12, 1, 0) >= LibraryVersion(12, 2, 0));
}

TEST(LibraryVersionTest, StreamOutput) {
    std::ostringstream oss1;
    oss1 << LibraryVersion(12, 3, 5);
    EXPECT_EQ(oss1.str(), "12.3.5");
    
    std::ostringstream oss2;
    oss2 << LibraryVersion();
    EXPECT_EQ(oss2.str(), "invalid");
}

} // namespace test
} // namespace nvimgcodec
