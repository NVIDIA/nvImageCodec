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
#include <nvimgcodec.h>
#include <cstdint>
#include <sstream>
#include <string>

namespace nvimgcodec { namespace test {

// The packed values of every 0.x release must stay as they were published, so
// that comparisons and extension API checks against an already released library
// keep their meaning after the encoding was widened to two digits of minor.
TEST(VersionTest, packed_values_of_released_versions_are_unchanged)
{
    EXPECT_EQ(600u, NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 6, 0));
    EXPECT_EQ(700u, NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 7, 0));
    EXPECT_EQ(800u, NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 8, 0));
    EXPECT_EQ(900u, NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 9, 0));
    EXPECT_EQ(901u, NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 9, 1));
}

TEST(VersionTest, two_digit_minor_does_not_collide_with_next_major)
{
    EXPECT_EQ(1000u, NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 10, 0));
    EXPECT_EQ(9999u, NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 99, 99));
    EXPECT_EQ(10000u, NVIMGCODEC_MAKE_SEMANTIC_VERSION(1, 0, 0));
    EXPECT_NE(NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 10, 0), NVIMGCODEC_MAKE_SEMANTIC_VERSION(1, 0, 0));
}

TEST(VersionTest, packed_values_keep_semantic_order)
{
    EXPECT_LT(NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 9, 0), NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 9, 1));
    EXPECT_LT(NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 9, 99), NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 10, 0));
    EXPECT_LT(NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 10, 0), NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 99, 99));
    EXPECT_LT(NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 99, 99), NVIMGCODEC_MAKE_SEMANTIC_VERSION(1, 0, 0));
    EXPECT_LT(NVIMGCODEC_MAKE_SEMANTIC_VERSION(1, 0, 0), NVIMGCODEC_MAKE_SEMANTIC_VERSION(1, 0, 1));
    EXPECT_LT(NVIMGCODEC_MAKE_SEMANTIC_VERSION(1, 0, 1), NVIMGCODEC_MAKE_SEMANTIC_VERSION(2, 3, 4));
}

TEST(VersionTest, components_survive_a_pack_extract_round_trip)
{
    const uint32_t versions[][3] = {{0, 6, 0}, {0, 9, 0}, {0, 9, 1}, {0, 10, 0}, {0, 99, 99}, {1, 0, 0}, {2, 3, 4}, {13, 45, 67}};
    for (const auto& v : versions) {
        const uint32_t packed = NVIMGCODEC_MAKE_SEMANTIC_VERSION(v[0], v[1], v[2]);
        EXPECT_EQ(v[0], NVIMGCODEC_MAJOR_FROM_SEMVER(packed));
        EXPECT_EQ(v[1], NVIMGCODEC_MINOR_FROM_SEMVER(packed));
        EXPECT_EQ(v[2], NVIMGCODEC_PATCH_FROM_SEMVER(packed));
    }
}

TEST(VersionTest, ver_matches_its_components)
{
    EXPECT_EQ(NVIMGCODEC_VER, NVIMGCODEC_MAKE_SEMANTIC_VERSION(NVIMGCODEC_VER_MAJOR, NVIMGCODEC_VER_MINOR, NVIMGCODEC_VER_PATCH));
    EXPECT_EQ(static_cast<uint32_t>(NVIMGCODEC_VER_MAJOR), NVIMGCODEC_MAJOR_FROM_SEMVER(NVIMGCODEC_VER));
    EXPECT_EQ(static_cast<uint32_t>(NVIMGCODEC_VER_MINOR), NVIMGCODEC_MINOR_FROM_SEMVER(NVIMGCODEC_VER));
    EXPECT_EQ(static_cast<uint32_t>(NVIMGCODEC_VER_PATCH), NVIMGCODEC_PATCH_FROM_SEMVER(NVIMGCODEC_VER));
}

TEST(VersionTest, get_properties_reports_the_header_version)
{
    nvimgcodecProperties_t properties{NVIMGCODEC_STRUCTURE_TYPE_PROPERTIES, sizeof(nvimgcodecProperties_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecGetProperties(&properties));

    EXPECT_EQ(static_cast<uint32_t>(NVIMGCODEC_VER), properties.version);
    EXPECT_EQ(static_cast<uint32_t>(NVIMGCODEC_VER_MAJOR), NVIMGCODEC_MAJOR_FROM_SEMVER(properties.version));
    EXPECT_EQ(static_cast<uint32_t>(NVIMGCODEC_VER_MINOR), NVIMGCODEC_MINOR_FROM_SEMVER(properties.version));
    EXPECT_EQ(static_cast<uint32_t>(NVIMGCODEC_VER_PATCH), NVIMGCODEC_PATCH_FROM_SEMVER(properties.version));

    std::stringstream expected;
    expected << NVIMGCODEC_VER_MAJOR << "." << NVIMGCODEC_VER_MINOR << "." << NVIMGCODEC_VER_PATCH;
    std::stringstream reported;
    reported << NVIMGCODEC_STREAM_VER(properties.version);
    EXPECT_EQ(expected.str(), reported.str());
}

TEST(VersionTest, stream_ver_renders_a_two_digit_minor)
{
    std::stringstream ss;
    ss << NVIMGCODEC_STREAM_VER(NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 10, 0));
    EXPECT_EQ(std::string("0.10.0"), ss.str());

    ss.str("");
    ss << NVIMGCODEC_STREAM_VER(NVIMGCODEC_MAKE_SEMANTIC_VERSION(1, 0, 0));
    EXPECT_EQ(std::string("1.0.0"), ss.str());
}

}} // namespace nvimgcodec::test
