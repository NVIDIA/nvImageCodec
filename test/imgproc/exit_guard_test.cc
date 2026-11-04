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
#include <cstdlib>
#include "imgproc/exit_guard.h"

namespace nvimgcodec {

TEST(ProcessExitFlag, DefaultsToUnset)
{
    ProcessExitFlag flag;
    EXPECT_FALSE(flag.IsSet());
}

TEST(ProcessExitFlag, TripSetsPermanently)
{
    ProcessExitFlag flag;
    flag.Trip();
    EXPECT_TRUE(flag.IsSet());
    // Tripping again is idempotent.
    flag.Trip();
    EXPECT_TRUE(flag.IsSet());
}

namespace {

// Subprocess body: registers an atexit callback (before calling
// IsProcessExiting() the first time, so this callback is invoked *after*
// IsProcessExiting()'s own atexit-registered flag-setter -- atexit callbacks
// run in reverse registration order) that checks IsProcessExiting() and
// std::exit()s with a code reflecting what it observed. This exercises the
// exact mechanism every guarded decoder/encoder destructor relies on: the
// flag must read false during normal operation and true once std::atexit has
// actually fired.
//
// This relies on nothing having armed the guard before this function runs (an
// earlier IsProcessExiting() call would register the flag-setter first, making
// it run *last* and this callback observe `false`). The death test runs in a
// freshly exec'd child (`death_test_style = "threadsafe"`) that executes only
// this test, and no static initializer in the test binary calls
// IsProcessExiting(), so the assumption holds. Exit code 1 below means that
// invariant was broken; codes 2/3 flag the other failure modes.
void CheckFlagAtRealAtexit()
{
    if (std::atexit([]() { std::_Exit(IsProcessExiting() ? 0 : 1); }) != 0) {
        std::_Exit(2);
    }

    if (IsProcessExiting()) {
        // Should not be set yet -- this call itself is what arms the guard.
        std::_Exit(3);
    }

    std::exit(0); // runs atexit handlers, including the one registered above
}

} // namespace

TEST(ExitGuard, IsProcessExitingBecomesTrueAtRealAtexit)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(CheckFlagAtRealAtexit(), testing::ExitedWithCode(0), "");
}

} // namespace nvimgcodec
