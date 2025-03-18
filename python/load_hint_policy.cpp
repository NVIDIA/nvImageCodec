/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "load_hint_policy.h"

namespace nvimgcodec {

void LoadHintPolicy::exportToPython(py::module& m)
{
    // clang-format off
    py::enum_<nvimgcodecLoadHintPolicy_t>(m, "LoadHintPolicy", 
            R"pbdoc(
            Enum representing load hint policies for backend batch processing.

            Load hint is used to calculate the fraction of the batch items that will be picked by
            this backend and the rest of the batch items would be passed to fallback codec.
            This is just a hint and a particular implementation can choose to ignore it.
            )pbdoc")
        .value("IGNORE", NVIMGCODEC_LOAD_HINT_POLICY_IGNORE,
            R"pbdoc(
            Ignore the load hint.

            In this policy, the backend does not take the load hint into account when 
            determining batch processing. It functions as if no hint was provided.
            )pbdoc")
        .value("FIXED", NVIMGCODEC_LOAD_HINT_POLICY_FIXED,
            R"pbdoc(
            Use the load hint to determine a fixed batch size.

            This policy calculates the backend batch size based on the provided load hint 
            once, and uses this fixed batch size for processing.  
            )pbdoc")
        .value("ADAPTIVE_MINIMIZE_IDLE_TIME", NVIMGCODEC_LOAD_HINT_POLICY_ADAPTIVE_MINIMIZE_IDLE_TIME,
            R"pbdoc(
            Adaptively use the load hint to minimize idle time.

            This policy uses the load hint as an initial starting point and recalculates 
            on each iteration to dynamically adjust and reduce the idle time of threads, 
            optimizing overall resource utilization.
            )pbdoc")
        .export_values();
    // clang-format on
};

} // namespace nvimgcodec
