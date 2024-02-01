/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "jpeg2k_prog_order.h"

namespace nvimgcodec {

void Jpeg2kProgOrder::exportToPython(py::module& m)
{
    // clang-format off
    py::enum_<nvimgcodecJpeg2kProgOrder_t>(m, "Jpeg2kProgOrder")
        .value("LRCP", NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP)
        .value("RLCP", NVIMGCODEC_JPEG2K_PROG_ORDER_RLCP)
        .value("RPCL", NVIMGCODEC_JPEG2K_PROG_ORDER_RPCL)
        .value("PCRL", NVIMGCODEC_JPEG2K_PROG_ORDER_PCRL)
        .value("CPRL", NVIMGCODEC_JPEG2K_PROG_ORDER_CPRL)
        .export_values();
    // clang-format on
} ;


} // namespace nvimgcodec
