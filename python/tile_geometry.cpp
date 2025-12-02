/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tile_geometry.h"

#include <iostream>

#include "error_handling.h"

namespace nvimgcodec {

TileGeometry::TileGeometry()
    : impl_{NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO, sizeof(nvimgcodecTileGeometryInfo_t), nullptr, 0, 0, 0, 0}
{
}

TileGeometry::TileGeometry(const TileGeometry& other)
    : impl_(other.impl_)
{
    // Set struct_next to nullptr as it should be managed separately
    impl_.struct_next = nullptr;
}

TileGeometry& TileGeometry::operator=(const TileGeometry& other)
{
    if (this != &other) {
        impl_ = other.impl_;
        // Set struct_next to nullptr as it should be managed separately
        impl_.struct_next = nullptr;
    }
    return *this;
}


} // namespace nvimgcodec 