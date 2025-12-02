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

#pragma once

#include <string>
#include <vector>
#include <tuple>

#include <nvimgcodec.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace nvimgcodec {

namespace py = pybind11;
using namespace py::literals;

class TileGeometry
{
  public:
    TileGeometry();
    
    TileGeometry(const TileGeometry& other);
    TileGeometry& operator=(const TileGeometry& other);

    uint32_t getNumTilesY() { return impl_.num_tiles_y; }
    void setNumTilesY(uint32_t num_tiles_y) { impl_.num_tiles_y = num_tiles_y; }

    uint32_t getNumTilesX() { return impl_.num_tiles_x; }
    void setNumTilesX(uint32_t num_tiles_x) { impl_.num_tiles_x = num_tiles_x; }

    uint32_t getTileHeight() { return impl_.tile_height; }
    void setTileHeight(uint32_t tile_height) { impl_.tile_height = tile_height; }

    uint32_t getTileWidth() { return impl_.tile_width; }
    void setTileWidth(uint32_t tile_width) { impl_.tile_width = tile_width; }

    nvimgcodecTileGeometryInfo_t impl_;
};

} // namespace nvimgcodec 