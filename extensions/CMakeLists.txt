# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if (BUILD_NVJPEG2K_EXT)
    add_subdirectory(nvjpeg2k)
endif()

if(BUILD_NVJPEG_EXT)
    add_subdirectory(nvjpeg)
endif()

if(BUILD_NVBMP_EXT)
    add_subdirectory(nvbmp)
endif()

if(BUILD_NVPNM_EXT)
    add_subdirectory(nvpnm)
endif()

if(BUILD_NVTIFF_EXT)
    add_subdirectory(nvtiff)
endif ()

if(BUILD_LIBJPEG_TURBO_EXT)
    add_subdirectory(libjpeg_turbo)
endif ()

if(BUILD_LIBTIFF_EXT)
    add_subdirectory(libtiff)
endif ()

if(BUILD_OPENCV_EXT)
    add_subdirectory(opencv)
endif ()
