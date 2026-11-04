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

set(NVIMGCODEC_EXAMPLE_NAME nvimtrans)  

set(NVIMGCODEC_EXAMPLE_SRC
      main.cpp
      command_line_params.h
)

add_executable(${NVIMGCODEC_EXAMPLE_NAME} ${NVIMGCODEC_EXAMPLE_SRC})

# Link the shared nvImageCodec library by default; opt in to the static archive
# (shipped in a separate package) with -DNVIMGCODEC_LINK_STATIC=ON.
option(NVIMGCODEC_LINK_STATIC "Link the static nvImageCodec library instead of the shared one" OFF)
if(NVIMGCODEC_LINK_STATIC)
    set(NVIMGCODEC_EXAMPLE_LINK_TARGET nvimgcodec::nvimgcodec_static)
else()
    set(NVIMGCODEC_EXAMPLE_LINK_TARGET nvimgcodec)
endif()

set_property(TARGET ${NVIMGCODEC_EXAMPLE_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${NVIMGCODEC_EXAMPLE_NAME} PUBLIC ${NVIMGCODEC_EXAMPLE_LINK_TARGET} CUDA::cudart_static)

# The CUDA driver stub needed by the static archive is carried by the exported
# nvimgcodec::nvimgcodec_static target, so consumers no longer link it by hand.
#
# Note: nvimgcodec_static.lib is built with the dynamic MSVC runtime (/MD), which
# matches the consumer default, so we must NOT force /MT here. The residual
# "LNK4098: LIBCMT conflicts" warning comes from CUDA::cudart_static (built /MT)
# and is harmless.

if(UNIX)
    target_link_libraries(${NVIMGCODEC_EXAMPLE_NAME} PUBLIC rt)
endif()

# On older GCC (e.g. GCC 8 on RHEL8) std::filesystem lives in a separate library;
# link it there so the sample builds outside our main toolchain.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1)
    target_link_libraries(${NVIMGCODEC_EXAMPLE_NAME} PUBLIC stdc++fs)
endif()

install(TARGETS ${NVIMGCODEC_EXAMPLE_NAME}
    DESTINATION "${NVIMGCODEC_INSTALL_BINDIR}" COMPONENT lib
)
