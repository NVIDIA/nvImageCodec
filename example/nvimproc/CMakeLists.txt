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

set(NVIMGCODEC_EXAMPLE_NAME nvimproc)

set(NVIMGCODEC_EXAMPLE_SRC
        main.cpp
)

find_package(NVCV_TYPES REQUIRED)
find_package(CVCUDA REQUIRED)

add_executable(${NVIMGCODEC_EXAMPLE_NAME} ${NVIMGCODEC_EXAMPLE_SRC})

target_link_libraries(${NVIMGCODEC_EXAMPLE_NAME} PUBLIC CUDA::cudart nvimgcodec nvcv_types cvcuda)

install(TARGETS ${NVIMGCODEC_EXAMPLE_NAME}
        DESTINATION bin COMPONENT lib
)