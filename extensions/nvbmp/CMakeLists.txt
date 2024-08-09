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

set(NVIMGCODEC_NVBMP_EXT_LIBRARY_NAME nvbmp_ext)

set(NVIMGCODEC_NVBMP_EXT_SRC
  nvbmp_ext.cpp
  encoder.cpp
  decoder.cpp
)

add_library(${NVIMGCODEC_NVBMP_EXT_LIBRARY_NAME} SHARED ${NVIMGCODEC_NVBMP_EXT_SRC} ext_module.cpp)
add_library(${NVIMGCODEC_NVBMP_EXT_LIBRARY_NAME}_static STATIC ${NVIMGCODEC_NVBMP_EXT_SRC})

if(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -fvisibility=hidden -Wl,--exclude-libs,ALL")
  target_link_libraries(${NVIMGCODEC_NVBMP_EXT_LIBRARY_NAME} PRIVATE "${NVIMGCODEC_COMMON_DEPENDENCIES}")
  target_link_libraries(${NVIMGCODEC_NVBMP_EXT_LIBRARY_NAME}_static PRIVATE "${NVIMGCODEC_COMMON_DEPENDENCIES}")

  set_target_properties(${NVIMGCODEC_NVBMP_EXT_LIBRARY_NAME} ${NVIMGCODEC_NVBMP_EXT_LIBRARY_NAME}_static PROPERTIES
                        POSITION_INDEPENDENT_CODE ON
                        VERSION ${PROJECT_VERSION}
                        NO_SONAME OFF)
else()
  set_target_properties(${NVIMGCODEC_NVBMP_EXT_LIBRARY_NAME} PROPERTIES
                        POSITION_INDEPENDENT_CODE ON
                        VERSION ${PROJECT_VERSION}
                        OUTPUT_NAME ${NVIMGCODEC_NVBMP_EXT_LIBRARY_NAME}_${PROJECT_VERSION_MAJOR}
                        ARCHIVE_OUTPUT_NAME ${NVIMGCODEC_NVBMP_EXT_LIBRARY_NAME})
endif()

if(UNIX)
  install(TARGETS ${NVIMGCODEC_NVBMP_EXT_LIBRARY_NAME} ${NVIMGCODEC_NVBMP_EXT_LIBRARY_NAME}_static
          LIBRARY DESTINATION extensions NAMELINK_SKIP COMPONENT lib
          ARCHIVE DESTINATION lib64 COMPONENT lib
          PUBLIC_HEADER DESTINATION include COMPONENT lib)
else()
  install(TARGETS ${NVIMGCODEC_NVBMP_EXT_LIBRARY_NAME}
          RUNTIME DESTINATION extensions COMPONENT lib
          LIBRARY DESTINATION lib COMPONENT lib
          ARCHIVE DESTINATION lib COMPONENT lib
          PUBLIC_HEADER DESTINATION include COMPONENT lib)
endif()
