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

set(SRCS
    nvimgcodec_tests.cpp
    test_utils.cpp
    codec_test.cpp
    code_stream_test.cpp
    codec_registry_test.cpp
    plugin_framework_test.cpp
    thread_pool_test.cpp
#    processing_results_test.cpp
    device_guard_test.cpp
#    decoder_worker_test.cpp
#    encoder_worker_test.cpp
    parsers/bmp_test.cpp
    parsers/jpeg_test.cpp
    parsers/jpeg2k_test.cpp
    parsers/tiff_test.cpp
    parsers/png_test.cpp
    parsers/pnm_test.cpp
    parsers/webp_test.cpp
    api/can_decode_test.cpp
    api/can_encode_test.cpp
    imgproc/convert_cuda_test.cu
    imgproc/convert_test.cc
    imgproc/geom_mat_test.cu
    imgproc/geom_vec_test.cu
    imgproc/math_util_test.cc
)

include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES "${CUDAToolkit_INCLUDE_DIRS}")
check_cxx_source_compiles(
  "#include <nvjpeg.h>
  int main(){
    return NVJPEG_BACKEND_LOSSLESS_JPEG;
  }"
  NVJPEG_LOSSLESS_SUPPORTED)

if (BUILD_NVJPEG_EXT)
    list(APPEND SRCS extensions/nvjpeg_ext_encoder_test.cpp)
    list(APPEND SRCS extensions/nvjpeg_ext_decoder_test.cpp)
    if(NVJPEG_LOSSLESS_SUPPORTED)
        list(APPEND SRCS extensions/nvjpeg_ext_lossless_decoder_test.cpp)
    endif()
endif()


if (BUILD_NVJPEG2K_EXT)
    list(APPEND SRCS extensions/nvjpeg2k_ext_encoder_test.cpp)
    list(APPEND SRCS extensions/nvjpeg2k_ext_decoder_test.cpp)
endif()

if (BUILD_LIBJPEG_TURBO_EXT)
    list(APPEND SRCS extensions/libjpeg_turbo_ext_decoder_test.cpp)
endif()

if (BUILD_LIBTIFF_EXT)
    list(APPEND SRCS extensions/libtiff_ext_decoder_test.cpp)
endif()

if (BUILD_OPENCV_EXT)
    list(APPEND SRCS extensions/opencv_ext_encoder_test.cpp)
    list(APPEND SRCS extensions/opencv_ext_decoder_test.cpp)
endif()

if (BUILD_NVBMP_EXT)
    list(APPEND SRCS extensions/nvbmp_ext_encoder_test.cpp)
    list(APPEND SRCS extensions/nvbmp_ext_decoder_test.cpp)
endif()

if (BUILD_NVPNM_EXT)
    list(APPEND SRCS extensions/nvpnm_ext_encoder_test.cpp)
endif()

if (BUILD_NVTIFF_EXT)
    list(APPEND SRCS extensions/nvtiff_ext_decoder_test.cpp)
endif()

set(FILESTOPACK nvimgcodec_tests)

if (SKIP_NVTIFF_WITH_NVCOMP_TESTS)
    add_compile_definitions(SKIP_NVTIFF_WITH_NVCOMP_TESTS)
endif()

if(WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --compiler-options \" /Zc:preprocessor  \"  ")
endif()

add_executable(nvimgcodec_tests ${SRCS})
foreach(testapp nvimgcodec_tests)
    target_include_directories(${testapp} SYSTEM BEFORE
        PRIVATE
        ${GTEST_INCLUDE_DIRS}
    )

    target_link_libraries(${testapp} PRIVATE "-pie")
    set_target_properties(${testapp} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    if(UNIX)
        set(PTHREAD pthread)
    endif()

    set(TARGET_LIBS nvimgcodec_static)

    if (BUILD_NVJPEG_EXT)
        list(APPEND TARGET_LIBS nvjpeg_ext_static)
    endif()

    if (WITH_DYNAMIC_NVJPEG)
        list(APPEND TARGET_LIBS dynlink_nvjpeg)
    else()
        list(APPEND TARGET_LIBS CUDA::nvjpeg_static)
    endif()

    if (BUILD_NVJPEG2K_EXT)
        list(APPEND TARGET_LIBS nvjpeg2k_ext_static)
    endif()

    if (WITH_DYNAMIC_NVJPEG2K)
        list(APPEND TARGET_LIBS dynlink_nvjpeg2k)
    else()
        list(APPEND TARGET_LIBS ${NVJPEG2K_LIBRARY})
    endif()

    if (BUILD_LIBJPEG_TURBO_EXT)
        list(APPEND TARGET_LIBS libjpeg_turbo_ext_static)
        list(APPEND TARGET_LIBS ${JPEG_LIBRARY})
    endif()

    if (BUILD_LIBTIFF_EXT)
        list(APPEND TARGET_LIBS libtiff_ext_static)
        #TODO On Windows linking with OpenCV_LIBRARIES already takes tiff library
        if (NOT WIN32)
        list(APPEND TARGET_LIBS ${TIFF_LIBRARY})
        endif()
        list(APPEND TARGET_LIBS ${JPEG_LIBRARY})
        list(APPEND TARGET_LIBS ${ZLIB_LIBRARY})
        list(APPEND TARGET_LIBS ${ZSTD_LIBRARY})
    endif()

    if (BUILD_OPENCV_EXT)
        list(APPEND TARGET_LIBS opencv_ext_static)
    endif()

    if (BUILD_NVBMP_EXT)
        list(APPEND TARGET_LIBS nvbmp_ext_static)
    endif()

    if (BUILD_NVPNM_EXT)
        list(APPEND TARGET_LIBS nvpnm_ext_static)
    endif()

    if (BUILD_NVTIFF_EXT)
        list(APPEND TARGET_LIBS nvtiff_ext_static)
        if (WITH_DYNAMIC_NVTIFF)
            list(APPEND TARGET_LIBS dynlink_nvtiff)
        else()
            list(APPEND TARGET_LIBS ${NVTIFF_LIB})
        endif()
    endif()

    target_link_libraries(${testapp} PUBLIC
        ${TARGET_LIBS}
        ${OpenCV_LIBRARIES}
        CUDA::cudart_static
        CUDA::cuda_driver
        gtest
        gmock
        ${PTHREAD})
endforeach()

install(TARGETS ${FILESTOPACK}
    DESTINATION test COMPONENT tests
)

install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/python
    DESTINATION test
    COMPONENT tests
)

install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_transcode.py
    DESTINATION test
    COMPONENT tests
)

if(WIN32)
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/test.bat
        ${CMAKE_CURRENT_SOURCE_DIR}/requirements_win_cu11.txt
        ${CMAKE_CURRENT_SOURCE_DIR}/requirements_win_cu12.txt
        DESTINATION test
        COMPONENT tests
    )
endif()

install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/../resources
    DESTINATION .
    COMPONENT tests
)



add_test(NAME nvImageCodecTest-L0 COMMAND nvimgcodec_tests --gtest_filter=-*Parser*:*Ext* --resources_dir ../resources WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME nvImageCodecTest-L1-Builtin_Parsers COMMAND nvimgcodec_tests --gtest_filter=*Parser* --resources_dir ../resources WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME nvImageCodecTest-L1-Extensions COMMAND nvimgcodec_tests --gtest_filter=*Ext* --resources_dir ../resources WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
#TODO add_test(NAME nvImageCodecTest-L2-Transcoding COMMAND ${PYTHON_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
