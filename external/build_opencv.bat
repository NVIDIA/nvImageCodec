REM  SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
REM  SPDX-License-Identifier: Apache-2.0
REM 
REM  Licensed under the Apache License, Version 2.0 (the "License");
REM  you may not use this file except in compliance with the License.
REM  You may obtain a copy of the License at
REM 
REM  http://www.apache.org/licenses/LICENSE-2.0
REM 
REM  Unless required by applicable law or agreed to in writing, software
REM  distributed under the License is distributed on an "AS IS" BASIS,
REM  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
REM  See the License for the specific language governing permissions and
REM  limitations under the License.

REM OpenCV

pushd external\opencv

mkdir build_dir
pushd build_dir

REM -DCMAKE_TOOLCHAIN_FILE=$PWD/../platforms/${OPENCV_TOOLCHAIN_FILE} \

cmake -DCMAKE_BUILD_TYPE=Release ^
      -DVIBRANTE_PDK:STRING=/ ^
      -DCMAKE_INSTALL_PREFIX=%INSTALL_PREFIX% ^
      -DBUILD_LIST=core,improc,imgcodecs ^
      -DBUILD_SHARED_LIBS=OFF ^
      -DWITH_EIGEN=OFF ^
      -DWITH_CUDA=OFF ^
      -DWITH_1394=OFF ^
      -DWITH_IPP=OFF ^
      -DWITH_OPENCL=OFF ^
      -DWITH_GTK=OFF ^
      -DBUILD_JPEG=OFF ^
      -DWITH_JPEG=ON ^
      -DBUILD_TIFF=OFF ^
      -DWITH_TIFF=ON ^
      -DWITH_QUIRC=OFF ^
      -DWITH_ADE=OFF ^
      -DBUILD_JASPER=OFF ^
      -DBUILD_DOCS=OFF ^
      -DBUILD_TESTS=OFF ^
      -DBUILD_PERF_TESTS=OFF ^
      -DBUILD_PNG=ON ^
      -DWITH_WEBP=ON ^
      -DBUILD_opencv_cudalegacy=OFF ^
      -DBUILD_opencv_stitching=OFF ^
      -DWITH_TBB=OFF ^
      -DWITH_QUIRC=OFF ^
      -DWITH_OPENMP=OFF ^
      -DWITH_PTHREADS_PF=OFF ^
      -DBUILD_EXAMPLES=OFF ^
      -DBUILD_opencv_java=OFF ^
      -DBUILD_opencv_python2=OFF ^
      -DBUILD_opencv_python3=OFF ^
      -DWITH_PROTOBUF=OFF ^
      -DWITH_FFMPEG=OFF ^
      -DWITH_GSTREAMER=OFF ^
      -DWITH_GSTREAMER_0_10=OFF ^
      -DWITH_VTK=OFF ^
      -DWITH_OPENEXR=OFF ^
      -DINSTALL_C_EXAMPLES=OFF ^
      -DINSTALL_TESTS=OFF ^
      -DVIBRANTE=TRUE ^
      -DWITH_CSTRIPES=OFF ^
      -DCMAKE_PREFIX_PATH=%INSTALL_PREFIX% ^
      -DBUILD_WITH_STATIC_CRT=OFF ^
      -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL ^
      -G %GENERATOR% ^
       ..

cmake --build . --config Release
cmake --install . --config=Release

popd
rd /s /q build_dir

popd
