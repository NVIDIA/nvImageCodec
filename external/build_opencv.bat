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
set OPENCV_VERSION=4.13.0
set OPENCV_URL=https://github.com/opencv/opencv/archive/refs/tags/%OPENCV_VERSION%.tar.gz
set OPENCV_SHA256=1d40ca017ea51c533cf9fd5cbde5b5fe7ae248291ddf2af99d4c17cf8e13017d

REM Download and extract OpenCV if not already present
IF NOT EXIST external\opencv-%OPENCV_VERSION% (
    pushd external
    powershell -Command "Invoke-WebRequest -Uri %OPENCV_URL% -OutFile opencv-%OPENCV_VERSION%.tar.gz"
    REM Verify checksum
    powershell -Command "$hash = (Get-FileHash -Algorithm SHA256 opencv-%OPENCV_VERSION%.tar.gz).Hash; if ($hash -ne '%OPENCV_SHA256%') { Write-Error 'Checksum mismatch'; exit 1 }"
    if errorlevel 1 (
        del opencv-%OPENCV_VERSION%.tar.gz
        popd
        exit /b 1
    )
    REM Extract the .tar.gz archive (requires tar in PATH, e.g. from Git Bash or Windows 10+)
    tar -xf opencv-%OPENCV_VERSION%.tar.gz
    del opencv-%OPENCV_VERSION%.tar.gz
    popd
)

pushd external\opencv-%OPENCV_VERSION%

mkdir build_dir
pushd build_dir

REM -DCMAKE_TOOLCHAIN_FILE=$PWD/../platforms/${OPENCV_TOOLCHAIN_FILE} \

REM Create a CMake script to define ZSTD::ZSTD target
set ZSTD_TARGET_CMAKE=%INSTALL_PREFIX%\define_zstd_target.cmake
echo if(NOT TARGET ZSTD::ZSTD) > %ZSTD_TARGET_CMAKE%
echo   add_library(ZSTD::ZSTD STATIC IMPORTED) >> %ZSTD_TARGET_CMAKE%
echo   set_target_properties(ZSTD::ZSTD PROPERTIES >> %ZSTD_TARGET_CMAKE%
echo     IMPORTED_LOCATION "%INSTALL_PREFIX%/lib/zstd_static.lib" >> %ZSTD_TARGET_CMAKE%
echo     INTERFACE_INCLUDE_DIRECTORIES "%INSTALL_PREFIX%/include" >> %ZSTD_TARGET_CMAKE%
echo   ) >> %ZSTD_TARGET_CMAKE%
echo endif() >> %ZSTD_TARGET_CMAKE%

cmake -DCMAKE_BUILD_TYPE=Release ^
      -DVIBRANTE_PDK:STRING=/ ^
      -DCMAKE_INSTALL_PREFIX=%INSTALL_PREFIX% ^
      -DCMAKE_PREFIX_PATH=%INSTALL_PREFIX% ^
      -DCMAKE_PROJECT_INCLUDE=%ZSTD_TARGET_CMAKE% ^
      -DBUILD_LIST=core,improc,imgcodecs ^
      -DBUILD_SHARED_LIBS=OFF ^
      -DWITH_EIGEN=OFF ^
      -DWITH_CUDA=OFF ^
      -DWITH_1394=OFF ^
      -DWITH_IPP=OFF ^
      -DWITH_OPENCL=OFF ^
      -DWITH_GTK=OFF ^
      -DBUILD_JPEG=OFF ^
      -DBUILD_OPENJPEG=OFF ^
      -DWITH_JPEG=ON ^
      -DBUILD_TIFF=OFF ^
      -DWITH_TIFF=ON ^
      -DBUILD_ZLIB=OFF ^
      -DWITH_ZLIB=ON ^
      -DZLIB_LIBRARY="%INSTALL_PREFIX%/lib/zs.lib" ^
      -DZLIB_INCLUDE_DIR="%INSTALL_PREFIX%/include" ^
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
      -DBUILD_WITH_STATIC_CRT=OFF ^
      -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL ^
      -G %GENERATOR% ^
       ..

cmake --build . --config Release
cmake --install . --config=Release

popd
rd /s /q build_dir

popd
