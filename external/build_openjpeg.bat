REM SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
REM SPDX-License-Identifier: Apache-2.0
REM
REM Licensed under the Apache License, Version 2.0 (the "License");
REM you may not use this file except in compliance with the License.
REM You may obtain a copy of the License at
REM
REM http://www.apache.org/licenses/LICENSE-2.0
REM
REM Unless required by applicable law or agreed to in writing, software
REM distributed under the License is distributed on an "AS IS" BASIS,
REM WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
REM See the License for the specific language governing permissions and
REM limitations under the License.

REM OpenJPEG
set OPENJPEG_VERSION=2.5.3
set OPENJPEG_URL=https://github.com/uclouvain/openjpeg/archive/refs/tags/v%OPENJPEG_VERSION%.tar.gz
set OPENJPEG_SHA256=368fe0468228e767433c9ebdea82ad9d801a3ad1e4234421f352c8b06e7aa707

REM Download and extract OpenJPEG if not already present
IF NOT EXIST external\openjpeg-%OPENJPEG_VERSION% (
    pushd external
    powershell -Command "Invoke-WebRequest -Uri %OPENJPEG_URL% -OutFile openjpeg-%OPENJPEG_VERSION%.tar.gz"
    REM Verify checksum
    powershell -Command "$hash = (Get-FileHash -Algorithm SHA256 openjpeg-%OPENJPEG_VERSION%.tar.gz).Hash; if ($hash -ne '%OPENJPEG_SHA256%') { Write-Error 'Checksum mismatch'; exit 1 }"
    if errorlevel 1 (
        del openjpeg-%OPENJPEG_VERSION%.tar.gz
        popd
        exit /b 1
    )
    REM Extract the .tar.gz archive (requires tar in PATH, e.g. from Git Bash or Windows 10+)
    tar -xf openjpeg-%OPENJPEG_VERSION%.tar.gz
    del openjpeg-%OPENJPEG_VERSION%.tar.gz
    popd
)

pushd external\openjpeg-%OPENJPEG_VERSION%

mkdir build_dir
pushd build_dir

cmake -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_INSTALL_PREFIX=%INSTALL_PREFIX% ^
      -DBUILD_CODEC=OFF ^
      -DBUILD_SHARED_LIBS=OFF ^
      -DBUILD_STATIC_LIBS=ON ^
      -DZLIB_LIBRARY="%INSTALL_PREFIX%/lib/zs.lib" ^
      -DZLIB_INCLUDE_DIR="%INSTALL_PREFIX%/include" ^
      -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL ^
      -G %GENERATOR% ^
       ..

cmake --build . --config Release
cmake --install . --config=Release

popd
rd /s /q build_dir

popd
