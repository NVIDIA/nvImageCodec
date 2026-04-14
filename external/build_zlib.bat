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

REM zlib
set ZLIB_VERSION=1.3.1.2
set ZLIB_URL=https://github.com/madler/zlib/archive/refs/tags/v%ZLIB_VERSION%.tar.gz
set ZLIB_SHA256=fbf1c8476136693e6c3f1fa26e6d8c4f2c8b5a5c44340c04df349dad02eed09e

REM Download and extract zlib if not already present
IF NOT EXIST external\zlib-%ZLIB_VERSION% (
    pushd external
    powershell -Command "Invoke-WebRequest -Uri %ZLIB_URL% -OutFile zlib-%ZLIB_VERSION%.tar.gz"
    REM Verify checksum
    powershell -Command "$hash = (Get-FileHash -Algorithm SHA256 zlib-%ZLIB_VERSION%.tar.gz).Hash; if ($hash -ne '%ZLIB_SHA256%') { Write-Error 'Checksum mismatch'; exit 1 }"
    if errorlevel 1 (
        del zlib-%ZLIB_VERSION%.tar.gz
        popd
        exit /b 1
    )
    REM Extract the .tar.gz archive (requires tar in PATH, e.g. from Git Bash or Windows 10+)
    tar -xf zlib-%ZLIB_VERSION%.tar.gz
    del zlib-%ZLIB_VERSION%.tar.gz
    popd
)

pushd external\zlib-%ZLIB_VERSION%

mkdir build_dir
pushd build_dir

cmake -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_INSTALL_PREFIX=%INSTALL_PREFIX% ^
      -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL ^
      -DZLIB_BUILD_SHARED=OFF ^
      -DZLIB_BUILD_STATIC=ON ^
      -G %GENERATOR% ^
       ..

cmake --build . --config Release
cmake --install . --config=Release

popd
rd /s /q build_dir

popd

