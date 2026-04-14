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

REM libjpeg-turbo
set LIBJPEG_TURBO_VERSION=3.1.3
set LIBJPEG_TURBO_URL=https://github.com/libjpeg-turbo/libjpeg-turbo/archive/refs/tags/%LIBJPEG_TURBO_VERSION%.tar.gz
set LIBJPEG_TURBO_SHA256=3a13a5ba767dc8264bc40b185e41368a80d5d5f945944d1dbaa4b2fb0099f4e5

REM Download and extract libjpeg-turbo if not already present
IF NOT EXIST external\libjpeg-turbo-%LIBJPEG_TURBO_VERSION% (
    pushd external
    powershell -Command "Invoke-WebRequest -Uri %LIBJPEG_TURBO_URL% -OutFile libjpeg-turbo-%LIBJPEG_TURBO_VERSION%.tar.gz"
    REM Verify checksum
    powershell -Command "$hash = (Get-FileHash -Algorithm SHA256 libjpeg-turbo-%LIBJPEG_TURBO_VERSION%.tar.gz).Hash; if ($hash -ne '%LIBJPEG_TURBO_SHA256%') { Write-Error 'Checksum mismatch'; exit 1 }"
    if errorlevel 1 (
        del libjpeg-turbo-%LIBJPEG_TURBO_VERSION%.tar.gz
        popd
        exit /b 1
    )
    REM Extract the .tar.gz archive (requires tar in PATH, e.g. from Git Bash or Windows 10+)
    tar -xf libjpeg-turbo-%LIBJPEG_TURBO_VERSION%.tar.gz
    del libjpeg-turbo-%LIBJPEG_TURBO_VERSION%.tar.gz
    popd
)

pushd external\libjpeg-turbo-%LIBJPEG_TURBO_VERSION%\

mkdir build_dir
pushd build_dir

cmake -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_INSTALL_PREFIX=%INSTALL_PREFIX% ^
      -DENABLE_SHARED=FALSE -DENABLE_STATIC=TRUE ^
      -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL ^
      -G %GENERATOR% ^
       ..

cmake --build . --config Release
cmake --install . --config=Release

popd
rd /s /q build_dir

popd
