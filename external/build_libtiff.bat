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

REM libtiff
set LIBTIFF_VERSION=4.7.1
set LIBTIFF_URL=https://gitlab.com/libtiff/libtiff/-/archive/v%LIBTIFF_VERSION%/libtiff-v%LIBTIFF_VERSION%.tar.gz
set LIBTIFF_SHA256=6ab956415ddb6cae497faad18398ff0fd056d17d8bf7b5921cc194c55b0191fc

REM Download and extract libtiff if not already present
IF NOT EXIST external\libtiff-%LIBTIFF_VERSION% (
    pushd external
    powershell -Command "Invoke-WebRequest -Uri %LIBTIFF_URL% -OutFile libtiff-%LIBTIFF_VERSION%.tar.gz"
    REM Verify checksum
    powershell -Command "$hash = (Get-FileHash -Algorithm SHA256 libtiff-%LIBTIFF_VERSION%.tar.gz).Hash; if ($hash -ne '%LIBTIFF_SHA256%') { Write-Error 'Checksum mismatch'; exit 1 }"
    if errorlevel 1 (
        del libtiff-%LIBTIFF_VERSION%.tar.gz
        popd
        exit /b 1
    )
    REM Extract the .tar.gz archive (requires tar in PATH, e.g. from Git Bash or Windows 10+)
    tar -xf libtiff-%LIBTIFF_VERSION%.tar.gz
    ren libtiff-v%LIBTIFF_VERSION% libtiff-%LIBTIFF_VERSION%
    del libtiff-%LIBTIFF_VERSION%.tar.gz
    popd
)

set CURRDIR=%cd%

pushd external\libtiff-%LIBTIFF_VERSION%

mkdir build_dir
pushd build_dir

cmake -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_INSTALL_PREFIX=%INSTALL_PREFIX% ^
      -DCMAKE_PREFIX_PATH=%INSTALL_PREFIX% ^
      -DBUILD_SHARED_LIBS=OFF ^
      -DZLIB_USE_STATIC_LIBS=ON ^
      -DZLIB_LIBRARY="%INSTALL_PREFIX%/lib/zs.lib" ^
      -DZLIB_INCLUDE_DIR="%INSTALL_PREFIX%/include" ^
      -Djbig=OFF ^
      -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL ^
      -G %GENERATOR% ^
       ..

cmake --build . --config Release
cmake --install . --config=Release

popd
rd /s /q build_dir

popd