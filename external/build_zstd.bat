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

REM zstandard compression library
set ZSTD_VERSION=1.5.7
set ZSTD_URL=https://github.com/facebook/zstd/archive/refs/tags/v%ZSTD_VERSION%.tar.gz
set ZSTD_SHA256=37d7284556b20954e56e1ca85b80226768902e2edabd3b649e9e72c0c9012ee3

REM Download and extract zstd if not already present
IF NOT EXIST external\zstd-%ZSTD_VERSION% (
    pushd external
    powershell -Command "Invoke-WebRequest -Uri %ZSTD_URL% -OutFile zstd-%ZSTD_VERSION%.tar.gz"
    REM Verify checksum
    powershell -Command "$hash = (Get-FileHash -Algorithm SHA256 zstd-%ZSTD_VERSION%.tar.gz).Hash; if ($hash -ne '%ZSTD_SHA256%') { Write-Error 'Checksum mismatch'; exit 1 }"
    if errorlevel 1 (
        del zstd-%ZSTD_VERSION%.tar.gz
        popd
        exit /b 1
    )
    REM Extract the .tar.gz archive (requires tar in PATH, e.g. from Git Bash or Windows 10+)
    tar -xf zstd-%ZSTD_VERSION%.tar.gz
    del zstd-%ZSTD_VERSION%.tar.gz
    popd
)

set CURRDIR=%cd%

pushd external\zstd-%ZSTD_VERSION%
REM https://github.com/facebook/zstd/issues/3999
pip install patch
python -m patch -p1 %CURRDIR%\external\patches\zstd-fix-windows-rc-compile.patch

mkdir build_dir
pushd build_dir

cmake -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_INSTALL_PREFIX=%INSTALL_PREFIX% ^
      -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL ^
      -DZSTD_BUILD_SHARED=OFF ^
      -DZSTD_BUILD_STATIC=ON ^
      -G %GENERATOR% ^
       ../build/cmake

cmake --build . --config Release
cmake --install . --config=Release

popd
rd /s /q build_dir

popd



