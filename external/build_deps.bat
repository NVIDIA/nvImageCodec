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

set SCRIPT_DIR=external
set INSTALL_PREFIX=..\..\..\install

if not defined GENERATOR (
    set GENERATOR="Visual Studio 17 2022"
)

echo Install prefix: %INSTALL_PREFIX%
echo Generator: %GENERATOR%

set PACKAGE_LIST=zlib ^
libjpeg-turbo ^
zstd ^
openjpeg ^
libtiff ^
opencv


for %%P in (%PACKAGE_LIST%) do (
    @echo Building %%P
    call %SCRIPT_DIR%/build_%%P.bat
)