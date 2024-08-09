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

pushd external\libjpeg-turbo\

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
