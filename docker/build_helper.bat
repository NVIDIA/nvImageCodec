@ECHO ON
SETLOCAL

REM Usage: build_helper.bat <cmake-build-dir> <cuda-version> [cmake-flags [...]]
REM Example: build_helper.bat ..\build 10.1 -DBUILD_EXAMPLES=FALSE -DBUILD_SHARED_LIBS=TRUE -DCMAKE_BUILD_TYPE=Release

IF [%1]==[] GOTO :error_build_dir
IF [%2]==[] GOTO :error_cuda_version

if not defined GENERATOR (
    SET GENERATOR="Visual Studio 17 2022"
)

SET "SCRIPT_DIR=%~dp0"
SET "SOURCE_DIR=%SCRIPT_DIR%\.."
SET "BUILD_DIR=%1"
SET "CUDA_VERSION=%2"
SET CMAKE_ARGS=

for /F "tokens=1,2*" %%a in ("%*") do (
  set BUILD_DIR=%%a
  set CUDA_VERSION=%%b
  set CMAKE_ARGS=%%c
)

echo "%SOURCE_DIR%"

set PATH=%PATH%;%SOURCE_DIR%\install\include;%SOURCE_DIR%\install\lib;%SOURCE_DIR%\install\x64\vc17\staticlib
set PATH=%PATH%;c:\nvimgcodec_deps\install\include;c:\nvimgcodec_deps\install\lib;c:\nvimgcodec_deps\install\x64\vc17\staticlib

if not defined HOST_PYTHON_VERSION (
    SET HOST_PYTHON_VERSION=3.12
)

set "HOST_PYTHON_VERSION_NODOT=%HOST_PYTHON_VERSION:.=%"
if not defined Python_EXECUTABLE (
    for /F "delims=" %%p in ('py -%HOST_PYTHON_VERSION% -c "import sys; print(sys.executable)" 2^>nul') do set "Python_EXECUTABLE=%%p"
)
if not defined Python_EXECUTABLE (
    set "Python_EXECUTABLE=C:\Program Files\Python%HOST_PYTHON_VERSION_NODOT%\python.exe"
)
if not exist "%Python_EXECUTABLE%" (
    echo ERROR: Python_EXECUTABLE=%Python_EXECUTABLE% not found.
    exit /b 1
)

set "HOST_PYTHON_VERSION_FILE=%TEMP%\nvimgcodec_host_python_version_%RANDOM%.txt"
"%Python_EXECUTABLE%" -c "import sys; print(sys.version_info[0], sys.version_info[1], sep=chr(46))" > "%HOST_PYTHON_VERSION_FILE%"
if %errorlevel% neq 0 exit /b %errorlevel%
set /p HOST_PYTHON_ACTUAL_VERSION=<"%HOST_PYTHON_VERSION_FILE%"
del "%HOST_PYTHON_VERSION_FILE%" >nul 2>nul
if not "%HOST_PYTHON_ACTUAL_VERSION%"=="%HOST_PYTHON_VERSION%" (
    echo ERROR: Python_EXECUTABLE=%Python_EXECUTABLE% is Python %HOST_PYTHON_ACTUAL_VERSION%, expected %HOST_PYTHON_VERSION%.
    exit /b 1
)

for %%p in ("%Python_EXECUTABLE%") do set "HOST_PYTHON_ROOT=%%~dpp"
if "%HOST_PYTHON_ROOT:~-1%"=="\" set "HOST_PYTHON_ROOT=%HOST_PYTHON_ROOT:~0,-1%"
set "PYTHON%HOST_PYTHON_VERSION_NODOT%_ROOT_DIR=%HOST_PYTHON_ROOT%"

"%Python_EXECUTABLE%" -c "import clang.cindex, setuptools, wheel"
if %errorlevel% neq 0 (
    echo ERROR: Required Python build packages are missing for %Python_EXECUTABLE%.
    echo ERROR: Rebuild the Windows builder image so setuptools, wheel, clang==14.0, and libclang==14.0.1 are available.
    exit /b %errorlevel%
)

cmake -DBUILD_ID="%NVIDIA_BUILD_ID%" ^
 -DCMAKE_BUILD_TYPE=Release ^
 -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL ^
 -DCMAKE_LIBRARY_PATH="c:/nvimgcodec_deps/install/lib" ^
 -DZLIB_LIBRARY="c:/nvimgcodec_deps/install/lib/zs.lib" ^
 -DZLIB_INCLUDE_DIR="c:/nvimgcodec_deps/install/include" ^
 -DPython_EXECUTABLE:FILEPATH="%Python_EXECUTABLE%" ^
 -S %SOURCE_DIR% ^
 -B %BUILD_DIR% ^
 -G %GENERATOR% ^
 -T host=x64 ^
 -A x64

if %errorlevel% neq 0 exit /b %errorlevel%

pushd %BUILD_DIR%

cmake --build . --config Release --parallel

if defined INSTALL_PREFIX_DIR (
    cmake --install . --prefix "%INSTALL_PREFIX_DIR%" --config Release
    REM Multi-config MSVC generator installs only the unnamed default component
    REM unless --component is specified. Explicitly install the `tests` component
    REM so test/test.bat, requirements_win_*.txt, etc. land in INSTALL_PREFIX_DIR/test/.
    cmake --install . --prefix "%INSTALL_PREFIX_DIR%" --config Release --component tests
)
if defined INSTALL_LIB_PREFIX_DIR (
    cmake --install . --prefix "%INSTALL_LIB_PREFIX_DIR%" --config Release --component lib
)
if defined INSTALL_TEST_PREFIX_DIR (
    cmake --install . --prefix "%INSTALL_TEST_PREFIX_DIR%" --config Release --component tests
)

REM Guards for the exported CMake package. Mirrors build_helper.sh: the driver
REM stages its own install, so it does not depend on the INSTALL_* variables
REM above being set. Set SKIP_CMAKE_PACKAGE_GUARDS=1 to bypass.
if not "%SKIP_CMAKE_PACKAGE_GUARDS%"=="1" (
    if exist ..\test\cmake_package\run_package_tests.cmake (
        echo Running CMake package guards...
        cmake -DNVIMGCODEC_BUILD_DIR=. -P ..\test\cmake_package\run_package_tests.cmake
        if %errorlevel% neq 0 exit /b %errorlevel%
    )
)

cpack --config CPackConfig.cmake -DCMAKE_BUILD_TYPE=Release

cmake --build . --target wheel --config Release --parallel

if defined WHL_OUTDIR (
    if not exist "%WHL_OUTDIR%" mkdir "%WHL_OUTDIR%"
    copy /Y python\*.whl "%WHL_OUTDIR%\"
)

if defined INSTALL_PREFIX_DIR (
    if not exist "%INSTALL_PREFIX_DIR%\test" mkdir "%INSTALL_PREFIX_DIR%\test"
    copy /Y python\*.whl "%INSTALL_PREFIX_DIR%\test\"
)

if defined INSTALL_TEST_PREFIX_DIR (
    if not exist "%INSTALL_TEST_PREFIX_DIR%\test" mkdir "%INSTALL_TEST_PREFIX_DIR%\test"
    copy /Y python\*.whl "%INSTALL_TEST_PREFIX_DIR%\test\"
)

call :archive_install_prefix INSTALL_ARCHIVE INSTALL_PREFIX_DIR
if %errorlevel% neq 0 exit /b %errorlevel%
call :archive_install_prefix INSTALL_LIB_ARCHIVE INSTALL_LIB_PREFIX_DIR
if %errorlevel% neq 0 exit /b %errorlevel%
call :archive_install_prefix INSTALL_TEST_ARCHIVE INSTALL_TEST_PREFIX_DIR
if %errorlevel% neq 0 exit /b %errorlevel%

popd

GOTO :eof

:error_build_dir
echo Build directory not specified
GOTO :eof

:error_cuda_version
echo CUDA toolkit version not specified

GOTO :eof

:archive_install_prefix
set "ARCHIVE_VAR=%~1"
set "PREFIX_VAR=%~2"
call set "ARCHIVE_VALUE=%%%ARCHIVE_VAR%%%"
if not defined ARCHIVE_VALUE exit /b 0
call set "PREFIX_VALUE=%%%PREFIX_VAR%%%"
if not defined PREFIX_VALUE (
    echo ERROR: %ARCHIVE_VAR% requires %PREFIX_VAR%
    exit /b 1
)
tar -czf "%ARCHIVE_VALUE%" -C "%PREFIX_VALUE%" .
exit /b %errorlevel%

:eof
