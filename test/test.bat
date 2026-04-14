@ECHO OFF
SETLOCAL

REM This script expects:
REM  * nvImageCodec python wheel to be in the same folder

REM Usage: test.bat <cuda-version-major> <path-to-python> <run-slow-tests>
REM Example: test.bat 11 ..\Python3.10\ false
REM          test.bat 11 python true

if [%1]==[] (
    echo CUDA toolkit version major not specified
    exit -1
) else (
    set "CUDA_VERSION_MAJOR=%1"
)

if [%2]==[] (
    set "PATH_TO_PYTHON="
) else (
    set "PATH_TO_PYTHON=%2"
)

if [%3]==[] (
    if not defined RUN_SLOW_TESTS set "RUN_SLOW_TESTS=false"
) else (
    set "RUN_SLOW_TESTS=%3"
)

echo Creating python virtual environment .nvimgcodec_test_venv
%PATH_TO_PYTHON%python -m venv .nvimgcodec_test_venv
if %errorlevel% neq 0 exit /b %errorlevel%

echo Activating python virtual environment .nvimgcodec_test_venv
call .nvimgcodec_test_venv\Scripts\activate.bat

set CUDA_PATH=%cd%\.nvimgcodec_test_venv\Lib\site-packages\nvidia\cuda_runtime
set PATH=%cd%\.nvimgcodec_test_venv\Lib\site-packages\nvidia\cuda_runtime\bin;%PATH%

echo Installing python requirements
python.exe -m pip install --upgrade pip setuptools wheel
pip install appdirs
pip install -r requirements_win_cu%CUDA_VERSION_MAJOR%.txt
if %errorlevel% neq 0 exit /b %errorlevel%

@REM nvjpeg in cuda13 have different install path that previous cuda versions
set PATH=%cd%\.nvimgcodec_test_venv\Lib\site-packages\nvidia\nvjpeg\bin;%PATH%
set PATH=%cd%\.nvimgcodec_test_venv\Lib\site-packages\nvidia\cu13\bin\x86_64\;%PATH%

set PATH=%cd%\.nvimgcodec_test_venv\Lib\site-packages\nvidia\nvjpeg2k\bin;%PATH%
set PATH=%cd%\.nvimgcodec_test_venv\Lib\site-packages\nvidia\nvtiff\bin;%PATH%
set PATH=%cd%\.nvimgcodec_test_venv\Lib\site-packages\nvidia\libnvcomp\bin;%PATH%

set NVIMGCODEC_EXTENSIONS_PATH=%cd%\..\extensions

echo Installing nvImageCodec python wheel(s) from current folder
for %%G in (".\*.whl") do (
    echo Found and installing: %%G
    pip install -I %%G
)

set TEST_RETURN_CODE=0
echo Runing transcoder (nvimtrans) tests
pytest -v test_transcode.py
if %errorlevel% neq 0 set TEST_RETURN_CODE=%errorlevel%

set EXTRA_PYTEST_ARGS=
@REM If PATH_TO_PYTHON contains "Python314", set EXTRA_PYTEST_ARGS to ignore the integration tests
if not "%PATH_TO_PYTHON:Python314=%"=="%PATH_TO_PYTHON%" set EXTRA_PYTEST_ARGS=--ignore=python\integration

echo Running python tests (excluding slow tests)
pytest -v .\python %EXTRA_PYTEST_ARGS%
if %errorlevel% neq 0 set TEST_RETURN_CODE=%errorlevel%

if "%RUN_SLOW_TESTS%"=="true" goto :run_slow_tests
goto :skip_slow_tests

:run_slow_tests
echo Running slow tests (parallel workers: 6)
pytest -v .\python -m "slow" -n 6
set SLOW_TEST_RESULT=%errorlevel%
if %SLOW_TEST_RESULT% equ 5 (
    echo No slow tests collected or all skipped, continuing.
) else if %SLOW_TEST_RESULT% neq 0 (
    set TEST_RETURN_CODE=%SLOW_TEST_RESULT%
)
:skip_slow_tests

echo Runing unit tests
nvimgcodec_tests.exe --resources_dir ..\resources
if %errorlevel% neq 0 set TEST_RETURN_CODE=%errorlevel%

if %TEST_RETURN_CODE% neq 0 echo "tests failed" & exit /b %TEST_RETURN_CODE%

echo Deactivating python virtual environment .nvimgcodec_test_venv
call deactivate
