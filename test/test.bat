@ECHO OFF
SETLOCAL

REM This script expects:
REM  * nvImageCodec python wheel to be in the same folder

REM Usage: test.bat <cuda-version-major > <path-to-python>
REM Example: test.bat 11 ..\Python3.9\

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

echo Creating python virtual environment .venv
%PATH_TO_PYTHON%python -m venv .venv
if %errorlevel% neq 0 exit /b %errorlevel%

echo Activating python virtual environment .venv
call .venv\Scripts\activate.bat

set CUDA_PATH=%cd%\.venv\Lib\site-packages\nvidia\cuda_runtime
set PATH=%cd%\.venv\Lib\site-packages\nvidia\cuda_runtime\bin;%PATH%

echo Installing python requirements
python.exe -m pip install --upgrade pip setuptools wheel
pip install --ignore-requires-python -r requirements_win_cu%CUDA_VERSION_MAJOR%.txt

set PATH=%cd%\.venv\Lib\site-packages\nvidia\nvjpeg\bin;%PATH%
set PATH=%cd%\.venv\Lib\site-packages\nvidia\nvjpeg2k\bin;%PATH%
set PATH=%cd%\.venv\Lib\site-packages\nvidia\nvtiff\bin;%PATH%
set PATH=%cd%\.venv\Lib\site-packages\nvidia\nvcomp;%PATH%

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

echo Runing python tests
pytest -v .\python
if %errorlevel% neq 0 set TEST_RETURN_CODE=%errorlevel%

echo Runing unit tests
nvimgcodec_tests.exe --resources_dir ..\resources
if %errorlevel% neq 0 set TEST_RETURN_CODE=%errorlevel%

if %TEST_RETURN_CODE% neq 0 echo "tests failed" & exit /b %TEST_RETURN_CODE%

echo Deactivating python virtual environment .venv
call deactivate