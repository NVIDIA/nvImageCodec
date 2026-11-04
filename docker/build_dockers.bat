REM @ECHO OFF

REM Usage: .\docker\build_dockers.bat

REM Update version when changing anything in the Dockerfiles
set VERSION=26

SET SCRIPT_DIR=%~dp0

REM Do nothing but resets errorlevel
VER > NUL

call "%SCRIPT_DIR%config-docker.bat"  
if %errorlevel% neq 0 call "%SCRIPT_DIR%default-config-docker.bat"

REM One Dockerfile for both CUDA versions; the version-specific values live here.
REM Preserve each version's previously working installer policy: CUDA 12.9 uses the
REM seven-component subset, while CUDA 13.4 uses the installer's default silent set.
set CUDA_129_URL=https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_576.57_windows.exe
set CUDA_129_ARGS=-s nvcc_12.9 cudart_12.9 nvjpeg_12.9 nvjpeg_dev_12.9 visual_studio_integration_12.9 nvprof_12.9 cuda_profiler_api_12.9

set CUDA_134_URL=https://developer.download.nvidia.com/compute/cuda/13.4.1/local_installers/cuda_13.4.1_windows_x86_64.exe
set CUDA_134_ARGS=-s

REM Stop at the first failure: without this a failed build falls through to the
REM push, which would publish whatever stale image already carries that tag.
docker build -m 4GB -t %REGISTRY_PREFIX%builder-cuda-13.4-vs17-amd64:v%VERSION% ^
    --build-arg "CUDA_INSTALLER_URL=%CUDA_134_URL%" ^
    --build-arg "CUDA_INSTALLER_ARGS=%CUDA_134_ARGS%" ^
    --build-arg "CUDA_VER_SHORT=13.4" ^
    -f docker\Dockerfile.amd64.vs17.deps .
if %errorlevel% neq 0 exit /b %errorlevel%

docker build -m 4GB -t %REGISTRY_PREFIX%builder-cuda-12.9-vs17-amd64:v%VERSION% ^
    --build-arg "CUDA_INSTALLER_URL=%CUDA_129_URL%" ^
    --build-arg "CUDA_INSTALLER_ARGS=%CUDA_129_ARGS%" ^
    --build-arg "CUDA_VER_SHORT=12.9" ^
    -f docker\Dockerfile.amd64.vs17.deps .
if %errorlevel% neq 0 exit /b %errorlevel%

call :push_with_retry %REGISTRY_PREFIX%builder-cuda-13.4-vs17-amd64:v%VERSION%
if %errorlevel% neq 0 exit /b %errorlevel%

call :push_with_retry %REGISTRY_PREFIX%builder-cuda-12.9-vs17-amd64:v%VERSION%
if %errorlevel% neq 0 exit /b %errorlevel%

exit /b 0

:push_with_retry
set PUSH_IMAGE=%~1
set PUSH_ATTEMPT=1

:push_retry
docker push %PUSH_IMAGE%
if not errorlevel 1 exit /b 0
if %PUSH_ATTEMPT% geq 5 exit /b 1
set /a PUSH_ATTEMPT+=1
powershell -NoProfile -Command "Start-Sleep -Seconds 15"
goto push_retry
