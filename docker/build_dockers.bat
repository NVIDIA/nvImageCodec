REM @ECHO OFF

REM Usage: .\docker\build_dockers.bat

REM Update version when changing anything in the Dockerfiles
set VERSION=10

SET SCRIPT_DIR=%~dp0

REM Do nothing but resets errorlevel
VER > NUL

call "%SCRIPT_DIR%config-docker.bat"  
if %errorlevel% neq 0 call "%SCRIPT_DIR%default-config-docker.bat"


docker build -t %REGISTRY_PREFIX%builder-cuda-13.0-vs17-amd64:v%VERSION% -f docker\Dockerfile.cuda130.amd64.vs17.deps .
docker build -t %REGISTRY_PREFIX%builder-cuda-12.9-vs17-amd64:v%VERSION% -f docker\Dockerfile.cuda129.amd64.vs17.deps .

docker push %REGISTRY_PREFIX%builder-cuda-13.0-vs17-amd64:v%VERSION%
docker push %REGISTRY_PREFIX%builder-cuda-12.9-vs17-amd64:v%VERSION%
