REM @ECHO OFF

REM Usage: .\docker\build_dockers.bat

REM Update version when changing anything in the Dockerfiles
set VERSION=5

SET SCRIPT_DIR=%~dp0

call "%SCRIPT_DIR%config-docker.bat"  
if %errorlevel% neq 0 call "%SCRIPT_DIR%default-config-docker.bat"

docker build -t %REGISTRY_PREFIX%builder-cuda-11.8-vs17-amd64:v%VERSION% -f docker\Dockerfile.cuda118.amd64.vs17.deps .
docker build -t %REGISTRY_PREFIX%builder-cuda-12.8-vs17-amd64:v%VERSION% -f docker\Dockerfile.cuda128.amd64.vs17.deps .
