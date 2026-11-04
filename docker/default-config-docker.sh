export DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-1}
export REGISTRY_PREFIX=${REGISTRY_PREFIX:-""}
export PLATFORM=${PLATFORM:-"linux/amd64"}  # or "linux/arm64"
export ARCH=${ARCH:-"x86_64"}  # or "aarch64"
export VER_UBUNTU=${VER_UBUNTU:-"24.04"}
# LOAD-BEARING for Python 3.9: pypa removes EOL interpreters from newer images, and
# this release still ships cp39 although 3.9 reached EOL on 2025-10-31. Do not bump
# past the release that drops cp39 without also dropping 3.9 from cmake/Utils.cmake
# PYTHON_VERSIONS. Dockerfile.deps asserts the full set exists, so a bad bump fails
# the image build.
export MANYLINUX_IMAGE_TAG="2026.10.10-1"
