#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Script to build nvimgcodec conda package using local source code
#
# Usage:
#   ./build.sh [options] [feedstock_type] [build-locally.py args...]
#   ./build.sh --setup-env                    # setup only, then exit
#   ./build.sh --setup-env --no-docker lib    # setup then build (pre-step)
#
# Options:
#   --no-docker, --local  Run conda build in the current environment instead of
#                         Docker (e.g. for Jenkins in a Miniconda container).
#                         Can also be enabled with CONDA_BUILD_LOCAL=1.
#   --output-dir <dir>    Write built packages to <dir>. Can also be set via
#                         CONDA_BUILD_OUTPUT_DIR. For "all", both lib and python
#                         outputs go to the same directory.
#   --tarball             After a successful build, create a single tarball with all
#                         packages (requires or uses --output-dir).
#   --cuda-version <ver>  CUDA compiler version for the build (e.g. 12.9, 13.0).
#                         Can also be set via CONDA_CUDA_COMPILER_VERSION. Default: 12.9.
#   --setup-env           Prepare conda environment for local builds: add conda-forge
#                         channel, install conda-build and boa. Can be used alone (then
#                         exits) or as a pre-step with any other command (e.g. --setup-env
#                         --no-docker lib).
#
# feedstock_type can be:
#   - "python" or "nvimgcodec" (default) - builds Python package from nvimgcodec-feedstock
#   - "lib" or "libnvimgcodec" - builds C library packages from libnvimgcodec-feedstock
#   - "all" or "both" - builds lib first, then python

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONDA_SCRIPTS_DIR="${SCRIPT_DIR}/scripts"

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
CONDA_BUILD_LOCAL="${CONDA_BUILD_LOCAL:-0}"
CONDA_BUILD_OUTPUT_DIR="${CONDA_BUILD_OUTPUT_DIR:-}"
CREATE_TARBALL=0
SETUP_ENV=0

while [ $# -gt 0 ]; do
    case "$1" in
        --no-docker|--local)
            CONDA_BUILD_LOCAL=1
            export CONDA_BUILD_LOCAL
            shift
            ;;
        --output-dir|-o)
            [ -z "${2:-}" ] && { echo "Error: --output-dir requires a directory argument"; exit 1; }
            CONDA_BUILD_OUTPUT_DIR="$2"
            export CONDA_BUILD_OUTPUT_DIR
            shift 2
            ;;
        --tarball)
            CREATE_TARBALL=1
            shift
            ;;
        --cuda-version)
            [ -z "${2:-}" ] && { echo "Error: --cuda-version requires a version argument (e.g. 12.9, 13.0)"; exit 1; }
            CONDA_CUDA_COMPILER_VERSION="$2"
            export CONDA_CUDA_COMPILER_VERSION
            shift 2
            ;;
        --setup-env)
            SETUP_ENV=1
            shift
            ;;
        *)
            break
            ;;
    esac
done

CUDA_VER="${CONDA_CUDA_COMPILER_VERSION:-12.9}"
export CUDA_VER

# -----------------------------------------------------------------------------
# Retry a command (for transient network failures, e.g. CondaHTTPError HTTP 000)
# Usage: conda_retry conda install -y -c conda-forge conda-build
# -----------------------------------------------------------------------------
CONDA_RETRY_COUNT="${CONDA_RETRY_COUNT:-3}"
CONDA_RETRY_DELAY="${CONDA_RETRY_DELAY:-15}"
conda_retry() {
    local attempt=1
    while [ $attempt -le "$CONDA_RETRY_COUNT" ]; do
        if "$@"; then
            return 0
        fi
        local ret=$?
        if [ $attempt -lt "$CONDA_RETRY_COUNT" ]; then
            echo "Attempt $attempt/$CONDA_RETRY_COUNT failed (exit $ret). Retrying in ${CONDA_RETRY_DELAY}s..."
            sleep "$CONDA_RETRY_DELAY"
            attempt=$((attempt + 1))
        else
            echo "All $CONDA_RETRY_COUNT attempts failed."
            return $ret
        fi
    done
}

# -----------------------------------------------------------------------------
# Setup conda environment (optional pre-step)
# -----------------------------------------------------------------------------
do_setup_env() {
    [ "$SETUP_ENV" != "1" ] && return 0

    echo "=========================================="
    echo "Preparing conda environment for local builds"
    echo "=========================================="
    if ! command -v conda &>/dev/null; then
        echo "Error: conda not found. Install Miniconda or Anaconda and run this again."
        exit 1
    fi
    echo "Adding conda-forge channel..."
    conda config --add channels conda-forge 2>/dev/null || true
    conda config --set channel_priority flexible 2>/dev/null || true
    echo "Installing conda-build (required for local builds)..."
    conda_retry conda install -y -c conda-forge conda-build
    echo ""
    echo "=========================================="
    echo "Setup complete."
    echo "=========================================="
    if [ $# -eq 0 ]; then
        exit 0
    fi
    echo "Proceeding with build..."
    echo ""
}

# -----------------------------------------------------------------------------
# Build "all" (lib then python, optional tarball)
# -----------------------------------------------------------------------------
build_all() {
    [ "$CREATE_TARBALL" = "1" ] && [ -z "$CONDA_BUILD_OUTPUT_DIR" ] && {
        CONDA_BUILD_OUTPUT_DIR="${SCRIPT_DIR}/.conda_build_output"
        export CONDA_BUILD_OUTPUT_DIR
    }
    RECURSE_OPTS=()
    [ "$CONDA_BUILD_LOCAL" = "1" ] && RECURSE_OPTS+=(--no-docker)
    [ -n "$CONDA_BUILD_OUTPUT_DIR" ] && RECURSE_OPTS+=(--output-dir "$CONDA_BUILD_OUTPUT_DIR")
    RECURSE_OPTS+=(--cuda-version "$CUDA_VER")

    echo "=========================================="
    echo "Building BOTH packages (lib + python)"
    echo "=========================================="
    [ -n "$CONDA_BUILD_OUTPUT_DIR" ] && echo "Output directory: $CONDA_BUILD_OUTPUT_DIR"
    echo "CUDA version: $CUDA_VER"
    echo ""

    echo "Step 1/2: Building C library packages..."
    echo "=========================================="
    "$0" "${RECURSE_OPTS[@]}" lib
    echo ""
    echo "Step 2/2: Building Python package..."
    echo "=========================================="
    "$0" "${RECURSE_OPTS[@]}" python
    echo ""

    if [ "$CREATE_TARBALL" = "1" ] && [ -n "$CONDA_BUILD_OUTPUT_DIR" ] && [ -d "$CONDA_BUILD_OUTPUT_DIR" ]; then
        TARBALL_NAME="nvimgcodec-conda-packages-$(date +%Y%m%d-%H%M%S).tar.gz"
        TARBALL_PATH="${SCRIPT_DIR}/${TARBALL_NAME}"
        echo "Creating tarball: $TARBALL_PATH"
        tar -czvf "$TARBALL_PATH" -C "$CONDA_BUILD_OUTPUT_DIR" .
        echo "Tarball created: $TARBALL_PATH"
    fi
    echo "=========================================="
    echo "All builds completed!"
    echo "=========================================="
    exit 0
}

# -----------------------------------------------------------------------------
# Resolve feedstock type -> repo and name
# -----------------------------------------------------------------------------
FEEDSTOCK_TYPE="${1:-python}"
case "$FEEDSTOCK_TYPE" in
    all|both)
        do_setup_env "$@"
        build_all
        ;;
    python|nvimgcodec)
        FEEDSTOCK_REPO="https://github.com/conda-forge/nvimgcodec-feedstock.git"
        FEEDSTOCK_NAME="nvimgcodec-feedstock"
        ;;
    lib|libnvimgcodec)
        FEEDSTOCK_REPO="https://github.com/conda-forge/libnvimgcodec-feedstock.git"
        FEEDSTOCK_NAME="libnvimgcodec-feedstock"
        ;;
    *)
        echo "Error: Unknown feedstock type '$FEEDSTOCK_TYPE'"
        echo "Usage: $0 [--setup-env] [--no-docker|--local] [--output-dir <dir>] [--tarball] [--cuda-version <ver>] [python|lib|all]"
        exit 1
        ;;
esac

do_setup_env "$@"

WORK_DIR="${SCRIPT_DIR}/.${FEEDSTOCK_NAME}_clone"

# Default output dir when --tarball is used but --output-dir was not set
[ "$CREATE_TARBALL" = "1" ] && [ -z "$CONDA_BUILD_OUTPUT_DIR" ] && {
    CONDA_BUILD_OUTPUT_DIR="${WORK_DIR}/build_artifacts"
    export CONDA_BUILD_OUTPUT_DIR
}

# -----------------------------------------------------------------------------
# Prepare work dir: clone feedstock, apply local patches
# -----------------------------------------------------------------------------
echo "=========================================="
echo "Building conda package: $FEEDSTOCK_NAME"
echo "=========================================="
echo "Feedstock: $FEEDSTOCK_REPO"
echo "Project root: $PROJECT_ROOT"
echo "Work directory: $WORK_DIR"
echo "CUDA version: $CUDA_VER"
[ -n "$CONDA_BUILD_OUTPUT_DIR" ] && echo "Output directory: $CONDA_BUILD_OUTPUT_DIR"
echo ""

if [ -d "$WORK_DIR" ]; then
    echo "Cleaning up previous work directory..."
    rm -rf "$WORK_DIR"
fi
echo "Cloning feedstock from $FEEDSTOCK_REPO..."
conda_retry git clone "$FEEDSTOCK_REPO" "$WORK_DIR"

# Remove feedstock patches listed in patches_to_disable.txt (so they are not applied)
PATCHES_TO_DISABLE_FILE="${SCRIPT_DIR}/patches_to_disable.txt"
RECIPE_PATCHES_DIR="${WORK_DIR}/recipe/patches"
if [ -f "$PATCHES_TO_DISABLE_FILE" ] && [ -d "$RECIPE_PATCHES_DIR" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%%#*}"
        line="$(echo "$line" | tr -d '[:space:]')"
        [ -z "$line" ] && continue
        if [ -f "$RECIPE_PATCHES_DIR/$line" ]; then
            echo "Disabling feedstock patch: $line"
            rm -f "$RECIPE_PATCHES_DIR/$line"
        fi
    done < "$PATCHES_TO_DISABLE_FILE"
fi

CONDA_PATCHES_DIR="${SCRIPT_DIR}/patches"
if [ -d "$CONDA_PATCHES_DIR" ] && [ -n "$(ls -A "$CONDA_PATCHES_DIR" 2>/dev/null)" ]; then
    echo "Applying local patches from $CONDA_PATCHES_DIR over recipe/patches..."
    cp -v "$CONDA_PATCHES_DIR"/*.patch "$RECIPE_PATCHES_DIR/" 2>/dev/null || true
fi

# -----------------------------------------------------------------------------
# Detect version and rewrite meta.yaml
# -----------------------------------------------------------------------------
echo "Detecting version from CMakeLists.txt..."
DETECTED_VERSION=$(grep 'set(NVIMGCODEC_VERSION "' "$PROJECT_ROOT/CMakeLists.txt" | head -1 | sed -E 's/.*"([0-9]+\.[0-9]+\.[0-9]+)".*/\1/')
[ -z "$DETECTED_VERSION" ] && { echo "Error: Could not detect version from CMakeLists.txt"; exit 1; }
echo "Detected version: $DETECTED_VERSION"
echo ""

META_YAML="${WORK_DIR}/recipe/meta.yaml"
cp "$META_YAML" "${META_YAML}.orig"
export DETECTED_VERSION META_YAML CONDA_BUILD_LOCAL PROJECT_ROOT PATCHES_TO_DISABLE_FILE
python3 "${CONDA_SCRIPTS_DIR}/rewrite_meta_yaml.py"

echo ""
echo "Modified meta.yaml diff (version and source):"
diff -u "${META_YAML}.orig" "$META_YAML" || true
echo ""

# -----------------------------------------------------------------------------
# For Python build: use local lib packages as channel if available
# -----------------------------------------------------------------------------
USE_LOCAL_CHANNEL=false
LOCAL_CHANNEL_DIR=""
LIB_FEEDSTOCK_WORK_DIR="${SCRIPT_DIR}/.libnvimgcodec-feedstock_clone"

is_python_feedstock() {
    [[ "$FEEDSTOCK_NAME" == *"nvimgcodec-feedstock"* ]] && [[ "$FEEDSTOCK_NAME" != *"libnvimgcodec"* ]]
}

if is_python_feedstock; then
    if [ -n "$CONDA_BUILD_OUTPUT_DIR" ] && [ -d "$CONDA_BUILD_OUTPUT_DIR" ] && [ "$(ls -A "$CONDA_BUILD_OUTPUT_DIR"/linux-64/*.conda 2>/dev/null)" ]; then
        LOCAL_CHANNEL_DIR="$CONDA_BUILD_OUTPUT_DIR"
        USE_LOCAL_CHANNEL=true
    elif [ -d "$LIB_FEEDSTOCK_WORK_DIR/build_artifacts" ] && [ "$(ls -A "$LIB_FEEDSTOCK_WORK_DIR/build_artifacts/linux-64/"*.conda 2>/dev/null)" ]; then
        echo "Found local libnvimgcodec packages, will copy them to feedstock build_artifacts"
        echo ""
        FEEDSTOCK_ARTIFACTS="${WORK_DIR}/build_artifacts"
        mkdir -p "${FEEDSTOCK_ARTIFACTS}/linux-64" "${FEEDSTOCK_ARTIFACTS}/noarch"
        echo "Copying packages..."
        cp -v "$LIB_FEEDSTOCK_WORK_DIR/build_artifacts/linux-64/"*.conda "${FEEDSTOCK_ARTIFACTS}/linux-64/" 2>/dev/null || true
        cp -v "$LIB_FEEDSTOCK_WORK_DIR/build_artifacts/noarch/"*.conda "${FEEDSTOCK_ARTIFACTS}/noarch/" 2>/dev/null || true
        echo "Indexing feedstock build_artifacts..."
        if command -v conda &>/dev/null; then
            conda index "${FEEDSTOCK_ARTIFACTS}"
        elif [ -f ~/miniconda3/bin/conda ]; then
            ~/miniconda3/bin/conda index "${FEEDSTOCK_ARTIFACTS}"
        fi
        echo "Local libnvimgcodec packages are now available in the build"
        LOCAL_CHANNEL_DIR="${FEEDSTOCK_ARTIFACTS}"
        USE_LOCAL_CHANNEL=true
    fi
fi

# -----------------------------------------------------------------------------
# Run conda build (local or Docker)
# -----------------------------------------------------------------------------
echo "=========================================="
echo "Starting conda build..."
echo "=========================================="
cd "$WORK_DIR"
export CI=true

if [ "$CONDA_BUILD_LOCAL" = "1" ]; then
    # Git safe.directory for when repo path differs (e.g. container vs host)
    GIT_SAFE_CONFIG=""
    if [ -d "${PROJECT_ROOT}/.git" ] || [ -f "${PROJECT_ROOT}/.git" ]; then
        GIT_SAFE_CONFIG=$(mktemp)
        printf '[safe]\n\tdirectory = %s\n\tdirectory = %s/.git\n' "$PROJECT_ROOT" "$PROJECT_ROOT" >>"$GIT_SAFE_CONFIG"
        export GIT_CONFIG_GLOBAL="$GIT_SAFE_CONFIG"
        if command -v git &>/dev/null; then
            git config --global --add safe.directory "$PROJECT_ROOT" 2>/dev/null || true
            git config --global --add safe.directory "$PROJECT_ROOT/.git" 2>/dev/null || true
        elif [ -x /usr/bin/git ]; then
            /usr/bin/git config --global --add safe.directory "$PROJECT_ROOT" 2>/dev/null || true
            /usr/bin/git config --global --add safe.directory "$PROJECT_ROOT/.git" 2>/dev/null || true
        fi
    fi

    is_python_feedstock && [ "$USE_LOCAL_CHANNEL" != "true" ] && {
        echo "Note: Build lib first so the solver can find libnvimgcodec-dev: ./build.sh lib  (or use ./build.sh all)"
        echo ""
    }

    PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
    echo "Using Python: ${PYVER:-unknown} (variant config will pin this version)."
    echo ""

    command -v conda &>/dev/null || { echo "Error: conda not found. Activate a conda environment and retry."; exit 1; }

    [ "$USE_LOCAL_CHANNEL" = "true" ] && [ -n "$LOCAL_CHANNEL_DIR" ] && [ "$LOCAL_CHANNEL_DIR" = "$CONDA_BUILD_OUTPUT_DIR" ] && {
        echo "Indexing local channel: $LOCAL_CHANNEL_DIR"
        conda index "$LOCAL_CHANNEL_DIR"
    }

    CHANNEL_ARGS="-c conda-forge"
    [ "$USE_LOCAL_CHANNEL" = "true" ] && [ -n "$LOCAL_CHANNEL_DIR" ] && [ -d "$LOCAL_CHANNEL_DIR" ] && {
        CHANNEL_ARGS="-c file://${LOCAL_CHANNEL_DIR} $CHANNEL_ARGS"
        echo "Using local channel: file://${LOCAL_CHANNEL_DIR}"
    }

    LOCAL_VARIANT_CONFIG=$(mktemp)
    trap "rm -f '$LOCAL_VARIANT_CONFIG'${GIT_SAFE_CONFIG:+ '$GIT_SAFE_CONFIG'}" EXIT
    cat >"$LOCAL_VARIANT_CONFIG" <<EOF
# Generated for local build: overrides internal_defaults (compiler, cuda, python).
c_compiler:
  - gcc
c_compiler_version:
  - "14"
cxx_compiler:
  - gxx
cxx_compiler_version:
  - "14"
c_stdlib:
  - sysroot
c_stdlib_version:
  - "2.17"
target_platform:
  - linux-64
cuda_compiler:
  - cuda-nvcc
cuda_compiler_version:
  - "${CUDA_VER}"
python:
  - "${PYVER}"
EOF

    VARIANT_ARGS="-m $LOCAL_VARIANT_CONFIG"
    echo "Using variant config (cuda=${CUDA_VER}, python=${PYVER}, c_compiler=gcc)."
    PY_ARGS=""
    is_python_feedstock && PY_ARGS="--python ${PYVER}"
    OUTPUT_FOLDER_ARGS=""
    [ -n "$CONDA_BUILD_OUTPUT_DIR" ] && { mkdir -p "$CONDA_BUILD_OUTPUT_DIR"; OUTPUT_FOLDER_ARGS="--output-folder $CONDA_BUILD_OUTPUT_DIR"; }

    RUN_BUILD=""
    conda mambabuild --help &>/dev/null && RUN_BUILD="conda mambabuild ${CHANNEL_ARGS} ${OUTPUT_FOLDER_ARGS} ${VARIANT_ARGS} ${PY_ARGS} recipe"
    [ -z "$RUN_BUILD" ] && conda build --help &>/dev/null && RUN_BUILD="conda build ${CHANNEL_ARGS} ${OUTPUT_FOLDER_ARGS} ${VARIANT_ARGS} ${PY_ARGS} recipe"
    [ -z "$RUN_BUILD" ] && python3 -c "import conda_build" 2>/dev/null && RUN_BUILD="python3 -m conda_build ${CHANNEL_ARGS} ${OUTPUT_FOLDER_ARGS} ${VARIANT_ARGS} ${PY_ARGS} recipe"

    if [ -n "$RUN_BUILD" ]; then
        $RUN_BUILD
    else
        echo "Error: conda-build is required for local builds. Install: conda install -c conda-forge conda-build"
        exit 1
    fi
else
    export CONDA_FORGE_DOCKER_RUN_ARGS="-v ${PROJECT_ROOT}:/home/conda/source_repo:ro,z,delegated"
    echo "Mounting source repository: ${PROJECT_ROOT} -> /home/conda/source_repo"
    echo ""

    [ ! -f "build-locally.py" ] && { echo "Error: build-locally.py not found in feedstock"; exit 1; }

    if [ "${1:-}" = "python" ] || [ "${1:-}" = "nvimgcodec" ] || [ "${1:-}" = "lib" ] || [ "${1:-}" = "libnvimgcodec" ]; then
        shift
    fi

    if [ $# -eq 0 ]; then
        if [[ "$FEEDSTOCK_NAME" == *"libnvimgcodec"* ]]; then
            CONFIG="linux_64_c_stdlib_version2.17cuda_compiler_version${CUDA_VER}"
        else
            CONFIG="linux_64_c_stdlib_version2.17cuda_compiler_version${CUDA_VER}python3.10.____cpython"
        fi
        echo "No config specified, using default (CUDA ${CUDA_VER}): $CONFIG"
        echo ""
        python build-locally.py "$CONFIG"
    else
        python build-locally.py "$@"
    fi

    [ -n "$CONDA_BUILD_OUTPUT_DIR" ] && [ -d "build_artifacts" ] && {
        echo "Copying build output to $CONDA_BUILD_OUTPUT_DIR"
        mkdir -p "$CONDA_BUILD_OUTPUT_DIR"
        cp -r build_artifacts/. "$CONDA_BUILD_OUTPUT_DIR"/
    }
fi

# -----------------------------------------------------------------------------
# Optional tarball (single feedstock build)
# -----------------------------------------------------------------------------
[ "$CREATE_TARBALL" = "1" ] && [ -n "$CONDA_BUILD_OUTPUT_DIR" ] && [ -d "$CONDA_BUILD_OUTPUT_DIR" ] && {
    TARBALL_NAME="nvimgcodec-conda-packages-$(date +%Y%m%d-%H%M%S).tar.gz"
    TARBALL_PATH="${SCRIPT_DIR}/${TARBALL_NAME}"
    echo "Creating tarball: $TARBALL_PATH"
    tar -czvf "$TARBALL_PATH" -C "$CONDA_BUILD_OUTPUT_DIR" .
    echo "Tarball created: $TARBALL_PATH"
}

echo ""
echo "=========================================="
echo "Build completed!"
echo "=========================================="
