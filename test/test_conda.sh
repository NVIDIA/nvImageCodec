#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test conda-built nvimgcodec packages
# Usage: ./test_conda.sh <cuda-major-version>
# Example: ./test_conda.sh 12
#
# Expects the conda tarball at conda/nvimgcodec-conda-packages-cu${CUDA_MAJOR}-*.tar.gz
# (produced by the build step with --tarball flag)

set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Error: CUDA major version not specified"
    echo "Usage: $0 <cuda-major-version>"
    exit 1
fi

CUDA_MAJOR="$1"

# Validate CUDA_MAJOR is a number
if ! [[ "$CUDA_MAJOR" =~ ^[0-9]+$ ]]; then
    echo "Error: CUDA major version must be a number (e.g. 12 or 13), got: $CUDA_MAJOR"
    exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Retry helper for conda commands (handles transient failures)
CONDA_RETRY_COUNT="${CONDA_RETRY_COUNT:-3}"
CONDA_RETRY_DELAY="${CONDA_RETRY_DELAY:-15}"
conda_retry() {
    local attempt=1
    while [ $attempt -le "$CONDA_RETRY_COUNT" ]; do
        local ret=0
        "$@" || ret=$?
        if [ $ret -eq 0 ]; then
            return 0
        fi
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

echo "=========================================="
echo "Conda package test — CUDA ${CUDA_MAJOR}"
echo "=========================================="

# -------------------------------------------------------------------------
# 1. Find and extract the conda tarball
# -------------------------------------------------------------------------
TARBALL_GLOB="nvimgcodec-conda-packages-cu${CUDA_MAJOR}-*.tar.gz"
mapfile -t TARBALL_MATCHES < <(find "${PROJECT_ROOT}/conda" -maxdepth 1 -name "$TARBALL_GLOB" 2>/dev/null | sort)
if [ "${#TARBALL_MATCHES[@]}" -eq 0 ]; then
    echo "Error: No conda tarball found at conda/${TARBALL_GLOB}"
    echo "Contents of conda/:"
    ls -la "${PROJECT_ROOT}/conda/" 2>/dev/null || echo "  (directory not found)"
    exit 1
fi
if [ "${#TARBALL_MATCHES[@]}" -gt 1 ]; then
    echo "Error: Multiple conda tarballs match conda/${TARBALL_GLOB}:"
    printf '  %s\n' "${TARBALL_MATCHES[@]}"
    echo "Remove stale tarballs before running this test."
    exit 1
fi
TARBALL="${TARBALL_MATCHES[0]}"
echo "Found tarball: $TARBALL"

CHANNEL_DIR=$(mktemp -d)
trap 'rm -rf "$CHANNEL_DIR"' EXIT
echo "Extracting to $CHANNEL_DIR ..."
tar -xzf "$TARBALL" -C "$CHANNEL_DIR"

# -------------------------------------------------------------------------
# 2. Index the extracted packages as a local conda channel
# -------------------------------------------------------------------------
echo "Indexing local channel..."
conda_retry python -m conda_index "$CHANNEL_DIR"

# -------------------------------------------------------------------------
# 3. Create a conda environment and install packages
# -------------------------------------------------------------------------
ENV_NAME="nvimgcodec_conda_test"
# Must match conda/build.sh's PYVER — a mismatch makes conda silently pull nvimgcodec
# from conda-forge instead of the local tarball. 3.12 because 3.13 breaks cupy-cuda12x==12.3.0
# (no 3.13 wheel) and torch/pillow/numpy entries in requirements_lnx_cu*.txt.
PYVER="${PYVER:-3.12}"
echo "Creating conda environment (python=${PYVER})..."
conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
conda_retry conda create -y -n "$ENV_NAME" python="$PYVER"

# The feedstock splits CPU codec extensions (libjpeg_turbo, libtiff, opencv) into
# separate packages that are `run_constrained` — not installed by default with
# `nvimgcodec`. Install them explicitly so the test suite exercises CPU backends.
REQUIRED_PKGS=(
    nvimgcodec
    libnvimgcodec-libjpeg-turbo-ext
    libnvimgcodec-libtiff-ext
    libnvimgcodec-libopencv-ext
)

# Fail fast if the tarball is missing any package we're about to install —
# otherwise conda would silently pull it from conda-forge and hide a build regression.
echo "Verifying tarball contains all required packages..."
missing=()
for pkg in "${REQUIRED_PKGS[@]}"; do
    if ! compgen -G "$CHANNEL_DIR/*/${pkg}-*.conda" >/dev/null \
        && ! compgen -G "$CHANNEL_DIR/*/${pkg}-*.tar.bz2" >/dev/null; then
        missing+=("$pkg")
    fi
done
if [ ${#missing[@]} -gt 0 ]; then
    echo "Error: the following packages are not in the tarball:"
    printf '  - %s\n' "${missing[@]}"
    echo "Tarball contents:"
    find "$CHANNEL_DIR" -name '*.conda' -o -name '*.tar.bz2' | sort
    exit 1
fi

echo "Installing nvimgcodec from local channel..."
conda_retry conda install -y -n "$ENV_NAME" -c "file://${CHANNEL_DIR}" -c conda-forge "${REQUIRED_PKGS[@]}"

# Activate the environment
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
set -u

# Assert that every {lib,}nvimgcodec* package resolved to the local channel.
# Catches the case where conda silently picked a conda-forge build even though
# our tarball provides the package (e.g. build-number / variant mismatch).
echo "Verifying {lib,}nvimgcodec* packages came from the local channel..."
conda list -n "$ENV_NAME" --show-channel-urls | awk -v ch="file://${CHANNEL_DIR}" '
    /^(lib)?nvimgcodec/ {
        # column 4 = channel URL; conda prints "<channel>/<subdir>" e.g. "file:///tmp/xxx/linux-64"
        if (index($4, ch) != 1) { print "Non-local:", $0; bad=1 }
    }
    END { exit bad }
' || { echo "Error: some nvimgcodec packages were not installed from the local channel"; exit 1; }

# -------------------------------------------------------------------------
# 4. Install pip test dependencies
# -------------------------------------------------------------------------
REQUIREMENTS="${PROJECT_ROOT}/test/requirements_lnx_cu${CUDA_MAJOR}.txt"
if [ ! -f "$REQUIREMENTS" ]; then
    echo "Error: Requirements file not found: $REQUIREMENTS"
    exit 1
fi
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel
echo "Installing test dependencies from $REQUIREMENTS ..."
pip install -r "$REQUIREMENTS"

# -------------------------------------------------------------------------
# 5. Run tests
# -------------------------------------------------------------------------
cd "$PROJECT_ROOT"

echo ""
echo "=========================================="
echo "Running Python tests (excluding integration)"
echo "=========================================="

python -m pytest -v test/python --ignore=test/python/integration

echo ""
echo "=========================================="
echo "All conda tests passed!"
echo "=========================================="
