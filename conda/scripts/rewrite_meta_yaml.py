#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Rewrite recipe/meta.yaml: set version and source.git_url for local or Docker build.
# Also removes any patches listed in patches_to_disable.txt from source.patches.

import os
import re
import sys


def _read_patches_to_disable(filepath):
    """Return set of patch filenames to disable (no path prefix)."""
    out = set()
    if not filepath or not os.path.isfile(filepath):
        return out
    with open(filepath, "r") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                out.add(line)
    return out


def main():
    meta_yaml = os.environ.get("META_YAML")
    detected_version = os.environ.get("DETECTED_VERSION")
    if not meta_yaml:
        print("rewrite_meta_yaml.py: error: META_YAML environment variable is required but not set.", file=sys.stderr)
        sys.exit(1)
    if not detected_version:
        print("rewrite_meta_yaml.py: error: DETECTED_VERSION environment variable is required but not set.", file=sys.stderr)
        sys.exit(1)

    build_local = os.environ.get("CONDA_BUILD_LOCAL", "0") == "1"
    project_root = os.environ.get("PROJECT_ROOT", "")
    patches_to_disable_file = os.environ.get("PATCHES_TO_DISABLE_FILE", "")

    git_url = project_root if build_local else "/home/conda/source_repo"

    with open(meta_yaml, "r") as f:
        content = f.read()

    version_pattern = r'{%\s*set\s+version\s*=\s*"[^"]+"\s*%}'
    version_replacement = f'{{% set version = "{detected_version}" %}}'
    new_content = re.sub(version_pattern, version_replacement, content)
    if new_content == content:
        match = re.search(version_pattern, content)
        msg = "rewrite_meta_yaml.py: error: version substitution did not match; meta.yaml was not modified."
        if match:
            msg += f" Pattern matched: {match.group(0)!r}."
        msg += f" Attempted value: detected_version={detected_version!r}."
        print(msg, file=sys.stderr)
        sys.exit(1)
    content = new_content

    source_pattern = r"(source:\s*\n\s*url:.*?\n\s*sha256:.*?\n)"
    source_replacement = f"source:\n  git_url: {git_url}\n"
    new_content = re.sub(source_pattern, source_replacement, content, flags=re.MULTILINE | re.DOTALL)
    if new_content == content:
        match = re.search(source_pattern, content, flags=re.MULTILINE | re.DOTALL)
        msg = "rewrite_meta_yaml.py: error: source block substitution did not match; meta.yaml was not modified."
        if match:
            msg += f" Pattern matched: {match.group(0)!r}."
        msg += f" Attempted value: git_url={git_url!r}."
        print(msg, file=sys.stderr)
        sys.exit(1)
    content = new_content

    # Remove disabled patches from source.patches so conda-build does not try to apply them
    disabled = _read_patches_to_disable(patches_to_disable_file)
    if disabled:
        new_lines = []
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("- patches/"):
                patch_name = stripped[len("- patches/") :].strip()
                if patch_name in disabled:
                    continue
            new_lines.append(line)
        content = "\n".join(new_lines)

    with open(meta_yaml, "w") as f:
        f.write(content)

    print(f"✓ Modified meta.yaml: version={detected_version}, git_url={git_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
