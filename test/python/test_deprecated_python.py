# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Coverage for the deprecated-interpreter notice emitted at import time.

The warning only fires on Python 3.9, which the test lanes no longer run, so
the checks drive ``_warn_if_deprecated_python`` with an explicit version tuple
instead of relying on the interpreter under test.
"""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest

from nvidia.nvimgcodec import (
    NvImageCodecDeprecatedPythonWarning,
    _warn_if_deprecated_python,
)


def test_python_39_is_reported_as_deprecated():
    with pytest.warns(NvImageCodecDeprecatedPythonWarning) as record:
        _warn_if_deprecated_python((3, 9, 21, "final", 0))
    assert len(record) == 1
    message = str(record[0].message)
    assert "3.9" in message
    assert "3.10" in message


@pytest.mark.parametrize(
    "version_info",
    [
        (3, 10, 0, "final", 0),
        (3, 13, 0, "final", 0),
        (3, 14, 0, "final", 0),
    ],
)
def test_supported_versions_are_silent(version_info):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_deprecated_python(version_info)


def test_import_emits_the_warning_on_python_39(monkeypatch):
    """Cover the module-level call, not just the helper.

    Patching ``sys.version_info`` is not enough on its own: the helper binds it as
    a default argument, so the value is captured when the ``def`` executes. Reloading
    the package re-executes the ``def`` *and* the module-level call, which is exactly
    the code path a real 3.9 user hits.
    """
    import nvidia.nvimgcodec as pkg

    monkeypatch.setattr(sys, "version_info", (3, 9, 21, "final", 0))
    try:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            importlib.reload(pkg)
        # Match on the class NAME, not the class object: reloading rebinds
        # NvImageCodecDeprecatedPythonWarning to a fresh type, so the one imported at
        # the top of this file is no longer the one the reloaded module raises and an
        # isinstance-based check (pytest.warns) would miss it.
        assert [
            w for w in record
            if type(w.message).__name__ == NvImageCodecDeprecatedPythonWarning.__name__
            and "3.9" in str(w.message)
        ]
    finally:
        # Restore the module against the real interpreter, or the rebound default
        # would leak into every later test in this session.
        monkeypatch.undo()
        importlib.reload(pkg)


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the running interpreter is itself deprecated"
)
def test_the_running_interpreter_is_silent():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", NvImageCodecDeprecatedPythonWarning)
        _warn_if_deprecated_python()
    assert record == []


def test_deprecation_warning_is_filterable_as_a_user_warning():
    assert issubclass(NvImageCodecDeprecatedPythonWarning, UserWarning)
