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

from datetime import date, time

import pydicom
import pytest

from nvidia.nvimgcodec.tools.dicom.dicom_utils import fix_dicom_datetime_attributes


@pytest.mark.parametrize(
    "raw_value,expected",
    [
        ("123456", time(12, 34, 56)),
        ("12:34:56", time(12, 34, 56)),
        ("123456.1", time(12, 34, 56, 100000)),
        ("1234", time(12, 34, 0)),
        ("12", time(12, 0, 0)),
        ("12:34", time(12, 34, 0)),
        ("123456.123456", time(12, 34, 56, 123456)),
        ("12:34:56.5", time(12, 34, 56, 500000)),
        # Malformed values must be removed, not guessed at.
        ("12345", None),
        ("123", None),
        ("12:34:5", None),
        ("123456.1234567", None),
        ("12.34.56", None),
        # A fraction is only valid on a full HHMMSS.
        ("1234.5", None),
        # Out-of-range hour/minute/second caught by the time() constructor.
        ("250000", None),
        ("126000", None),
        ("2400", None),
        # Unicode decimal digits must not bypass the ASCII-only check.
        ("１２３４５６", None),
    ],
)
def test_fix_dicom_time_attribute(raw_value, expected):
    ds = pydicom.Dataset()
    ds.StudyTime = raw_value

    fix_dicom_datetime_attributes(ds)

    if expected is None:
        assert not hasattr(ds, "StudyTime")
    else:
        assert ds.StudyTime == expected


@pytest.mark.parametrize(
    "raw_value,expected",
    [
        ("20200101", date(2020, 1, 1)),
        # Extra digits beyond YYYYMMDD must be rejected, not truncated.
        ("202001019", None),
        # Too few digits.
        ("2020010", None),
        # Out-of-range month caught by the date() constructor.
        ("20201301", None),
        # Unicode decimal digits must not bypass the ASCII-only check.
        ("２０２００１０１", None),
    ],
)
def test_fix_dicom_date_attribute(raw_value, expected):
    ds = pydicom.Dataset()
    ds.StudyDate = raw_value

    fix_dicom_datetime_attributes(ds)

    if expected is None:
        assert not hasattr(ds, "StudyDate")
    else:
        assert ds.StudyDate == expected
