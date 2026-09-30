# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DICOM date and time strings that do not match the value representation."""

from datetime import date, time

import pydicom

from nvidia.nvimgcodec.tools.dicom.dicom_utils import fix_dicom_datetime_attributes


def test_odd_length_time_is_removed():
    ds = pydicom.Dataset()
    ds.StudyTime = "12345"
    ds.SeriesTime = "123"
    ds.AcquisitionTime = "12:34:5"
    ds.ContentTime = "123456.1234567"
    fix_dicom_datetime_attributes(ds)
    assert not hasattr(ds, "StudyTime")
    assert not hasattr(ds, "SeriesTime")
    assert not hasattr(ds, "AcquisitionTime")
    assert not hasattr(ds, "ContentTime")


def test_dotted_time_with_two_points_is_removed():
    ds = pydicom.Dataset()
    ds.StudyTime = "12.34.56"
    fix_dicom_datetime_attributes(ds)
    assert not hasattr(ds, "StudyTime")


def test_extra_date_digit_is_removed():
    ds = pydicom.Dataset()
    ds.StudyDate = "202001019"
    fix_dicom_datetime_attributes(ds)
    assert not hasattr(ds, "StudyDate")


def test_valid_date_and_time_stay():
    ds = pydicom.Dataset()
    ds.StudyDate = "20200101"
    ds.StudyTime = "123456"
    ds.SeriesTime = "12:34:56"
    ds.AcquisitionTime = "123456.1"
    ds.ContentTime = "12"
    fix_dicom_datetime_attributes(ds)
    assert ds.StudyDate == date(2020, 1, 1)
    assert ds.StudyTime == time(12, 34, 56)
    assert ds.SeriesTime == time(12, 34, 56)
    assert ds.AcquisitionTime == time(12, 34, 56, 100000)
    assert ds.ContentTime == time(12, 0, 0)
