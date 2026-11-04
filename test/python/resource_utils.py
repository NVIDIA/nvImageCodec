# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import os
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProcessResources:
    rss_bytes: int | None
    handles: int | None


def _windows_process_resources() -> ProcessResources:
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("page_fault_count", wintypes.DWORD),
            ("peak_working_set_size", ctypes.c_size_t),
            ("working_set_size", ctypes.c_size_t),
            ("quota_peak_paged_pool_usage", ctypes.c_size_t),
            ("quota_paged_pool_usage", ctypes.c_size_t),
            ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
            ("quota_non_paged_pool_usage", ctypes.c_size_t),
            ("pagefile_usage", ctypes.c_size_t),
            ("peak_pagefile_usage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetProcessHandleCount.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]

    process = kernel32.GetCurrentProcess()
    handle_count = wintypes.DWORD()
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not kernel32.GetProcessHandleCount(process, ctypes.byref(handle_count)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return ProcessResources(rss_bytes=counters.working_set_size, handles=handle_count.value)


def _proc_process_resources() -> ProcessResources:
    rss_bytes = None
    for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
        if line.startswith("VmRSS:"):
            rss_bytes = int(line.split()[1]) * 1024
            break
    return ProcessResources(rss_bytes=rss_bytes, handles=len(os.listdir("/proc/self/fd")))


def snapshot_process_resources() -> ProcessResources:
    """Return best-effort process memory and handle counts on Windows or Linux."""
    try:
        if os.name == "nt":
            return _windows_process_resources()
        if Path("/proc/self/status").exists():
            return _proc_process_resources()
    except (OSError, ValueError):
        pass
    return ProcessResources(rss_bytes=None, handles=None)


def wait_for_process_resources_to_settle(
        timeout_seconds: float = 10.0,
        stable_seconds: float = 1.0,
        poll_seconds: float = 0.1) -> None:
    """Wait for one-time runtime initialization to stop changing handle counts.

    Windows CUDA/plugin startup can populate a file-association registry cache
    after codec construction returns. Requiring an unchanged count for a full
    interval keeps that unrelated startup work out of repeated-reuse checks.
    """
    deadline = time.monotonic() + timeout_seconds
    previous = snapshot_process_resources()
    stable_since = time.monotonic()
    while time.monotonic() < deadline:
        time.sleep(poll_seconds)
        current = snapshot_process_resources()
        if current.handles == previous.handles:
            if time.monotonic() - stable_since >= stable_seconds:
                return
        else:
            stable_since = time.monotonic()
        previous = current
