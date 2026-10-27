"""Elapsed system-awake time for finite jobs; never changes power settings.

Windows time.monotonic includes suspend, whereas QueryUnbiasedInterruptTime
excludes sleep/hibernation (100 ns ticks):
https://learn.microsoft.com/windows/win32/api/realtimeapiset/nf-realtimeapiset-queryunbiasedinterrupttime
"""
import ctypes
import os
import subprocess
import time


def _windows_clock(query):
    def now():
        ticks = ctypes.c_uint64()
        if not query(ctypes.byref(ticks)):
            raise ctypes.WinError(ctypes.get_last_error())
        return ticks.value / 10_000_000
    return now


if os.name == "nt":
    _query = ctypes.WinDLL("kernel32", use_last_error=True).QueryUnbiasedInterruptTime
    _query.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    _query.restype = ctypes.c_int
    awake_seconds = _windows_clock(_query)
else:
    awake_seconds = time.monotonic


def wait_process(process, timeout):
    """Bound a child by awake seconds, retaining wall time in separate provenance."""
    deadline = awake_seconds() + timeout
    while process.poll() is None:
        left = deadline - awake_seconds()
        if left <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        try:
            return process.wait(timeout=min(1.0, left))
        except subprocess.TimeoutExpired:
            pass
    return process.returncode
