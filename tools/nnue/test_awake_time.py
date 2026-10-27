"""Suspend-aware budgets without suspending the test machine."""
import ctypes
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import awake_time as A


class AwakeClockTests(unittest.TestCase):
    def test_windows_ticks_are_converted_without_wall_clock(self):
        def query(pointer):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint64))[0] = 123_456_789
            return 1
        with patch.object(A.time, "monotonic", side_effect=AssertionError("wall clock used")):
            self.assertEqual(A._windows_clock(query)(), 12.3456789)

    def test_child_deadline_ignores_simulated_long_suspend(self):
        calls = []
        def wait(timeout):
            calls.append(timeout)
            if len(calls) == 1:
                raise subprocess.TimeoutExpired("fixture", timeout)
            return 0
        child = SimpleNamespace(args=["fixture"], poll=lambda: None, wait=wait)
        with patch.object(A, "awake_seconds", side_effect=[100,100,101]), \
             patch.object(A.time, "monotonic", side_effect=[100,10000]):
            self.assertEqual(A.wait_process(child, 3), 0)
        self.assertEqual(calls, [1,1])

    def test_child_timeout_still_enforces_awake_budget(self):
        child = SimpleNamespace(args=["fixture"], poll=lambda: None)
        with patch.object(A, "awake_seconds", side_effect=[100,104]):
            with self.assertRaises(subprocess.TimeoutExpired):
                A.wait_process(child, 3)

    def test_native_clock_is_available_and_monotonic(self):
        a,b=A.awake_seconds(),A.awake_seconds()
        self.assertGreaterEqual(a, 0)
        self.assertGreaterEqual(b,a)


if __name__ == "__main__":
    unittest.main()
