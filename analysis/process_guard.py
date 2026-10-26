"""Windows Job Object: a terminated GUI bridge must not orphan a busy engine.

Only the engine process is assigned, never the user's browser or shell.
https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
"""
import os


class EngineJob:
    def __init__(self):
        self.handle = None
        if os.name != "nt":
            return
        import ctypes as c
        from ctypes import wintypes as w

        class Basic(c.Structure):
            _fields_ = [("ProcessTime", c.c_int64), ("JobTime", c.c_int64),
                        ("LimitFlags", w.DWORD), ("MinWorkingSet", c.c_size_t),
                        ("MaxWorkingSet", c.c_size_t), ("ActiveProcesses", w.DWORD),
                        ("Affinity", c.c_size_t), ("Priority", w.DWORD), ("Scheduling", w.DWORD)]

        class Extended(c.Structure):
            _fields_ = [("Basic", Basic), ("IoCounters", c.c_uint64 * 6),
                        ("ProcessMemory", c.c_size_t), ("JobMemory", c.c_size_t),
                        ("PeakProcessMemory", c.c_size_t), ("PeakJobMemory", c.c_size_t)]

        kernel = c.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [c.c_void_p, w.LPCWSTR]
        kernel.CreateJobObjectW.restype = w.HANDLE
        kernel.SetInformationJobObject.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD]
        kernel.SetInformationJobObject.restype = w.BOOL
        kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        kernel.AssignProcessToJobObject.restype = w.BOOL
        kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        kernel.OpenProcess.restype = w.HANDLE
        kernel.CloseHandle.argtypes = [w.HANDLE]
        kernel.CloseHandle.restype = w.BOOL
        self.kernel = kernel
        handle = kernel.CreateJobObjectW(None, None)
        if not handle:
            raise c.WinError(c.get_last_error())
        limits = Extended()
        limits.Basic.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(handle, 9, c.byref(limits), c.sizeof(limits)):
            error = c.WinError(c.get_last_error())
            kernel.CloseHandle(handle)
            raise error
        self.handle = handle

    def assign(self, pid):
        if self.handle is None:
            return
        import ctypes as c
        process = self.kernel.OpenProcess(0x0101, False, pid)  # SET_QUOTA | TERMINATE
        if not process:
            raise c.WinError(c.get_last_error())
        try:
            if not self.kernel.AssignProcessToJobObject(self.handle, process):
                raise c.WinError(c.get_last_error())
        finally:
            self.kernel.CloseHandle(process)

    def close(self):
        if self.handle is not None:
            self.kernel.CloseHandle(self.handle)
            self.handle = None
