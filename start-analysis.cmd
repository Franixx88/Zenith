@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-nnue\Scripts\python.exe" (
    echo Python environment missing. See analysis\README.md for setup.
    pause
    exit /b 1
)
if not exist "target\release\zenith.exe" (
    echo Building Zenith...
    cargo build --release --bin zenith
    if errorlevel 1 (
        pause
        exit /b 1
    )
)
".venv-nnue\Scripts\python.exe" "analysis\server.py" --open %*
if errorlevel 1 pause
