@echo off
REM ============================================================
REM  Chain Reaction — Thin launcher .exe builder
REM
REM  Produces: ChainReaction.exe  (in the same folder as this bat)
REM
REM  The exe is a THIN STUB — it just finds pythonw.exe and runs
REM  main.py from the same folder. Game source files (.py) stay
REM  live and editable; no rebuild needed after code changes.
REM
REM  Run this ONCE, or any time you want to update the icon.
REM  Requires: Python on PATH  (pip install pyinstaller runs auto)
REM ============================================================

cd /d "%~dp0"

echo.
echo  =============================================
echo   Chain Reaction — Launcher Builder
echo  =============================================
echo.

echo [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found on PATH.
    echo  Install Python from https://python.org and try again.
    pause
    exit /b 1
)
python --version

echo.
echo [2/4] Installing PyInstaller...
pip install pyinstaller --quiet
if errorlevel 1 (
    echo  WARNING: pip install had issues. Continuing...
)

echo.
echo [3/4] Building ChainReaction.exe stub (takes ~30 seconds)...
echo.

REM Build ONLY the tiny launcher stub — not the game code.
REM --onefile   : single exe, no dist/ subfolder mess
REM --windowed  : no console window when the exe runs
REM --distpath  : put the exe right here in the game folder (not dist/)
python -m PyInstaller ^
  --onefile ^
  --windowed ^
  --name "ChainReaction" ^
  --icon "assets\icon.ico" ^
  --distpath "." ^
  --workpath "build_tmp" ^
  launcher_stub.py

if errorlevel 1 (
    echo.
    echo  ERROR: Build failed. See output above.
    pause
    exit /b 1
)

REM Clean up PyInstaller temp files — keep only the exe
if exist build_tmp rmdir /s /q build_tmp
if exist ChainReaction.spec del /q ChainReaction.spec

echo.
echo  =============================================
echo   SUCCESS!
echo   Executable: ChainReaction.exe  (this folder)
echo.
echo   Double-click ChainReaction.exe to play.
echo   Edit any .py file and re-run — no rebuild.
echo  =============================================
echo.
pause
