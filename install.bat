@echo off
cd /d "%~dp0"

echo.
echo  =============================================
echo   Chain Reaction - Installing Requirements
echo  =============================================
echo.

REM Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found on PATH.
    echo  Download and install Python from https://python.org
    echo  Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  Found: %PYVER%
echo.

REM Upgrade pip silently
echo [1/3] Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Detect Python version to pick correct pygame
for /f "tokens=2 delims=." %%m in ('python -c "import sys; print(sys.version)"') do set MINOR=%%m

echo [2/3] Installing pygame...
if %MINOR% GEQ 14 (
    echo  Python 3.14+ detected - installing pygame-ce
    python -m pip install pygame-ce
) else (
    echo  Installing pygame
    python -m pip install pygame
)

echo.
echo [3/3] Installing AI dependencies (torch + numpy^)...
echo  This may take a few minutes on first install.
echo.
python -m pip install torch numpy

echo.
echo  =============================================
echo   Done! Run the game with:
echo.
echo     python main.py
echo.
echo   Or double-click ChainReaction.pyw
echo  =============================================
echo.
pause
