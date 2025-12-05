@echo off
setlocal

rem === Move to this script directory ===
cd /d "%~dp0"

echo [INFO] install_requirements_conda.bat started
echo   Current directory: %CD%
echo.

rem === Miniforge Python path ===
set "CONDA_ROOT=%USERPROFILE%\miniforge3"
set "PYTHON_EXE=%CONDA_ROOT%\python.exe"

echo [INFO] Looking for Miniforge Python at:
echo        "%PYTHON_EXE%"
echo.

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Miniforge Python not found.
    echo Please install Miniforge to "%CONDA_ROOT%" first.
    pause
    exit /b 1
)

if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found in:
    echo        %CD%
    pause
    exit /b 1
)

echo [INFO] Upgrading pip...
"%PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip.
    pause
    exit /b 1
)

echo [INFO] Installing packages from requirements.txt...
"%PYTHON_EXE%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed.
    pause
    exit /b 1
)

echo.
echo [INFO] All dependencies installed successfully.
echo [INFO] You can now run "run_app_emb.bat".
echo.
pause

endlocal
