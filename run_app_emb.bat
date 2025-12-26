@echo off
setlocal EnableExtensions

rem ============================================
rem  Lilot (Embedding Search) Launcher
rem  - Streamlit app: app\app_emb.py
rem  - Python: Miniforge (base) at %USERPROFILE%\miniforge3
rem ============================================

rem Move to this script directory (project root)
cd /d "%~dp0"

echo ============================================
echo  Lilot (Embedding Search) Starting...
echo  Project: %CD%
echo ============================================
echo.

rem --- Miniforge (Miniconda) root ---
set "CONDA_ROOT=%USERPROFILE%\miniforge3"
set "PYTHON_EXE=%CONDA_ROOT%\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Miniforge Python not found:
    echo         "%PYTHON_EXE%"
    echo.
    echo         Please install Miniforge OR run setup.bat first.
    echo         Miniforge: https://github.com/conda-forge/miniforge
    echo.
    pause
    exit /b 1
)

set "APP_PATH=app\app_emb.py"
if not exist "%APP_PATH%" (
    echo [ERROR] Streamlit app not found:
    echo         "%CD%\%APP_PATH%"
    echo.
    echo         Expected folder structure:
    echo           lilot\
    echo             app\app_emb.py
    echo             data\knowledge.txt
    echo.
    pause
    exit /b 1
)

rem Optional: reduce Streamlit telemetry prompts
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
rem Ensure UTF-8 on Windows console & file IO where possible
set "PYTHONUTF8=1"

echo [INFO] Using Python: "%PYTHON_EXE%"
echo [INFO] Launching: streamlit run "%APP_PATH%"
echo.

"%PYTHON_EXE%" -m streamlit run "%APP_PATH%"
set "ERR=%ERRORLEVEL%"

echo.
echo [INFO] Streamlit exited (code=%ERR%).
echo       Press any key to close.
pause >nul

endlocal
exit /b %ERR%
