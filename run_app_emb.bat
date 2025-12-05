@echo off
setlocal

rem ==============================
rem  Locallm embedding launcher
rem ==============================

rem Move to this script directory
cd /d "%~dp0"

echo [INFO] Locallm (embedding) starting ...

rem Miniforge activate.bat (fixed path)
set "MINIFORGE_ACT=%USERPROFILE%\miniforge3\Scripts\activate.bat"

rem ---- Miniforge check (no parentheses version) ----
if exist "%MINIFORGE_ACT%" goto HAVE_MINIFORGE

echo [ERROR] Miniforge not found at:
echo         "%MINIFORGE_ACT%"
echo.
echo Please install Miniforge (miniforge3) and retry.
echo.
pause
goto END

:HAVE_MINIFORGE
echo [INFO] Activating Miniforge base ...
call "%MINIFORGE_ACT%" base

if errorlevel 1 goto ACTIVATE_ERROR

echo [INFO] Launching Streamlit app (embedding version) ...
python -m streamlit run app\app_emb.py

echo.
echo [INFO] Streamlit exited. Press any key to close.
pause
goto END

:ACTIVATE_ERROR
echo [ERROR] Failed to activate Miniforge base.
echo        Try running the following manually:
echo          "%MINIFORGE_ACT%" base
echo.
pause

:END
endlocal
