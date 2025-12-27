@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================
REM Lilot Setup (Windows)
REM - Creates/updates conda env "lilot"
REM - Installs requirements.txt
REM ============================================

cd /d "%~dp0"

set "ENV_NAME=lilot"
set "PY_VER=3.11"

REM Try to find conda
set "CONDA_EXE="
for %%P in ("%USERPROFILE%\miniforge3\Scripts\conda.exe" ^
            "%USERPROFILE%\miniconda3\Scripts\conda.exe" ^
            "%USERPROFILE%\anaconda3\Scripts\conda.exe") do (
  if exist "%%~P" set "CONDA_EXE=%%~P"
)

if "%CONDA_EXE%"=="" (
  where conda >nul 2>nul
  if %ERRORLEVEL%==0 (
    for /f "delims=" %%C in ('where conda') do (
      set "CONDA_EXE=%%C"
      goto :found_conda
    )
  )
)

:found_conda
if "%CONDA_EXE%"=="" (
  echo [ERROR] conda が見つかりません。
  echo         Miniforge/Miniconda/Anaconda を先にインストールしてください。
  pause
  exit /b 1
)

echo ============================================
echo  Lilot Setup (Windows)
echo  Project: %CD%
echo ============================================
echo [INFO] Using conda: "%CONDA_EXE%"

REM Create env if not exists
"%CONDA_EXE%" env list | findstr /r /c:"^%ENV_NAME% " >nul 2>nul
if %ERRORLEVEL%==0 (
  echo [INFO] Conda env "%ENV_NAME%" already exists.
) else (
  echo [INFO] Creating conda env "%ENV_NAME%" (python=%PY_VER%) ...
  "%CONDA_EXE%" create -n "%ENV_NAME%" python=%PY_VER% -y
  if %ERRORLEVEL% neq 0 (
    echo [ERROR] conda create failed.
    pause
    exit /b 1
  )
)

REM Activate env
call "%CONDA_EXE%" activate "%ENV_NAME%"
if %ERRORLEVEL% neq 0 (
  echo [ERROR] conda activate failed.
  pause
  exit /b 1
)

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

echo [INFO] Installing requirements...
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
  echo [ERROR] pip install failed.
  pause
  exit /b 1
)

echo.
echo ✅ Setup complete.
echo 次は run_win.bat（または streamlit run app\app_emb.py）で起動できます。
pause
endlocal
