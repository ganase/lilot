@echo off
setlocal ENABLEDELAYEDEXPANSION

rem ============================================
rem  Lilot Setup Wizard (Portable Installation)
rem ============================================

set "SCRIPT_DIR=%~dp0"
set "TARGET_ROOT=%SCRIPT_DIR%"   rem ← ZIP を展開したフォルダそのもの

echo ============================================
echo  Lilot Setup Wizard
echo ============================================
echo.
echo This wizard will:
echo   - Install Python dependencies via Miniforge (base)
echo   - Create a desktop shortcut for run_app_emb.bat
echo.
pause

rem --------------------------------------------
rem STEP 1: Check Miniforge
rem --------------------------------------------
set "MINIFORGE_ACT=%USERPROFILE%\miniforge3\Scripts\activate.bat"
if not exist "%MINIFORGE_ACT%" (
    echo [ERROR] Miniforge not found at:
    echo         "%MINIFORGE_ACT%"
    echo Please install Miniforge and run setup again.
    pause
    goto :EOF
)

rem --------------------------------------------
rem STEP 2: Install requirements
rem --------------------------------------------
echo.
echo [STEP 2] Installing Python dependencies...

pushd "%TARGET_ROOT%"
call "%MINIFORGE_ACT%" base

if errorlevel 1 (
    echo [ERROR] Failed to activate Miniforge base.
    pause
    popd
    goto :EOF
)

call install_requirements_conda.bat
popd

rem --------------------------------------------
rem STEP 3: Create Desktop Shortcut
rem --------------------------------------------
echo.
echo [STEP 3] Creating desktop shortcut...

set "DESKTOP=%USERPROFILE%\Desktop"
set "SHORTCUT=%DESKTOP%\Lilot.lnk"
set "APP_BAT=%TARGET_ROOT%\run_app_emb.bat"
set "ICON=%TARGET_ROOT%\lilot_mark.ico"

powershell -NoLogo -NoProfile -Command ^
 "$s=(New-Object -COM WScript.Shell).CreateShortcut('%SHORTCUT%');" ^
 "$s.TargetPath='%APP_BAT%';" ^
 "$s.WorkingDirectory='%TARGET_ROOT%';" ^
 "$s.IconLocation='%ICON%';" ^
 "$s.Save();"

echo.
echo ============================================
echo  Setup Completed Successfully!
echo  You can start Lilot from:
echo      Desktop → Lilot
echo ============================================
echo.
pause

endlocal
