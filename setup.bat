@echo off
setlocal ENABLEDELAYEDEXPANSION

rem ============================================
rem  Locallm setup wizard
rem ============================================

set "SCRIPT_DIR=%~dp0"
rem コピー先は固定パス C:\TMP\Locallm
set "TARGET_ROOT=C:\TMP\Locallm"

echo ============================================
echo  Locallm setup wizard
echo ============================================
echo.
echo This wizard will:
echo   - Copy Locallm to "%TARGET_ROOT%"
echo   - Install Python dependencies via Miniforge (base)
echo   - Create desktop shortcuts for:
echo       * run_app_emb.bat  (embedding search)
echo.
pause

rem C:\TMP が無ければ作成
if not exist "C:\TMP\" (
    mkdir "C:\TMP"
)

echo.
echo [STEP 1] Copy files to "%TARGET_ROOT%" ...

if exist "%TARGET_ROOT%\" (
    echo Target folder already exists.
    choice /C YN /M "Overwrite existing Locallm folder? (Y/N)"
    if errorlevel 2 (
        echo -> Keeping existing folder. Skipping copy.
    ) else (
        echo -> Removing old "%TARGET_ROOT%" ...
        rmdir /S /Q "%TARGET_ROOT%"
        echo -> Copying from "%SCRIPT_DIR%" ...
        rem 自分自身 (setup.bat) は除外してコピー
        robocopy "%SCRIPT_DIR%" "%TARGET_ROOT%" /E /NFL /NDL /NP /NJH /NJS /XF setup.bat
    )
) else (
    echo -> Copying from "%SCRIPT_DIR%" ...
    robocopy "%SCRIPT_DIR%" "%TARGET_ROOT%" /E /NFL /NDL /NP /NJH /NJS /XF setup.bat
)

echo.
echo [STEP 2] Install Python dependencies with Miniforge ...

set "MINIFORGE_ACT=%USERPROFILE%\miniforge3\Scripts\activate.bat"
if not exist "%MINIFORGE_ACT%" (
    echo [ERROR] Miniforge not found at:
    echo         "%MINIFORGE_ACT%"
    echo Please install Miniforge (miniforge3) under:
    echo         "%USERPROFILE%\miniforge3"
    echo and then run setup.bat again.
    echo.
    pause
    goto :EOF
)

pushd "%TARGET_ROOT%"
echo -> Activating Miniforge base ...
call "%MINIFORGE_ACT%" base

if errorlevel 1 (
    echo [ERROR] Failed to activate Miniforge base.
    echo        Try running this manually in a terminal:
    echo          "%MINIFORGE_ACT%" base
    echo.
    pause
    popd
    goto :EOF
)

echo -> Running install_requirements_conda.bat ...
call install_requirements_conda.bat
popd

echo.
echo [STEP 3] Create desktop shortcuts ...

set "DESKTOP=%USERPROFILE%\Desktop"

powershell -NoLogo -NoProfile -Command ^
 "$WshShell = New-Object -ComObject WScript.Shell; " ^
 "$desktop  = [Environment]::GetFolderPath('Desktop'); " ^
 "$icon     = [System.IO.Path]::Combine($env:SystemRoot, 'System32\\shell32.dll'); " ^
 "$s2 = Join-Path $desktop 'Locallm (embedding).lnk'; " ^
 "$c2 = $WshShell.CreateShortcut($s2); " ^
 "$c2.TargetPath      = 'C:\\TMP\\Locallm\\run_app_emb.bat'; " ^
 "$c2.WorkingDirectory= 'C:\\TMP\\Locallm'; " ^
 "$c2.IconLocation    = \"$icon,43\"; " ^
 "$c2.Save(); "

echo.
echo ============================================
echo  Setup finished.
echo  You can start Locallm from the desktop:
echo    - Locallm (keyword)
echo    - Locallm (embedding)
echo ============================================
echo.
pause

endlocal
