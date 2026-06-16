@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   Title Classifier Release Script
echo ============================================================
echo.

set PROJECT_DIR=%~dp0..
set RELEASE_DIR=%PROJECT_DIR%\release
set DIST_DIR=%PROJECT_DIR%\dist\title-classifier
set VERSION=8.1.0
set OUTPUT_NAME=title-classifier-v%VERSION%-win64
set SZIP=D:\scoop\apps\7zip\current\7z.exe
set SFX=D:\scoop\apps\7zip\current\7z.sfx

if not exist "%SZIP%" (
    echo ERROR: 7z.exe not found at %SZIP%
    pause
    exit /b 1
)

if exist "%RELEASE_DIR%" (
    echo Cleaning old release directory...
    rmdir /s /q "%RELEASE_DIR%"
)

mkdir "%RELEASE_DIR%"

echo.
echo [1/4] Running PyInstaller build...
call "%PROJECT_DIR%\scripts\build.bat"
if errorlevel 1 (
    echo ERROR: Build failed
    pause
    exit /b 1
)

if not exist "%DIST_DIR%\title-classifier.exe" (
    echo ERROR: Executable not found after build
    pause
    exit /b 1
)

echo.
echo [2/4] Creating README.txt...
(
echo ============================================================
echo   Title Classifier v%VERSION%
echo ============================================================
echo.
echo Usage:
echo   1. Double-click title-classifier.exe to start
echo   2. First run will auto-initialize database
echo.
echo Models:
echo   - YOLO models included in models/yolo/
echo   - CLIP model: Menu - Tools - Model Manager - Download
echo.
echo Configuration:
echo   - API keys: Menu - Tools - Settings - Advanced API Config
echo   - Keys saved in .env file
echo.
echo FAQ:
echo   Q: Program won't start?
echo   A: Ensure Windows 10+ and Visual C++ Runtime installed
echo.
echo   Q: How to download CLIP model?
echo   A: Menu - Tools - Model Manager - Download CLIP Model
echo.
echo ============================================================
) > "%DIST_DIR%\README.txt"

echo.
echo [3/4] Creating 7z archive...
"%SZIP%" a -t7z -mx5 "%RELEASE_DIR%\%OUTPUT_NAME%.7z" "%DIST_DIR%\*" -r
if errorlevel 1 (
    echo ERROR: 7z compression failed
    pause
    exit /b 1
)

echo.
echo [4/4] Creating self-extracting exe...
copy /b "%SFX%" + "%PROJECT_DIR%\scripts\sfx_config.txt" + "%RELEASE_DIR%\%OUTPUT_NAME%.7z" "%RELEASE_DIR%\%OUTPUT_NAME%.exe"
if errorlevel 1 (
    echo ERROR: SFX creation failed
    pause
    exit /b 1
)

REM Clean up intermediate 7z file
del "%RELEASE_DIR%\%OUTPUT_NAME%.7z"

for %%A in ("%RELEASE_DIR%\%OUTPUT_NAME%.exe") do set EXE_SIZE=%%~zA
set /a EXE_SIZE_MB=!EXE_SIZE! / 1048576

echo.
echo ============================================================
echo   Release Complete!
echo ============================================================
echo.
echo   Output: %RELEASE_DIR%\%OUTPUT_NAME%.exe
echo   Size: !EXE_SIZE_MB! MB
echo.
echo   Distribution:
echo   1. Upload exe to GitHub Releases or cloud storage
echo   2. User downloads and double-clicks to extract
echo   3. Run title-classifier.exe from extracted folder
echo.
echo ============================================================

pause
