@echo off
REM Build script for Title Classifier

echo ============================================================
echo   Title Classifier Build Script
echo ============================================================
echo.

cd /d "%~dp0.."

uv run python scripts/build.py

if errorlevel 1 (
    echo.
    echo ERROR: Build failed
    pause
    exit /b 1
)

echo.
echo Build complete!
