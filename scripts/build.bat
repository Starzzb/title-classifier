@echo off
REM ============================================================
REM Title Classifier 打包脚本
REM 功能：使用 PyInstaller 打包为可执行文件
REM ============================================================

echo ============================================================
echo   Title Classifier 打包脚本
echo ============================================================
echo.

cd /d "%~dp0.."

REM 运行 Python 打包脚本
uv run python scripts/build.py

if errorlevel 1 (
    echo.
    echo 错误: 打包失败
    pause
    exit /b 1
)

echo.
echo 打包完成!
