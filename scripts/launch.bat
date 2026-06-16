@echo off
REM title-classifier 启动脚本
REM 双击运行此文件启动 title-classifier

echo ========================================
echo   Title Classifier v8.1.0
echo ========================================
echo.

REM 检查可执行文件是否存在
if not exist "%~dp0dist\title-classifier\title-classifier.exe" (
    echo 错误: 未找到可执行文件
    echo 请先运行构建脚本: python scripts/build.py
    pause
    exit /b 1
)

REM 启动程序
echo 正在启动 title-classifier...
start "" "%~dp0dist\title-classifier\title-classifier.exe"

echo 程序已启动
