@echo off
REM ============================================================
REM Title Classifier 发布脚本
REM 功能：打包并压缩为 zip 文件用于分发
REM ============================================================

setlocal enabledelayedexpansion

echo ============================================================
echo   Title Classifier 发布脚本
echo ============================================================
echo.

REM 设置变量
set PROJECT_DIR=%~dp0..
set RELEASE_DIR=%PROJECT_DIR%\release
set DIST_DIR=%PROJECT_DIR%\dist\title-classifier
set VERSION=8.1.0
set OUTPUT_NAME=title-classifier-v%VERSION%-win64

REM 清理旧的 release 目录
if exist "%RELEASE_DIR%" (
    echo 清理旧的 release 目录...
    rmdir /s /q "%RELEASE_DIR%"
)

REM 创建 release 目录
mkdir "%RELEASE_DIR%"

REM 运行打包脚本
echo.
echo [1/3] 运行 PyInstaller 打包...
call "%PROJECT_DIR%\scripts\build.bat"
if errorlevel 1 (
    echo 错误: 打包失败
    pause
    exit /b 1
)

REM 检查打包结果
if not exist "%DIST_DIR%\title-classifier.exe" (
    echo 错误: 未找到打包后的可执行文件
    pause
    exit /b 1
)

REM 创建 README.txt
echo.
echo [2/3] 创建 README.txt...
(
echo ============================================================
echo   Title Classifier v%VERSION%
echo   视频标题分类和重命名工具
echo ============================================================
echo.
echo 使用说明：
echo -----------
echo 1. 双击 title-classifier.exe 启动程序
echo 2. 首次运行时，程序会自动初始化数据库
echo.
echo 模型说明：
echo -----------
echo - YOLO 模型已包含在 models/yolo/ 目录
echo - CLIP 模型需要在程序中下载（工具 - 模型管理）
echo - CLIP 模型约 650MB，用于视觉特征提取
echo.
echo 配置说明：
echo -----------
echo - API 密钥配置：工具 - 设置 - 高级 API 配置
echo - API 密钥保存在 .env 文件中
echo - 模型配置保存在 config/ 目录
echo.
echo 目录结构：
echo -----------
echo title-classifier/
echo ├── title-classifier.exe    主程序
echo ├── _internal/              依赖文件（请勿删除）
echo ├── models/yolo/            YOLO 模型文件
echo ├── config/                 配置文件
echo ├── .env                    API 密钥
echo └── README.txt              本文件
echo.
echo 常见问题：
echo -----------
echo Q: 程序无法启动？
echo A: 确保 Windows 版本为 10 或更高，且已安装 Visual C++ 运行库
echo.
echo Q: 如何下载 CLIP 模型？
echo A: 启动程序后，点击 菜单 - 工具 - 模型管理 - 下载 CLIP 模型
echo.
echo Q: 如何配置 API 密钥？
echo A: 点击 菜单 - 工具 - 设置 - 高级 API 配置
echo.
echo ============================================================
) > "%DIST_DIR%\README.txt"

REM 压缩为 zip
echo.
echo [3/3] 压缩为 zip 文件...
powershell -Command "Compress-Archive -Path '%DIST_DIR%\*' -DestinationPath '%RELEASE_DIR%\%OUTPUT_NAME%.zip' -Force"
if errorlevel 1 (
    echo 错误: 压缩失败
    pause
    exit /b 1
)

REM 获取文件大小
for %%A in ("%RELEASE_DIR%\%OUTPUT_NAME%.zip") do set ZIP_SIZE=%%~zA
set /a ZIP_SIZE_MB=%ZIP_SIZE% / 1048576

echo.
echo ============================================================
echo   发布完成!
echo ============================================================
echo.
echo   输出文件: %RELEASE_DIR%\%OUTPUT_NAME%.zip
echo   文件大小: %ZIP_SIZE_MB% MB
echo.
echo   分发方式:
echo   1. 将 zip 文件上传到网盘或 GitHub Releases
echo   2. 用户下载后解压
echo   3. 双击 title-classifier.exe 运行
echo.
echo ============================================================

pause
