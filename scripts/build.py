"""PyInstaller 构建脚本 - 自动化打包 title-classifier"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# 项目根目录
PROJECT_DIR = Path(__file__).parent.parent.resolve()
DIST_DIR = PROJECT_DIR / "dist"
BUILD_DIR = PROJECT_DIR / "build"
SPEC_FILE = PROJECT_DIR / "title-classifier.spec"

# 入口文件（使用 __main__.py 作为入口，支持相对导入）
GUI_ENTRY = PROJECT_DIR / "src" / "title_classifier" / "__main__.py"

# 需要包含的数据文件
DATA_FILES = [
    ("src/title_classifier/gui/assets", "title_classifier/gui/assets"),
    ("models/yolo", "models/yolo"),
    ("config", "config"),
]

# 需要排除的目录（CLIP 模型太大，不打包）
EXCLUDE_DIRS = [
    "models/clip",
    "data",
    "tests",
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
]

# 隐藏导入
HIDDEN_IMPORTS = [
    "title_classifier",
    "title_classifier.core",
    "title_classifier.core.db_store",
    "title_classifier.detectors",
    "title_classifier.detectors.yolo",
    "title_classifier.gui",
    "title_classifier.gui.app",
    "title_classifier.gui.settings",
    "title_classifier.gui.api_config",
    "title_classifier.gui.model_manager",
    "title_classifier.providers",
    "title_classifier.utils",
    "title_classifier.utils.config",
    "title_classifier.utils.audio",
    "ttkbootstrap",
    "ultralytics",
    "open_clip",
    "torch",
    "torchvision",
    "cv2",
    "numpy",
    "PIL",
    "onnxruntime",
    "openvino",
    "silero_vad",
]


def clean_build():
    """清理构建目录"""
    print("清理构建目录...")
    for dir_name in [BUILD_DIR, DIST_DIR]:
        if dir_name.exists():
            shutil.rmtree(dir_name)
            print(f"  已删除: {dir_name}")

    if SPEC_FILE.exists():
        SPEC_FILE.unlink()
        print(f"  已删除: {SPEC_FILE}")


def create_spec_file():
    """创建 PyInstaller spec 文件"""
    print("创建 spec 文件...")

    # 构建 datas 列表
    datas_lines = []
    for src, dst in DATA_FILES:
        src_path = PROJECT_DIR / src
        if src_path.exists():
            datas_lines.append(f"('{src}', '{dst}')")
        else:
            print(f"  警告: {src} 不存在，跳过")

    datas_str = ",\n        ".join(datas_lines)

    # 构建 hiddenimports 列表
    hiddenimports_str = ",\n        ".join([f"'{m}'" for m in HIDDEN_IMPORTS])

    spec_content = f"""# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for title-classifier

block_cipher = None

a = Analysis(
    ['src/title_classifier/__main__.py'],
    pathex=['src'],
    binaries=[],
    datas=[
        {datas_str}
    ],
    hiddenimports=[
        {hiddenimports_str}
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'scipy',
        'pandas',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'pytest-cov',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='title-classifier',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI 应用，不显示控制台
    icon='src/title_classifier/gui/assets/icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='title-classifier',
)
"""

    with open(SPEC_FILE, "w", encoding="utf-8") as f:
        f.write(spec_content)

    print(f"  已创建: {SPEC_FILE}")


def run_pyinstaller():
    """运行 PyInstaller"""
    print("运行 PyInstaller...")

    # 检查 PyInstaller 是否安装
    try:
        import PyInstaller
        print(f"  PyInstaller 版本: {PyInstaller.__version__}")
    except ImportError:
        print("  错误: PyInstaller 未安装")
        print("  请运行: pip install pyinstaller")
        return False

    # 运行 PyInstaller
    cmd = [
        sys.executable, "-m", "PyInstaller",
        str(SPEC_FILE),
        "--clean",
        "--noconfirm",
    ]

    print(f"  命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_DIR))

    if result.returncode != 0:
        print("  错误: PyInstaller 构建失败")
        return False

    print("  PyInstaller 构建完成")
    return True


def copy_additional_files():
    """复制额外的文件到 dist 目录"""
    print("复制额外文件...")

    dist_app_dir = DIST_DIR / "title-classifier"
    if not dist_app_dir.exists():
        print("  错误: dist 目录不存在")
        return

    # 复制 .env 文件（如果存在）
    env_file = PROJECT_DIR / ".env"
    if env_file.exists():
        shutil.copy2(env_file, dist_app_dir / ".env")
        print("  已复制: .env")

    # 复制 README
    readme_file = PROJECT_DIR / "README.md"
    if readme_file.exists():
        shutil.copy2(readme_file, dist_app_dir / "README.md")
        print("  已复制: README.md")


def print_build_info():
    """打印构建信息"""
    print("\n" + "=" * 60)
    print("构建完成!")
    print("=" * 60)

    dist_app_dir = DIST_DIR / "title-classifier"
    if dist_app_dir.exists():
        # 计算总大小
        total_size = sum(f.stat().st_size for f in dist_app_dir.rglob("*") if f.is_file())
        size_mb = total_size / (1024 * 1024)

        print(f"\n输出目录: {dist_app_dir}")
        print(f"总大小: {size_mb:.1f} MB")

        # 列出主要文件
        exe_file = dist_app_dir / "title-classifier.exe"
        if exe_file.exists():
            exe_size = exe_file.stat().st_size / (1024 * 1024)
            print(f"可执行文件: {exe_file} ({exe_size:.1f} MB)")

        # 检查模型文件
        models_dir = dist_app_dir / "models" / "yolo"
        if models_dir.exists():
            model_files = list(models_dir.glob("*.pt"))
            print(f"YOLO 模型: {len(model_files)} 个")

    print("\n运行方式:")
    print(f"  双击: {dist_app_dir / 'title-classifier.exe'}")
    print(f"  命令行: {dist_app_dir / 'title-classifier.exe'}")
    print("=" * 60)


def main():
    """主函数"""
    print("=" * 60)
    print("title-classifier 打包脚本")
    print("=" * 60)

    # 检查入口文件
    if not GUI_ENTRY.exists():
        print(f"错误: 入口文件不存在: {GUI_ENTRY}")
        return 1

    # 清理
    clean_build()

    # 创建 spec 文件
    create_spec_file()

    # 运行 PyInstaller
    if not run_pyinstaller():
        return 1

    # 复制额外文件
    copy_additional_files()

    # 打印构建信息
    print_build_info()

    return 0


if __name__ == "__main__":
    sys.exit(main())
