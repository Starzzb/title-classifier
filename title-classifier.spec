# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for title-classifier

block_cipher = None

a = Analysis(
    ['src/title_classifier/__main__.py'],
    pathex=['src'],
    binaries=[],
    datas=[
        ('src/title_classifier/gui/assets', 'title_classifier/gui/assets'),
        ('src/title_classifier/core/db_schema.sql', 'title_classifier/core'),
        ('models/yolo', 'models/yolo'),
        ('config', 'config')
    ],
    hiddenimports=[
        'title_classifier',
        'title_classifier.core',
        'title_classifier.core.db_store',
        'title_classifier.detectors',
        'title_classifier.detectors.yolo',
        'title_classifier.gui',
        'title_classifier.gui.app',
        'title_classifier.gui.settings',
        'title_classifier.gui.api_config',
        'title_classifier.gui.model_manager',
        'title_classifier.providers',
        'title_classifier.utils',
        'title_classifier.utils.config',
        'title_classifier.utils.audio',
        'ttkbootstrap',
        'ultralytics',
        'open_clip',
        'torch',
        'torchvision',
        'cv2',
        'numpy',
        'PIL',
        'onnxruntime',
        'openvino',
        'silero_vad'
    ],
    hookspath=[],
    hooksconfig={},
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
