# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onedir build for Wire Detection field tests.

This build intentionally excludes SAM2, PyTorch, TorchVision, Hydra, and model
checkpoints. It is meant for quick Wire Detection validation with saved virtual
camera folders or a D405 camera, not for SAM2 teaching/tracking.

Build:
    .venv\Scripts\pyinstaller.exe --noconfirm --clean build_wire_test.spec

Result:
    dist/WireVision_WireTest/
      WireVision_WireTest.exe
      _internal/
"""


a = Analysis(
    ["PythonUI.py"],
    pathex=[],
    binaries=[],
    datas=[
        ("bolt.png", "."),
        ("bolt.ico", "."),
    ],
    hiddenimports=[
        "pyrealsense2",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch",
        "torchvision",
        "torchaudio",
        "sam2",
        "hydra",
        "hydra_core",
        "omegaconf",
        "matplotlib",
        "pytest",
        "IPython",
        "notebook",
        "tkinter",
        "numpy.testing",
        "PyQt5.QtWebEngineWidgets",
        "PyQt5.QtBluetooth",
        "PyQt5.QtQuick",
        "PyQt5.QtQml",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="WireVision_WireTest",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="bolt.ico",
    manifest="windows_per_monitor_v2.manifest",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="WireVision_WireTest",
)
