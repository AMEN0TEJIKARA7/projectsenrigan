# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Project Senrigan desktop app.

Build from the repository root, after training the artifact:
    python src/ingest.py && python src/train_final.py
    pyinstaller app/senrigan.spec

Produces dist/ProjectSenrigan.exe (Windows) or dist/ProjectSenrigan (macOS/Linux)
as a single self-contained file. The repository's `src/` scripts ride along as
plain files so the in-app "Update ratings" can import and run them, and the
trained artifact is bundled so the first launch works before any update.
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).resolve().parent          # repository root
ARTIFACT = ROOT / "models" / "lol_logistic_sigmoid_v1.joblib"
if not ARTIFACT.exists():
    raise SystemExit(f"train the artifact first (missing {ARTIFACT})")

datas = [
    (str(ROOT / "app" / "ui"), "app/ui"),
    (str(ARTIFACT), "models"),
] + [(str(p), "src") for p in (ROOT / "src").glob("*.py")]

hiddenimports = collect_submodules("sklearn") + [
    "pyarrow", "pyarrow.parquet", "pyarrow.lib", "pyarrow._parquet",
    "scipy.special", "scipy.sparse", "scipy.stats",
]
binaries = []
for pkg in ("webview", "pyarrow"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    [str(ROOT / "app" / "main.py")],
    pathex=[str(ROOT / "src"), str(ROOT / "app")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["lightgbm", "matplotlib", "tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6",
              "IPython", "jupyter", "notebook", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="ProjectSenrigan",
    debug=False,
    strip=False,
    upx=False,
    console=False,                 # no terminal window behind the app
    disable_windowed_traceback=False,
    icon=str(ROOT / "app" / "icon.ico") if (ROOT / "app" / "icon.ico").exists() else None,
)
