# -*- mode: python ; coding: utf-8 -*-
"""日股因子研究工作台 打包配置
构建: buildenv314\Scripts\python.exe -m PyInstaller quant_app.spec --noconfirm
产物: dist\日股因子研究工作台.exe (onefile, 无控制台)
"""
import os

from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
for pkg in ("PyQt5", "matplotlib", "pandas", "pyarrow"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

hiddenimports += [
    "xlrd",                      # 读JPX的xls名录
    "pandas._libs.tslibs.base",
    "matplotlib.backends.backend_qtagg",
    # 本项目模块：多处是函数内延迟导入(import portfolio 等)，静态分析可能漏收，显式声明
    "universe", "data", "evaluate", "factors", "neutralize", "costs",
    "research_log", "signals", "diagnosis", "levels", "sector_rel",
    "watchlist", "portfolio", "tracker", "screener", "sectors_custom",
    "plain", "upside_score", "run_validate_diagnosis", "run_weight_criteria",
    "run_upside_oos", "run_calibration", "run_robustness",
]

a = Analysis(
    ["app.py"],
    pathex=[os.path.abspath(SPECPATH)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "PyQt6", "PySide2", "PySide6", "scipy", "IPython", "jupyter",
              "notebook", "sphinx", "pytest", "cv2"],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    a.zipfiles,
    [],
    name="日股因子研究工作台",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,          # 无黑框
    disable_windowed_traceback=False,
)
