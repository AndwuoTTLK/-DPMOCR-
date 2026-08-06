# -*- mode: python ; coding: utf-8 -*-
import sys
import sysconfig
import pkgutil
from pathlib import Path

datas = [('model', 'model')]
binaries = []

# 把整个 site-packages 原样复制进 _internal，运行时直接从包内 import。
# 打包分析阶段不执行 paddle/paddlex 等重型依赖，避免低内存环境下崩溃。
_site_packages = Path(sysconfig.get_paths()['purelib'])
datas.append((str(_site_packages), '.'))

# 分析阶段排除全部第三方包，只分析业务代码、标准库和下面的标准库模块清单
_excludes = []
for _entry in _site_packages.iterdir():
    if _entry.name.endswith(('.dist-info', '.egg-info', '.pth')):
        continue
    if _entry.is_dir():
        _excludes.append(_entry.name)
    elif _entry.suffix == '.py':
        _excludes.append(_entry.stem)

def _walk_pkg_modules(package_dir, prefix):
    """不 import 包体，仅静态列出包内所有模块名。"""
    mods = []
    stack = [(package_dir, prefix)]
    while stack:
        d, pre = stack.pop()
        for info in pkgutil.iter_modules([str(d)]):
            mods.append(pre + info.name)
            if info.ispkg:
                stack.append((d / info.name, pre + info.name + '.'))
    return mods

# 把标准库整体静态扫描进包，避免 paddle/paddlex 运行时缺标准库模块
_SKIP_STDLIB = {'test', 'idlelib', 'lib2to3', 'ensurepip', 'turtledemo', 'venv',
                'site-packages', '__pycache__'}
_stdlib_dir = Path(sys.base_prefix) / 'Lib'
hiddenimports = ['__future__']
for _p in sorted(_stdlib_dir.iterdir()):
    if _p.name in _SKIP_STDLIB or '-' in _p.name:
        continue
    if _p.suffix == '.py':
        hiddenimports.append(_p.stem)
    elif _p.is_dir() and (_p / '__init__.py').exists():
        hiddenimports.append(_p.name)
        hiddenimports += _walk_pkg_modules(_p, _p.name + '.')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=_excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='OCR',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['图标文件.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='OCR',
)
