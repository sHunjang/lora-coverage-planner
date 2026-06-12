# LoRaPlanner.spec
# 사용법: pyinstaller LoRaPlanner.spec

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs

block_cipher = None

# ── 숨겨진 import 목록 ────────────────────────────────────────
hiddenimports = [
    # rasterio 관련 (핵심)
    'rasterio',
    'rasterio.serde',
    'rasterio._base',
    'rasterio._shim',
    'rasterio.control',
    'rasterio.crs',
    'rasterio.drivers',
    'rasterio.dtypes',
    'rasterio.enums',
    'rasterio.env',
    'rasterio.errors',
    'rasterio.features',
    'rasterio.fill',
    'rasterio.gdal',
    'rasterio.mask',
    'rasterio.merge',
    'rasterio.plot',
    'rasterio.profiles',
    'rasterio.sample',
    'rasterio.transform',
    'rasterio.vrt',
    'rasterio.warp',
    'rasterio.windows',
    'rasterio._err',
    'rasterio._filepath',
    'rasterio._warp',
    'rasterio._features',

    # geopandas
    'geopandas',
    'geopandas._compat',
    'geopandas.array',
    'geopandas.geodataframe',
    'geopandas.geoseries',
    'geopandas.io.file',
    'geopandas.io.arrow',

    # shapely
    'shapely',
    'shapely.geometry',
    'shapely.vectorized',
    'shapely.ops',
    'shapely.affinity',
    'shapely.prepared',
    'shapely.lib',

    # pyproj
    'pyproj',
    'pyproj.transformer',
    'pyproj._crs',
    'pyproj._transformer',

    # pyogrio
    'pyogrio',
    'pyogrio._geometry',
    'pyogrio._io',
    'pyogrio.errors',
    'pyogrio.raw',

    # sklearn
    'sklearn',
    'sklearn.cluster',
    'sklearn.cluster._kmeans',
    'sklearn.utils',
    'sklearn.utils._cython_blas',
    'sklearn.neighbors',
    'sklearn.neighbors._partition_nodes',

    # scipy
    'scipy',
    'scipy.ndimage',
    'scipy.ndimage._filters',
    'scipy.ndimage._interpolation',
    'scipy.ndimage._measurements',
    'scipy.ndimage._morphology',
    'scipy.ndimage._label',
    'scipy.ndimage._ni_label',
    'scipy.stats',
    'scipy.spatial',

    # matplotlib
    'matplotlib',
    'matplotlib.backends.backend_qt5agg',
    'matplotlib.backends.backend_agg',
    'matplotlib.figure',
    'matplotlib.font_manager',
    'matplotlib.contour',
    'matplotlib.tri',

    # PIL / Pillow
    'PIL',
    'PIL.Image',
    'PIL.ImageFilter',

    # numpy
    'numpy',
    'numpy.core',
    'numpy.core._multiarray_umath',

    # pandas
    'pandas',
    'pandas.core.frame',
    'pandas.core.series',

    # PyQt5
    'PyQt5',
    'PyQt5.QtWebEngineWidgets',
    'PyQt5.QtWebChannel',
    'PyQt5.QtWebEngineCore',
    'PyQt5.sip',
    'PyQt5.QtPrintSupport',

    # folium
    'folium',
    'folium.raster_layers',
    'folium.features',

    # xyzservices (folium 타일 서비스)
    'xyzservices',

    # pulp
    'pulp',
    'pulp.apis',
    'pulp.apis.coin_api',
    'pulp.constants',

    # concurrent.futures (LinkMatrixWorker)
    'concurrent.futures',
    'concurrent.futures.thread',
    'concurrent.futures._base',

    # multiprocessing / pickle
    'multiprocessing',
    'multiprocessing.context',
    'multiprocessing.reduction',
    'multiprocessing.managers',
    'multiprocessing.pool',
    'multiprocessing.process',
    'pickle',
    '_pickle',
    'copyreg',

    # 프로젝트 내부 — core
    'core.utils',
    'core.coverage',
    'core.dem_loader',
    'core.propagation',
    'core.gw_optimizer',
    'core.link_matrix',
    'core.app_config',
    'core.license',
    # core._secret_key 는 빌드 시 회사별로 존재하며 datas로 포함됨.
    # 없을 수도 있으므로 hiddenimports에는 넣지 않는다.

    # 프로젝트 내부 — ui
    'ui.splash_screen',
    'ui.main_window',
    'ui.map_widget',
    'ui.settings_window',
    'ui.result_panel',
    'ui.legend_window',
    'ui.dialogs',
    'ui.gw_list_window',
    'ui.gw_optimize_window',
    'ui.gw_node_detail_window',
    'ui.node_list_window',
    'ui.node_gw_detail_window',
    'ui.distance_window',
    'ui.linkbudget_window',
    'ui.profile_window',
    'ui.compare_window',
    'ui.graph_window',
    'ui.report_window',
    'ui.initial_setup_dialog',
    'ui.license_dialog',

    # 기타
    'certifi',
    'packaging',
    'attr',
    'attrs',
    'click',
    'jinja2',
    'branca',
]

# ── 데이터 파일 수집 ─────────────────────────────────────────
datas = []
datas += collect_data_files('rasterio')
datas += collect_data_files('geopandas')
datas += collect_data_files('pyogrio')
datas += collect_data_files('pyproj')
datas += collect_data_files('shapely')
datas += collect_data_files('folium')
datas += collect_data_files('branca')
datas += collect_data_files('matplotlib')
datas += collect_data_files('certifi')
datas += collect_data_files('pulp')

try:
    datas += collect_data_files('xyzservices')
except Exception:
    pass

import os

# ── 회사별 비밀키 (빌드 시 반드시 포함) ─────────────────────
# core/_secret_key.py 가 없으면 라이선스 검증이 폴백 키로 동작하므로
# 배포 빌드 전에 tools/make_company_key.py 로 반드시 생성해야 한다.
_secret_path = os.path.join('core', '_secret_key.py')
if os.path.isfile(_secret_path):
    datas.append((_secret_path, 'core'))
    print(f"[빌드] 비밀키 포함: {_secret_path}")
else:
    print("[경고] core/_secret_key.py 없음 — "
          "make_company_key.py로 먼저 생성하세요! "
          "(현재 빌드는 개발용 폴백 키로 동작합니다)")

# data/ 폴더는 exe 옆에 별도로 위치 (번들에 포함하지 않음)

# assets 폴더 포함
if os.path.isdir('assets'):
    datas.append(('assets', 'assets'))

# conda 환경의 proj 데이터 (pyproj용)
conda_base = os.path.dirname(os.path.dirname(os.__file__))
proj_candidates = [
    os.path.join(conda_base, 'Library', 'share', 'proj'),
    os.path.join(conda_base, 'share', 'proj'),
]
for p in proj_candidates:
    if os.path.isdir(p):
        datas.append((p, 'proj'))
        break

# rasterio gdal_data
gdal_candidates = [
    os.path.join(conda_base, 'Library', 'share', 'gdal'),
    os.path.join(conda_base, 'share', 'gdal'),
]
for p in gdal_candidates:
    if os.path.isdir(p):
        datas.append((p, 'gdal'))
        break

# ── 바이너리 (DLL 등) 수집 ──────────────────────────────────
binaries = []
binaries += collect_dynamic_libs('rasterio')
binaries += collect_dynamic_libs('pyogrio')
binaries += collect_dynamic_libs('shapely')
binaries += collect_dynamic_libs('pyproj')

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'wx',
        'PySide2',
        'PySide6',
        'PyQt6',
        'IPython',
        'jupyter',
        'notebook',
        'ipykernel',
        'ipywidgets',
        'test',
        'tests',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ── onedir 방식 (실행 속도 향상) ────────────────────────────
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LoRaPlanner',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=False,          # 디버깅 시 True로 변경
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,              # 아이콘 있으면 'icon.ico'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='LoRaPlanner',
)
