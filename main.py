# main.py — LoRa Coverage Planner 진입점
import sys, os

if getattr(sys, 'frozen', False):
    _base = sys._MEIPASS
    for _p in [os.path.join(_base, 'proj'),
               os.path.join(_base, 'pyproj', 'proj_dir', 'share', 'proj')]:
        if os.path.isdir(_p):
            os.environ.setdefault('PROJ_DATA', _p)
            os.environ.setdefault('PROJ_LIB', _p)
            break
    for _p in [os.path.join(_base, 'gdal'),
               os.path.join(_base, 'rasterio', 'gdal_data')]:
        if os.path.isdir(_p):
            os.environ.setdefault('GDAL_DATA', _p)
            break
    if sys.platform == 'win32':
        for _dll_dir in [
            os.path.join(_base, 'rasterio.libs'),
            os.path.join(_base, 'shapely.libs'),
            os.path.join(_base, 'pyproj.libs'),
            os.path.join(_base, 'pyogrio.libs'),
        ]:
            if os.path.isdir(_dll_dir):
                os.add_dll_directory(_dll_dir)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from ui.main_window import MainWindow
from ui.splash_screen import SplashScreen

if getattr(sys, 'frozen', False):
    _APP_DIR = os.path.dirname(sys.executable)
else:
    _APP_DIR = os.path.dirname(os.path.abspath(__file__))

SHP_PATH = os.path.join(_APP_DIR, "data", "Outline_Seongnam_3857.shp")
DEM_PATH = os.path.join(_APP_DIR, "data", "dem_build_seongnam_3857-2.img")

print(f"[INFO] APP_DIR : {_APP_DIR}")
print(f"[INFO] SHP_PATH: {SHP_PATH}  exists={os.path.exists(SHP_PATH)}")
print(f"[INFO] DEM_PATH: {DEM_PATH}  exists={os.path.exists(DEM_PATH)}")

if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # ── 스플래시 스크린 ──────────────────────────────────────
    splash = SplashScreen()
    splash.show()

    # START 버튼 클릭 시 메인창 실행
    def _launch():
        win = MainWindow(SHP_PATH, DEM_PATH)
        win.show()
        # win을 전역에 유지 (GC 방지)
        app._main_win = win

    splash.sig_start.connect(_launch)

    sys.exit(app.exec_())