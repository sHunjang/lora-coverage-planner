# main.py — LoRa Coverage Planner 진입점
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from ui.main_window import MainWindow

SHP_PATH = "data/Outline_Seongnam_3857.shp"
DEM_PATH = "data/dem_build_seongnam_3857-2.img"

if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    win = MainWindow(SHP_PATH, DEM_PATH)
    win.show()
    sys.exit(app.exec_())
