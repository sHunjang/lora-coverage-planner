# main.py — LoRa Coverage Planner 진입점
import sys, os, json

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

from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QMessageBox,
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QGroupBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from ui.main_window import MainWindow
from ui.splash_screen import SplashScreen

if getattr(sys, 'frozen', False):
    _APP_DIR = os.path.dirname(sys.executable)
else:
    _APP_DIR = os.path.dirname(os.path.abspath(__file__))

# 기본 데이터 경로
_DEFAULT_SHP = os.path.join(_APP_DIR, "data", "Outline_Seongnam_3857.shp")
_DEFAULT_DEM = os.path.join(_APP_DIR, "data", "dem_build_seongnam_3857-2.img")

# 마지막 사용 경로 저장 파일
_DATA_CFG = os.path.join(_APP_DIR, "data_paths.json")

DARK   = "#181b22"
PANEL  = "#1e2130"
TEXT   = "#e0e4ef"
MUTED  = "#7a8099"
BORDER = "#2a2f3b"

STYLE_DLG = f"""
    QDialog    {{ background:{DARK};  color:{TEXT};  }}
    QLabel     {{ color:{TEXT};       }}
    QLineEdit  {{ background:{PANEL}; color:{TEXT};
                  border:1px solid {BORDER}; border-radius:4px;
                  padding:4px 8px; font-size:11px; }}
    QGroupBox  {{ color:{MUTED}; border:1px solid {BORDER};
                  border-radius:6px; margin-top:6px; padding-top:8px;
                  font-size:11px; }}
    QGroupBox::title {{ subcontrol-origin:margin; left:8px; }}
"""
BTN = ("QPushButton{background:#1c2a3a;color:#7ab8e8;"
       "border:1px solid #2a4a6a;border-radius:5px;"
       "padding:6px 16px;font-size:11px;}"
       "QPushButton:hover{background:#254d78;}"
       "QPushButton:disabled{color:#3a5a6a;border-color:#1a2a3a;}")
BTN_GREEN = ("QPushButton{background:#1d3a1d;color:#7ae87a;"
             "border:1px solid #2a5a2a;border-radius:5px;"
             "padding:7px 20px;font-size:12px;font-weight:bold;}"
             "QPushButton:hover{background:#256a25;}"
             "QPushButton:disabled{color:#3a5a3a;}")


def _load_saved_paths() -> tuple[str, str]:
    """마지막으로 사용한 경로를 data_paths.json에서 불러옵니다."""
    try:
        if os.path.exists(_DATA_CFG):
            with open(_DATA_CFG, encoding='utf-8') as f:
                d = json.load(f)
            return d.get('shp', ''), d.get('dem', '')
    except Exception:
        pass
    return '', ''


def _save_paths(shp: str, dem: str):
    """사용한 경로를 data_paths.json에 저장합니다."""
    try:
        with open(_DATA_CFG, 'w', encoding='utf-8') as f:
            json.dump({'shp': shp, 'dem': dem}, f,
                      ensure_ascii=False, indent=2)
    except Exception:
        pass


class DataPathDialog(QDialog):
    """
    SHP / DEM 파일 경로 선택 다이얼로그.
    - data/ 폴더에 기본 파일이 있으면 자동 채움
    - 마지막 사용 경로도 복원
    - 두 파일 모두 선택해야 확인 버튼 활성화
    """

    def __init__(self, shp_hint: str = '', dem_hint: str = '', parent=None):
        super().__init__(parent)
        self.setWindowTitle("데이터 파일 선택")
        self.setStyleSheet(STYLE_DLG)
        self.setFixedWidth(580)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)

        self.shp_path = ''
        self.dem_path = ''

        self._build(shp_hint, dem_hint)
        self._validate()

    def _build(self, shp_hint: str, dem_hint: str):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)

        # ── 안내 ─────────────────────────────────────────
        lbl_info = QLabel(
            "분석에 사용할 지역 경계(Shapefile)와 "
            "수치 표면 모델(DEM/DSM) 파일을 선택하세요.\n"
            "data/ 폴더에 파일이 있으면 자동으로 불러옵니다.")
        lbl_info.setStyleSheet(f"color:{MUTED};font-size:11px;")
        lbl_info.setWordWrap(True)
        lay.addWidget(lbl_info)

        # ── SHP 선택 ──────────────────────────────────────
        grp_shp = QGroupBox("지역 경계 파일 (Shapefile *.shp)")
        shp_lay = QHBoxLayout(grp_shp)
        self.edit_shp = QLineEdit(shp_hint)
        self.edit_shp.setPlaceholderText("경계 Shapefile 경로 (.shp)")
        self.edit_shp.setReadOnly(True)
        btn_shp = QPushButton("찾아보기")
        btn_shp.setStyleSheet(BTN)
        btn_shp.setFixedWidth(90)
        btn_shp.clicked.connect(self._browse_shp)
        shp_lay.addWidget(self.edit_shp, 1)
        shp_lay.addWidget(btn_shp)
        lay.addWidget(grp_shp)

        # ── DEM 선택 ──────────────────────────────────────
        grp_dem = QGroupBox("수치 표면 모델 파일 (DEM/DSM *.img / *.tif)")
        dem_lay = QHBoxLayout(grp_dem)
        self.edit_dem = QLineEdit(dem_hint)
        self.edit_dem.setPlaceholderText("DEM/DSM 래스터 파일 경로")
        self.edit_dem.setReadOnly(True)
        btn_dem = QPushButton("찾아보기")
        btn_dem.setStyleSheet(BTN)
        btn_dem.setFixedWidth(90)
        btn_dem.clicked.connect(self._browse_dem)
        dem_lay.addWidget(self.edit_dem, 1)
        dem_lay.addWidget(btn_dem)
        lay.addWidget(grp_dem)

        # ── 상태 라벨 ─────────────────────────────────────
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet(
            f"color:{MUTED};font-size:10px;padding:2px 0;")
        self.lbl_status.setWordWrap(True)
        lay.addWidget(self.lbl_status)

        # ── 버튼 ─────────────────────────────────────────
        bot = QHBoxLayout()
        btn_cancel = QPushButton("취소")
        btn_cancel.setStyleSheet(BTN)
        btn_cancel.clicked.connect(self.reject)

        self.btn_ok = QPushButton("✔  확인 — 분석 시작")
        self.btn_ok.setStyleSheet(BTN_GREEN)
        self.btn_ok.clicked.connect(self._accept)

        bot.addStretch()
        bot.addWidget(btn_cancel)
        bot.addWidget(self.btn_ok)
        lay.addLayout(bot)

    def _browse_shp(self):
        start = (os.path.dirname(self.edit_shp.text())
                 if self.edit_shp.text() else _APP_DIR)
        path, _ = QFileDialog.getOpenFileName(
            self, "경계 Shapefile 선택", start,
            "Shapefile (*.shp);;모든 파일 (*.*)")
        if path:
            self.edit_shp.setText(path)
            self._validate()

    def _browse_dem(self):
        start = (os.path.dirname(self.edit_dem.text())
                 if self.edit_dem.text() else _APP_DIR)
        path, _ = QFileDialog.getOpenFileName(
            self, "DEM/DSM 파일 선택", start,
            "래스터 파일 (*.img *.tif *.tiff *.vrt);;모든 파일 (*.*)")
        if path:
            self.edit_dem.setText(path)
            self._validate()

    def _validate(self):
        """두 파일 존재 여부를 확인하고 확인 버튼 활성화."""
        shp = self.edit_shp.text().strip()
        dem = self.edit_dem.text().strip()

        shp_ok = bool(shp) and os.path.exists(shp)
        dem_ok = bool(dem) and os.path.exists(dem)

        msgs = []
        if shp and not shp_ok:
            msgs.append("⚠ SHP 파일을 찾을 수 없습니다.")
        if dem and not dem_ok:
            msgs.append("⚠ DEM 파일을 찾을 수 없습니다.")
        if not shp:
            msgs.append("• 경계 Shapefile을 선택하세요.")
        if not dem:
            msgs.append("• DEM/DSM 파일을 선택하세요.")

        self.lbl_status.setText("  ".join(msgs))
        self.btn_ok.setEnabled(shp_ok and dem_ok)

        if shp_ok and dem_ok:
            self.lbl_status.setText(
                f"✔ SHP: {os.path.basename(shp)}  |  "
                f"DEM: {os.path.basename(dem)}")
            self.lbl_status.setStyleSheet(
                f"color:#7ae87a;font-size:10px;padding:2px 0;")
        else:
            self.lbl_status.setStyleSheet(
                f"color:{MUTED};font-size:10px;padding:2px 0;")

    def _accept(self):
        self.shp_path = self.edit_shp.text().strip()
        self.dem_path = self.edit_dem.text().strip()
        self.accept()


def _resolve_data_paths() -> tuple[str, str] | None:
    """
    데이터 경로 결정 로직:
    1. data/ 폴더에 기본 파일 존재 → 자동 사용 (다이얼로그 없음)
    2. 저장된 경로 존재 → 다이얼로그에 미리 채움
    3. 아무것도 없음 → 빈 다이얼로그
    반환값: (shp_path, dem_path) 또는 None (취소)
    """
    # 기본 파일이 모두 있으면 바로 사용
    if os.path.exists(_DEFAULT_SHP) and os.path.exists(_DEFAULT_DEM):
        print(f"[INFO] 기본 데이터 파일 자동 사용")
        print(f"[INFO] SHP: {_DEFAULT_SHP}")
        print(f"[INFO] DEM: {_DEFAULT_DEM}")
        return _DEFAULT_SHP, _DEFAULT_DEM

    # 저장된 경로 복원
    saved_shp, saved_dem = _load_saved_paths()

    # 저장된 경로가 모두 유효하면 바로 사용
    if (saved_shp and os.path.exists(saved_shp)
            and saved_dem and os.path.exists(saved_dem)):
        print(f"[INFO] 저장된 데이터 경로 복원")
        print(f"[INFO] SHP: {saved_shp}")
        print(f"[INFO] DEM: {saved_dem}")
        return saved_shp, saved_dem

    # 힌트: 기본 파일 중 존재하는 것은 미리 채움
    shp_hint = _DEFAULT_SHP if os.path.exists(_DEFAULT_SHP) else saved_shp
    dem_hint = _DEFAULT_DEM if os.path.exists(_DEFAULT_DEM) else saved_dem

    # 다이얼로그 표시
    dlg = DataPathDialog(shp_hint=shp_hint, dem_hint=dem_hint)
    if dlg.exec_() != QDialog.Accepted:
        return None  # 취소

    _save_paths(dlg.shp_path, dlg.dem_path)
    return dlg.shp_path, dlg.dem_path


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # ── 데이터 경로 결정 ─────────────────────────────────────
    result = _resolve_data_paths()
    if result is None:
        QMessageBox.information(
            None, "종료",
            "데이터 파일을 선택하지 않아 프로그램을 종료합니다.")
        sys.exit(0)

    shp_path, dem_path = result
    print(f"[INFO] SHP_PATH: {shp_path}  exists={os.path.exists(shp_path)}")
    print(f"[INFO] DEM_PATH: {dem_path}  exists={os.path.exists(dem_path)}")

    # ── 스플래시 스크린 ──────────────────────────────────────
    splash = SplashScreen()
    splash.show()

    def _launch():
        """START 클릭 후 호출 — 공간 데이터 로드 + 진행률 표시."""
        # 단계 1: 초기화
        splash.update_progress(10, "공간 데이터 로드 중...")

        from ui.main_window import MainWindow
        from core.dem_loader import SpatialData

        # 단계 2: SHP 로드
        splash.update_progress(30, f"경계 데이터 로드 중... {os.path.basename(shp_path)}")
        try:
            spatial = SpatialData(shp_path, dem_path)
        except Exception as e:
            splash.update_progress(0, f"오류: {e}")
            QMessageBox.critical(None, "데이터 로드 실패",
                                 f"공간 데이터를 불러올 수 없습니다.\n{e}")
            sys.exit(1)

        # 단계 3: DEM 로드
        splash.update_progress(60, f"DEM/DSM 로드 중... {os.path.basename(dem_path)}")
        try:
            spatial.load()
        except Exception as e:
            splash.update_progress(0, f"오류: {e}")
            QMessageBox.critical(None, "DEM 로드 실패",
                                 f"DEM 데이터를 불러올 수 없습니다.\n{e}")
            sys.exit(1)

        # 단계 4: MainWindow 생성
        splash.update_progress(85, "메인 창 초기화 중...")
        win = MainWindow(shp_path, dem_path)

        # spatial을 MainWindow에 직접 주입 (재로드 방지)
        win.spatial = spatial
        win.status.showMessage("준비")
        win.map_w.refresh()

        splash.update_progress(95, "세션 복원 중...")
        win._load_session()

        # 단계 5: 완료
        splash.update_progress(100, "준비 완료")
        win.show()
        app._main_win = win

        # 500ms 후 스플래시 자동 닫기
        splash.finish_loading(delay_ms=500)

    splash.sig_start.connect(_launch)
    sys.exit(app.exec_())