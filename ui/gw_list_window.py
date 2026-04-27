# ui/gw_list_window.py
from __future__ import annotations
import csv
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QFileDialog, QLabel, QMessageBox,
)
from PyQt5.QtCore import Qt, pyqtSignal
from core.coverage import GWEntry
from ui.dialogs import GWParamDialog, CoverageSettingsDialog, STYLE_DLG, DARK, PANEL, TEXT, MUTED, BORDER

COLS = ['☑', 'Callsign', '경도', '위도', 'Pt(dBm)',
        'Gt(dBi)', 'Lt(dB)', '높이(m)']

BTN = ("QPushButton{background:#1c2a3a;color:#7ab8e8;"
       "border:1px solid #2a4a6a;border-radius:4px;"
       "padding:5px 12px;font-size:11px;}"
       "QPushButton:hover{background:#254d78;}")
BTN_GREEN = ("QPushButton{background:#1d3a1d;color:#7ae87a;"
             "border:1px solid #2a5a2a;border-radius:4px;"
             "padding:5px 12px;font-size:11px;}"
             "QPushButton:hover{background:#256a25;}")
BTN_RED = ("QPushButton{background:#3a1a1a;color:#e87a7a;"
           "border:1px solid #5a2a2a;border-radius:4px;"
           "padding:5px 12px;font-size:11px;}"
           "QPushButton:hover{background:#5a2020;}")


class GWListWindow(QDialog):
    sig_coverage_requested = pyqtSignal(list, dict)
    sig_coverage_clear     = pyqtSignal()
    sig_coverage_analyze   = pyqtSignal(list)
    sig_map_refresh        = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gateway 목록")
        self.setStyleSheet(STYLE_DLG)
        self.resize(880, 500)
        self.setWindowFlag(Qt.Window)

        self._gws: list[GWEntry] = []
        self._settings: dict = {
            'rx_height': 1.5, 'radius_km': 10.0,
            'min_rx': -126.6,
            'color_levels': [
                {'pr': -90,  'color': '#FF2020'},
                {'pr': -100, 'color': '#FF8C00'},
                {'pr': -110, 'color': '#FFD700'},
                {'pr': -120, 'color': '#00C94A'},
                {'pr': -130, 'color': '#4f8ef7'},
            ],
        }
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)

        top = QHBoxLayout()
        self.btn_add  = QPushButton("+ GW 추가");        self.btn_add.setStyleSheet(BTN_GREEN)
        self.btn_del  = QPushButton("- 선택 삭제");      self.btn_del.setStyleSheet(BTN_RED)
        self.btn_cov  = QPushButton("▶ 선택 커버리지");  self.btn_cov.setStyleSheet(BTN)
        self.btn_detail = QPushButton("📋 연결 단말기"); self.btn_detail.setStyleSheet(BTN)
        self.btn_anl  = QPushButton("🔄 커버리지 분석"); self.btn_anl.setStyleSheet(BTN)
        self.btn_clr  = QPushButton("✕ 커버리지 지우기");self.btn_clr.setStyleSheet(BTN_RED)
        self.btn_prof = QPushButton("📈 단면도");         self.btn_prof.setStyleSheet(BTN)
        self.btn_lnk  = QPushButton("📊 링크버짓");      self.btn_lnk.setStyleSheet(BTN)
        self.btn_dist = QPushButton("📏 거리 분석");      self.btn_dist.setStyleSheet(BTN)
        self.btn_cfg  = QPushButton("⚙ 설정");           self.btn_cfg.setStyleSheet(BTN)
        self.btn_imp  = QPushButton("CSV 가져오기");      self.btn_imp.setStyleSheet(BTN)
        self.btn_exp  = QPushButton("CSV 내보내기");      self.btn_exp.setStyleSheet(BTN)

        for b in [self.btn_add, self.btn_del, self.btn_cov, self.btn_detail,
                  self.btn_anl, self.btn_clr, self.btn_prof, self.btn_dist,
                  self.btn_lnk, self.btn_cfg, self.btn_imp, self.btn_exp]:
            top.addWidget(b)
        top.addStretch()
        lay.addLayout(top)

        self.tbl = QTableWidget(0, len(COLS))
        self.tbl.setHorizontalHeaderLabels(COLS)
        hh = self.tbl.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Stretch)
        hh.setSectionResizeMode(0, QHeaderView.Fixed)
        self.tbl.setColumnWidth(0, 36)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl.setStyleSheet(
            f"QTableWidget{{background:{PANEL};color:{TEXT};"
            f"gridline-color:{BORDER};selection-background-color:#253a5a;}}"
            f"QHeaderView::section{{background:{DARK};color:{MUTED};"
            f"border:none;padding:4px;}}")
        self.tbl.doubleClicked.connect(self._on_double_click)
        lay.addWidget(self.tbl)

        hint = QLabel("더블클릭: 파라미터 편집  |  ☑ 체크 후 [▶ 선택 커버리지]: 지도에 표시")
        hint.setStyleSheet(f"color:{MUTED};font-size:11px;")
        lay.addWidget(hint)

        self.btn_add.clicked.connect(self._add_default)
        self.btn_del.clicked.connect(self._del_selected)
        self.btn_cov.clicked.connect(self._request_coverage)
        self.btn_detail.clicked.connect(self._open_detail)
        self.btn_anl.clicked.connect(self._request_analyze)
        self.btn_clr.clicked.connect(self.sig_coverage_clear.emit)
        self.btn_prof.clicked.connect(self._open_profile)
        self.btn_dist.clicked.connect(self._open_distance)
        self.btn_lnk.clicked.connect(self._open_linkbudget)
        self.btn_cfg.clicked.connect(self._open_settings)
        self.btn_imp.clicked.connect(self._import_csv)
        self.btn_exp.clicked.connect(self._export_csv)

    def _refresh_table(self, suppress_map=False):
        self.tbl.setRowCount(0)
        for gw in self._gws:
            r = self.tbl.rowCount(); self.tbl.insertRow(r)
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk.setCheckState(Qt.Unchecked)
            self.tbl.setItem(r, 0, chk)
            for c, v in enumerate([
                gw.callsign, f"{gw.lon:.6f}", f"{gw.lat:.6f}",
                f"{gw.pt_dbm:.1f}", f"{gw.gt_dbi:.2f}",
                f"{gw.lt_db:.2f}", f"{gw.hb_m:.1f}",
            ], 1):
                self.tbl.setItem(r, c, QTableWidgetItem(str(v)))
        if not suppress_map:
            self.sig_map_refresh.emit()

    def add_gw(self, gw: GWEntry):
        self._gws.append(gw)
        self._refresh_table()

    def set_coord(self, lon: float, lat: float):
        rows = list({i.row() for i in self.tbl.selectedItems()})
        if not rows: return
        r = rows[0]
        if r < len(self._gws):
            g = self._gws[r]
            self._gws[r] = GWEntry(g.callsign, lon, lat,
                                    g.pt_dbm, g.gt_dbi, g.lt_db,
                                    g.hb_m, g.enabled)
            self._refresh_table()
            self.tbl.selectRow(r)

    def get_gws(self) -> list[GWEntry]:
        return list(self._gws)

    def get_settings(self) -> dict:
        return dict(self._settings)

    def _add_default(self):
        n  = len(self._gws) + 1
        # 부모(MainWindow)에서 설정값 가져오기
        s  = getattr(self.parent(), '_settings', {})
        gw = GWEntry(
            callsign = f"GW{n}",
            lon      = 127.10, lat=37.40,
            pt_dbm   = s.get('gw_pt_dbm', 14.0),
            gt_dbi   = s.get('gw_gt_dbi', 2.15),
            lt_db    = s.get('gw_lt_db',  0.0),
            hb_m     = s.get('gw_hb_m',   15.0),
        )
        self._gws.append(gw)
        self._refresh_table()

    def _del_selected(self):
        rows = sorted(
            {i.row() for i in self.tbl.selectedItems()},
            reverse=True)
        for r in rows:
            if r < len(self._gws):
                self._gws.pop(r)
        self._refresh_table()

    def _on_double_click(self, idx):
        r = idx.row()
        if r >= len(self._gws): return
        dlg = GWParamDialog(self._gws[r], self)
        if dlg.exec_() == GWParamDialog.Accepted:
            self._gws[r] = dlg.result_gw(self._gws[r])
            self._refresh_table()
            self.tbl.selectRow(r)

    def _request_coverage(self):
        selected = []
        for r in range(self.tbl.rowCount()):
            it = self.tbl.item(r, 0)
            if it and it.checkState() == Qt.Checked and r < len(self._gws):
                selected.append(self._gws[r])
        if not selected:
            selected = list(self._gws)
        if not selected:
            QMessageBox.information(self, "알림", "GW를 먼저 추가하세요.")
            return
        self.sig_coverage_requested.emit(selected, self._settings)

    def _open_detail(self):
            from ui.gw_node_detail_window import GWNodeDetailWindow

            # 선택된 GW 가져오기
            rows = list({i.row() for i in self.tbl.selectedItems()})
            if not rows:
                QMessageBox.information(self, "알림",
                    "GW를 먼저 선택하세요. (행 클릭)")
                return
            r = rows[0]
            if r >= len(self._gws):
                return

            gw       = self._gws[r]
            node_win = getattr(self.parent(), '_node_win', None)
            result   = getattr(self.parent(), '_result',   None)
            nodes    = node_win.get_nodes() if node_win else []

            if not nodes:
                QMessageBox.information(self, "알림",
                    "단말기를 먼저 추가하세요.")
                return

            dlg = GWNodeDetailWindow(gw, nodes, result, parent=self)
            dlg.show()

    def _request_analyze(self):
        if not self._gws:
            QMessageBox.information(self, "알림", "GW를 먼저 추가하세요.")
            return
        self.sig_coverage_analyze.emit(list(self._gws))

    def _open_linkbudget(self):
        from ui.linkbudget_window import LinkBudgetWindow
        spatial  = getattr(self.parent(), 'spatial', None)
        if spatial is None:
            QMessageBox.warning(self, "알림", "공간 데이터 로드 후 사용 가능합니다.")
            return
        node_win = getattr(self.parent(), '_node_win', None)
        nodes    = node_win.get_nodes() if node_win else []
        if not nodes:
            QMessageBox.information(self, "알림", "단말(Node)을 먼저 추가하세요.")
            return
        if not self._gws:
            QMessageBox.information(self, "알림", "GW를 먼저 추가하세요.")
            return
        dlg = LinkBudgetWindow(spatial, self._gws, nodes, parent=self)
        dlg.show()

    def _open_distance(self):
            from ui.distance_window import DistanceWindow
            spatial  = getattr(self.parent(), 'spatial', None)
            node_win = getattr(self.parent(), '_node_win', None)
            result   = getattr(self.parent(), '_result', None)
            nodes    = node_win.get_nodes() if node_win else []
            if not nodes:
                QMessageBox.information(self, "알림", "단말(Node)을 먼저 추가하세요.")
                return
            if not self._gws:
                QMessageBox.information(self, "알림", "GW를 먼저 추가하세요.")
                return
            dlg = DistanceWindow(self._gws, nodes, result, parent=self)
            dlg.show()

    def _open_profile(self):
        from ui.profile_window import ProfileWindow
        spatial  = getattr(self.parent(), 'spatial', None)
        if spatial is None:
            QMessageBox.warning(self, "알림", "공간 데이터 로드 후 사용 가능합니다.")
            return
        node_win = getattr(self.parent(), '_node_win', None)
        nodes    = node_win.get_nodes() if node_win else []
        if not nodes:
            QMessageBox.information(self, "알림",
                "단말(Node)을 먼저 추가하세요.")
            return
        if not self._gws:
            QMessageBox.information(self, "알림", "GW를 먼저 추가하세요.")
            return
        dlg = ProfileWindow(spatial, self._gws, nodes, parent=self)
        dlg.show()

    def _open_settings(self):
        dlg = CoverageSettingsDialog(self._settings, self)
        if dlg.exec_() == CoverageSettingsDialog.Accepted:
            self._settings = dlg.get_settings()

    def _import_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "GW CSV 가져오기", "", "CSV (*.csv)")
        if not path: return
        with open(path, newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                try:
                    self._gws.append(GWEntry(
                        callsign = row.get('callsign', 'GW'),
                        lon      = float(row.get('lon', row.get('longitude', 127.1))),
                        lat      = float(row.get('lat', row.get('latitude', 37.4))),
                        pt_dbm   = float(row.get('pt_dbm', 14)),
                        gt_dbi   = float(row.get('gt_dbi', 2.15)),
                        lt_db    = float(row.get('lt_db', 0)),
                        hb_m     = float(row.get('hb_m', 15)),
                    ))
                except Exception:
                    continue
        self._refresh_table()

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "GW CSV 내보내기", "gw_list.csv", "CSV (*.csv)")
        if not path: return
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['callsign', 'lon', 'lat', 'pt_dbm',
                        'gt_dbi', 'lt_db', 'hb_m'])
            for g in self._gws:
                w.writerow([g.callsign, g.lon, g.lat,
                            g.pt_dbm, g.gt_dbi, g.lt_db, g.hb_m])