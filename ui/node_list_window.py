# ui/node_list_window.py
from __future__ import annotations
import csv
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QFileDialog, QLabel, QMessageBox,
    QSpinBox, QFormLayout, QGroupBox,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from core.coverage import NodeEntry
from ui.dialogs import NodeParamDialog, DARK, PANEL, TEXT, MUTED, BORDER, STYLE_DLG

# SF 감도 기준
SF_SENS = {
    7: -123.0, 8: -126.0, 9: -129.0,
    10: -132.0, 11: -134.5, 12: -137.0,
}
SF_COLORS = {
    7: '#FF4444', 8: '#FF8C00', 9: '#FFD700',
    10: '#00C94A', 11: '#4f8ef7', 12: '#9B59B6',
}

COLS = [
    'Callsign', '경도', '위도', 'Gr(dBi)', 'Lr(dB)', '높이(m)',
    '최소수신(dBm)', '연결 GW', '수신전력(dBm)', 'SF 등급', '상태'
]

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
BTN_PURPLE = ("QPushButton{background:#2a1a3a;color:#c87ae8;"
              "border:1px solid #5a2a8a;border-radius:4px;"
              "padding:5px 12px;font-size:11px;}"
              "QPushButton:hover{background:#3a2050;}")


def _pr_to_sf(pr: float) -> str:
    """수신전력으로 SF 등급 판별."""
    if pr <= -999:
        return "─"
    for sf in sorted(SF_SENS.keys()):
        if pr >= SF_SENS[sf]:
            return f"SF{sf}"
    return "불가"


# ── 랜덤 배치 다이얼로그 ─────────────────────────────────────
class RandomPlaceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Node 랜덤 배치")
        self.setStyleSheet(STYLE_DLG)
        self.setFixedWidth(360)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(10)

        grp = QGroupBox("랜덤 배치 설정")
        grp.setStyleSheet(
            f"QGroupBox{{color:{MUTED};border:1px solid {BORDER};"
            f"border-radius:6px;margin-top:6px;padding-top:8px;}}"
            f"QGroupBox::title{{subcontrol-origin:margin;left:8px;}}")
        fl = QFormLayout(grp); fl.setSpacing(8)

        self.sp_count = QSpinBox()
        self.sp_count.setRange(1, 1000)
        self.sp_count.setValue(50)
        self.sp_count.setStyleSheet(
            f"QSpinBox{{background:{PANEL};color:{TEXT};"
            f"border:1px solid {BORDER};border-radius:4px;"
            f"padding:4px 6px;min-height:26px;}}")

        self.sp_seed = QSpinBox()
        self.sp_seed.setRange(0, 9999)
        self.sp_seed.setValue(42)
        self.sp_seed.setStyleSheet(self.sp_count.styleSheet())

        fl.addRow("Node 개수", self.sp_count)
        fl.addRow("랜덤 시드", self.sp_seed)
        lay.addWidget(grp)

        info = QLabel("성남시 경계 내에 균일하게 랜덤 배치됩니다.\n기존 Node 목록에 추가됩니다.")
        info.setStyleSheet(f"color:{MUTED};font-size:11px;")
        lay.addWidget(info)

        btn_row = QHBoxLayout()
        ok  = QPushButton("배치")
        ok.setStyleSheet(
            f"QPushButton{{background:#1d4a1d;color:#7ae87a;"
            f"border:1px solid #2a6a2a;border-radius:4px;"
            f"padding:6px 18px;font-size:12px;}}"
            f"QPushButton:hover{{background:#256a25;}}")
        cxl = QPushButton("취소")
        cxl.setStyleSheet(
            f"QPushButton{{background:#3a1a1a;color:#e87a7a;"
            f"border:1px solid #5a2a2a;border-radius:4px;"
            f"padding:6px 18px;font-size:12px;}}"
            f"QPushButton:hover{{background:#5a2020;}}")
        ok.clicked.connect(self.accept)
        cxl.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(cxl)
        btn_row.addWidget(ok)
        lay.addLayout(btn_row)

    def get_params(self):
        return self.sp_count.value(), self.sp_seed.value()


# ── Node 리스트 창 ───────────────────────────────────────────
class NodeListWindow(QDialog):
    sig_map_refresh = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Node(단말) 목록")
        self.setStyleSheet(STYLE_DLG)
        self.resize(1200, 480)
        self.setWindowFlag(Qt.Window)
        self._nodes: list[NodeEntry] = []
        self._result = None
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)

        top = QHBoxLayout()
        self.btn_add    = QPushButton("+ Node 추가");     self.btn_add.setStyleSheet(BTN_GREEN)
        self.btn_del    = QPushButton("- 선택 삭제");     self.btn_del.setStyleSheet(BTN_RED)
        self.btn_rnd    = QPushButton("🎲 랜덤 배치");    self.btn_rnd.setStyleSheet(BTN_PURPLE)
        self.btn_detail = QPushButton("📋 연결 GW 보기"); self.btn_detail.setStyleSheet(BTN)
        self.btn_imp    = QPushButton("CSV 가져오기");     self.btn_imp.setStyleSheet(BTN)
        self.btn_exp    = QPushButton("CSV 내보내기");     self.btn_exp.setStyleSheet(BTN)
        self.btn_clr_all = QPushButton("✕ 전체 삭제");   self.btn_clr_all.setStyleSheet(BTN_RED)

        for b in [self.btn_add, self.btn_del, self.btn_rnd,
                  self.btn_detail, self.btn_imp, self.btn_exp, self.btn_clr_all]:
            top.addWidget(b)
        top.addStretch()
        lay.addLayout(top)

        self.tbl = QTableWidget(0, len(COLS))
        self.tbl.setHorizontalHeaderLabels(COLS)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl.setStyleSheet(
            f"QTableWidget{{background:{PANEL};color:{TEXT};"
            f"gridline-color:{BORDER};selection-background-color:#253a5a;}}"
            f"QHeaderView::section{{background:{DARK};color:{MUTED};"
            f"border:none;padding:4px;}}")
        self.tbl.doubleClicked.connect(self._on_double_click)
        lay.addWidget(self.tbl)

        bot = QHBoxLayout()
        self.lbl_count = QLabel("Node: 0개")
        self.lbl_count.setStyleSheet(f"color:{MUTED};font-size:11px;")
        bot.addWidget(self.lbl_count)
        bot.addStretch()
        hint = QLabel("더블클릭: 파라미터 편집  |  SF 등급: 수신전력 기준 자동 계산")
        hint.setStyleSheet(f"color:{MUTED};font-size:11px;")
        bot.addWidget(hint)
        lay.addLayout(bot)

        self.btn_add.clicked.connect(self._add_default)
        self.btn_del.clicked.connect(self._del_selected)
        self.btn_rnd.clicked.connect(self._random_place)
        self.btn_detail.clicked.connect(self._open_detail)
        self.btn_imp.clicked.connect(self._import_csv)
        self.btn_exp.clicked.connect(self._export_csv)
        self.btn_clr_all.clicked.connect(self._clear_all)

    def _refresh_table(self, suppress_map=False):
        self.tbl.setRowCount(0)
        for ni, nd in enumerate(self._nodes):
            r = self.tbl.rowCount()
            self.tbl.insertRow(r)

            best_gw      = "─"
            best_pr      = "─"
            sf_grade     = "─"
            status       = "─"
            status_color = None
            pr_color     = None
            sf_color     = None

            if self._result and ni < len(self._result.nodes):
                info = self._result.nodes[ni]
                if info.best_gw:
                    best_gw = info.best_gw
                if info.best_pr > -999:
                    best_pr  = f"{info.best_pr:.1f}"
                    sf_grade = _pr_to_sf(info.best_pr)
                    # 수신전력 색상
                    if info.best_pr >= -90:
                        pr_color = '#FF4444'
                    elif info.best_pr >= -100:
                        pr_color = '#FF8C00'
                    elif info.best_pr >= -110:
                        pr_color = '#FFD700'
                    elif info.best_pr >= -120:
                        pr_color = '#00C94A'
                    else:
                        pr_color = '#4f8ef7'
                    # SF 색상
                    for sf in sorted(SF_SENS.keys()):
                        if info.best_pr >= SF_SENS[sf]:
                            sf_color = SF_COLORS[sf]
                            break
                if info.covered:
                    status       = "✓ 커버"
                    status_color = "#00C94A"
                else:
                    status       = "✗ 미커버"
                    status_color = "#FF4444"

            values = [
                nd.callsign,
                f"{nd.lon:.6f}",
                f"{nd.lat:.6f}",
                f"{nd.gr_dbi:.2f}",
                f"{nd.lr_db:.2f}",
                f"{nd.hm_m:.1f}",
                f"{nd.min_rx_dbm:.1f}",
                best_gw,
                best_pr,
                sf_grade,
                status,
            ]

            # COLS 인덱스 매핑
            COL_PR     = 8
            COL_SF     = 9
            COL_STATUS = 10

            for c, v in enumerate(values):
                it = QTableWidgetItem(str(v))
                it.setTextAlignment(Qt.AlignCenter)
                if c == COL_PR and pr_color:
                    it.setForeground(QColor(pr_color))
                if c == COL_SF and sf_color:
                    it.setForeground(QColor(sf_color))
                    it.setBackground(QColor(sf_color + '22'))
                if c == COL_STATUS and status_color:
                    it.setForeground(QColor(status_color))
                    bg = '#0d2010' if status_color == '#00C94A' else '#200d0d'
                    it.setBackground(QColor(bg))
                self.tbl.setItem(r, c, it)

        self.lbl_count.setText(f"Node: {len(self._nodes)}개")
        if not suppress_map:
            self.sig_map_refresh.emit()

    def update_result(self, result):
        self._result = result
        self._refresh_table(suppress_map=True)

    def add_node(self, nd: NodeEntry):
        self._nodes.append(nd)
        self._refresh_table()

    def set_coord(self, lon: float, lat: float):
        rows = list({i.row() for i in self.tbl.selectedItems()})
        if not rows: return
        r = rows[0]
        if r < len(self._nodes):
            n = self._nodes[r]
            self._nodes[r] = NodeEntry(n.callsign, lon, lat,
                                        n.gr_dbi, n.lr_db,
                                        n.hm_m, n.min_rx_dbm)
            self._refresh_table()
            self.tbl.selectRow(r)

    def get_nodes(self) -> list[NodeEntry]:
        return list(self._nodes)

    def _add_default(self):
        n = len(self._nodes) + 1
        p = self.parent()
        s = getattr(p, '_settings', {})
        self._nodes.append(NodeEntry(
            callsign       = f"Node{n}",
            lon            = 127.10, lat=37.40,
            gr_dbi         = s.get('nd_gr_dbi', 2.15),
            lr_db          = s.get('nd_lr_db',  0.0),
            hm_m           = s.get('nd_hm_m',   1.5),
            min_rx_dbm     = s.get('nd_min_rx', -126.6),
            indoor_loss_db = s.get('nd_indoor_loss', 0.0),
        ))
        self._refresh_table()

    def _del_selected(self):
        rows = sorted({i.row() for i in self.tbl.selectedItems()}, reverse=True)
        for r in rows:
            if r < len(self._nodes): self._nodes.pop(r)
        self._refresh_table()

    def _clear_all(self):
        if not self._nodes:
            return
        ret = QMessageBox.question(
            self, "전체 삭제",
            f"Node {len(self._nodes)}개를 모두 삭제하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No)
        if ret == QMessageBox.Yes:
            self._nodes.clear()
            self._refresh_table()

    def _open_detail(self):
        from ui.node_gw_detail_window import NodeGWDetailWindow
        rows = list({i.row() for i in self.tbl.selectedItems()})
        if not rows:
            QMessageBox.information(self, "알림", "Node를 먼저 선택하세요. (행 클릭)")
            return
        r = rows[0]
        if r >= len(self._nodes):
            return
        nd     = self._nodes[r]
        gw_win = getattr(self.parent(), '_gw_win',  None)
        result = getattr(self.parent(), '_result',  None)
        gws    = gw_win.get_gws() if gw_win else []
        if not gws:
            QMessageBox.information(self, "알림", "GW를 먼저 추가하세요.")
            return
        dlg = NodeGWDetailWindow(nd, gws, r, result, parent=self)
        dlg.show()

    def _random_place(self):
        dlg = RandomPlaceDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        count, seed = dlg.get_params()

        spatial = None
        p = self.parent()
        while p is not None:
            if hasattr(p, 'spatial'):
                spatial = p.spatial
                break
            p = p.parent() if hasattr(p, 'parent') else None

        if spatial is None:
            QMessageBox.warning(self, "오류", "공간 데이터가 로드되지 않았습니다.")
            return

        try:
            from shapely.geometry import MultiPoint
            b = [float(x) for x in spatial.bounds]
            poly = spatial.polygon_4326
            s    = getattr(self.parent(), '_settings', {})
            np.random.seed(seed)
            lon_list, lat_list = [], []

            self.status_msg("랜덤 배치 중...")
            while len(lon_list) < count:
                batch = max(count * 3, 200)
                lons  = b[0] + (b[2] - b[0]) * np.random.rand(batch)
                lats  = b[1] + (b[3] - b[1]) * np.random.rand(batch)
                pts   = MultiPoint(list(zip(lons, lats)))
                mask  = np.array([poly.contains(pt) for pt in pts.geoms])
                lon_list.extend(lons[mask].tolist())
                lat_list.extend(lats[mask].tolist())

            lon_arr = np.array(lon_list[:count])
            lat_arr = np.array(lat_list[:count])
            start_n = len(self._nodes)

            for i in range(count):
                self._nodes.append(NodeEntry(
                    callsign       = f"Node{start_n + i + 1}",
                    lon            = float(lon_arr[i]),
                    lat            = float(lat_arr[i]),
                    gr_dbi         = s.get('nd_gr_dbi', 2.15),
                    lr_db          = s.get('nd_lr_db',  0.0),
                    hm_m           = s.get('nd_hm_m',   1.5),
                    min_rx_dbm     = s.get('nd_min_rx', -126.6),
                    indoor_loss_db = s.get('nd_indoor_loss', 0.0),
                ))

            self._refresh_table()
            QMessageBox.information(
                self, "완료",
                f"Node {count}개가 성남시 내에 랜덤 배치되었습니다.")

        except Exception as e:
            QMessageBox.critical(self, "오류", f"랜덤 배치 실패:\n{e}")

    def status_msg(self, msg):
        parent = self.parent()
        if parent and hasattr(parent, 'status'):
            parent.status.showMessage(msg)

    def _on_double_click(self, idx):
        r = idx.row()
        if r >= len(self._nodes): return
        dlg = NodeParamDialog(self._nodes[r], self)
        if dlg.exec_() == NodeParamDialog.Accepted:
            self._nodes[r] = dlg.result_node(self._nodes[r])
            self._refresh_table()
            self.tbl.selectRow(r)

    def _import_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Node CSV", "", "CSV (*.csv)")
        if not path: return
        with open(path, newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                try:
                    self._nodes.append(NodeEntry(
                        callsign       = row.get('callsign', 'Node'),
                        lon            = float(row.get('lon', row.get('longitude', 127.1))),
                        lat            = float(row.get('lat', row.get('latitude', 37.4))),
                        gr_dbi         = float(row.get('gr_dbi', 2.15)),
                        lr_db          = float(row.get('lr_db', 0)),
                        hm_m           = float(row.get('hm_m', 1.5)),
                        min_rx_dbm     = float(row.get('min_rx_dbm', -126.6)),
                        indoor_loss_db = float(row.get('indoor_loss_db', 0.0)),
                    ))
                except Exception: continue
        self._refresh_table()

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Node CSV", "node_list.csv", "CSV (*.csv)")
        if not path: return
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow([
                'callsign', 'lon', 'lat', 'gr_dbi', 'lr_db',
                'hm_m', 'min_rx_dbm', 'indoor_loss_db',
                '연결 GW', '수신전력(dBm)', 'SF 등급', '상태'
            ])
            for ni, n in enumerate(self._nodes):
                best_gw  = ""
                best_pr  = ""
                sf_grade = ""
                status   = ""
                if self._result and ni < len(self._result.nodes):
                    info     = self._result.nodes[ni]
                    best_gw  = info.best_gw or ""
                    best_pr  = f"{info.best_pr:.1f}" if info.best_pr > -999 else ""
                    sf_grade = _pr_to_sf(info.best_pr) if info.best_pr > -999 else ""
                    status   = "커버" if info.covered else "미커버"
                w.writerow([
                    n.callsign, n.lon, n.lat,
                    n.gr_dbi, n.lr_db, n.hm_m,
                    n.min_rx_dbm,
                    getattr(n, 'indoor_loss_db', 0.0),
                    best_gw, best_pr, sf_grade, status,
                ])