# ui/dialogs.py — 파라미터 편집 다이얼로그들
from __future__ import annotations
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QDoubleSpinBox, QPushButton, QLabel,
    QGroupBox, QColorDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QSpinBox, QWidget,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from core.coverage import GWEntry, NodeEntry

DARK = "#181b22"
PANEL = "#1e2130"
TEXT = "#e0e4ef"
MUTED = "#7a8099"
BORDER = "#2a2f3b"
ACCENT = "#4f8ef7"

STYLE_DLG = f"""
    QDialog {{ background:{DARK}; color:{TEXT}; }}
    QLabel {{ color:{TEXT}; }}
    QLineEdit, QDoubleSpinBox, QSpinBox {{
        background:{PANEL}; color:{TEXT};
        border:1px solid {BORDER}; border-radius:4px;
        padding:4px 6px; min-height:26px;
    }}
    QGroupBox {{
        color:{MUTED}; border:1px solid {BORDER};
        border-radius:6px; margin-top:8px; padding-top:8px;
    }}
    QGroupBox::title {{ subcontrol-origin:margin; left:8px; }}
    QPushButton {{
        background:#253a5a; color:{TEXT};
        border:1px solid #3a5a8a; border-radius:5px;
        padding:6px 18px; font-size:12px;
    }}
    QPushButton:hover {{ background:#2e4a7a; }}
    QPushButton[role="ok"] {{ background:#1d4a1d; border-color:#2a6a2a; }}
    QPushButton[role="ok"]:hover {{ background:#256a25; }}
    QPushButton[role="cancel"] {{ background:#3a1a1a; border-color:#6a2a2a; }}
"""


def _dspin(lo, hi, val, dec=2, suffix="", step=0.1):
    s = QDoubleSpinBox()
    s.setRange(lo, hi); s.setValue(val)
    s.setDecimals(dec); s.setSuffix(suffix); s.setSingleStep(step)
    return s


# ── GW 파라미터 편집 ────────────────────────────────────────
class GWParamDialog(QDialog):
    def __init__(self, gw: GWEntry, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"GW 파라미터 — {gw.callsign}")
        self.setStyleSheet(STYLE_DLG)
        self.setMinimumWidth(380)
        self._build(gw)

    def _build(self, gw: GWEntry):
        lay = QVBoxLayout(self); lay.setSpacing(10)

        grp = QGroupBox("GW 파라미터")
        fl  = QFormLayout(grp); fl.setSpacing(8)

        self.e_cs  = QLineEdit(gw.callsign)
        self.e_lon = QLineEdit(f"{gw.lon:.6f}")
        self.e_lat = QLineEdit(f"{gw.lat:.6f}")
        self.sp_pt = _dspin(-30, 50, gw.pt_dbm, 1, " dBm")
        self.sp_gt = _dspin(0, 30, gw.gt_dbi, 2, " dBi")
        self.sp_lt = _dspin(0, 20, gw.lt_db,  2, " dB")
        self.sp_fc = _dspin(100, 2000, 915.0,  1, " MHz", 1.0)
        self.sp_bw = _dspin(1, 500, 125.0, 1, " kHz", 1.0)
        self.sp_hb = _dspin(1, 200, gw.hb_m,  1, " m")

        fl.addRow("Callsign",        self.e_cs)
        fl.addRow("경도 (Longitude)", self.e_lon)
        fl.addRow("위도 (Latitude)",  self.e_lat)
        fl.addRow("송신 출력 Pt",     self.sp_pt)
        fl.addRow("안테나 이득 Gt",   self.sp_gt)
        fl.addRow("케이블 손실 Lt",   self.sp_lt)
        fl.addRow("주파수",           self.sp_fc)
        fl.addRow("대역폭",           self.sp_bw)
        fl.addRow("안테나 높이",      self.sp_hb)
        lay.addWidget(grp)

        btn_l = QHBoxLayout()
        ok  = QPushButton("확인"); ok.setProperty("role","ok")
        cxl = QPushButton("취소"); cxl.setProperty("role","cancel")
        ok.clicked.connect(self.accept)
        cxl.clicked.connect(self.reject)
        btn_l.addStretch(); btn_l.addWidget(cxl); btn_l.addWidget(ok)
        lay.addLayout(btn_l)

    def result_gw(self, orig: GWEntry) -> GWEntry:
        return GWEntry(
            callsign = self.e_cs.text(),
            lon      = float(self.e_lon.text()),
            lat      = float(self.e_lat.text()),
            pt_dbm   = self.sp_pt.value(),
            gt_dbi   = self.sp_gt.value(),
            lt_db    = self.sp_lt.value(),
            hb_m     = self.sp_hb.value(),
            enabled  = orig.enabled,
        )


# ── Node 파라미터 편집 ───────────────────────────────────────
class NodeParamDialog(QDialog):
    def __init__(self, nd: NodeEntry, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Node 파라미터 — {nd.callsign}")
        self.setStyleSheet(STYLE_DLG)
        self.setMinimumWidth(380)
        self._build(nd)

    def _build(self, nd: NodeEntry):
        lay = QVBoxLayout(self); lay.setSpacing(10)

        grp = QGroupBox("단말(Node) 파라미터")
        fl  = QFormLayout(grp); fl.setSpacing(8)

        self.e_cs     = QLineEdit(nd.callsign)
        self.e_lon    = QLineEdit(f"{nd.lon:.6f}")
        self.e_lat    = QLineEdit(f"{nd.lat:.6f}")
        self.sp_gr    = _dspin(0, 30,   nd.gr_dbi,       2, " dBi")
        self.sp_lr    = _dspin(0, 20,   nd.lr_db,        2, " dB")
        self.sp_fc    = _dspin(100, 2000, 915.0,          1, " MHz", 1.0)
        self.sp_bw    = _dspin(1, 500, 125.0,             1, " kHz", 1.0)
        self.sp_hm    = _dspin(0.1, 50, nd.hm_m,         1, " m")
        self.sp_rxm   = _dspin(-160, -50, nd.min_rx_dbm, 1, " dBm")
        self.sp_indoor= _dspin(0, 30,
                               getattr(nd, 'indoor_loss_db', 0.0),
                               1, " dB")

        fl.addRow("Callsign",          self.e_cs)
        fl.addRow("경도 (Longitude)",  self.e_lon)
        fl.addRow("위도 (Latitude)",   self.e_lat)
        fl.addRow("수신 이득 Gr",      self.sp_gr)
        fl.addRow("수신 손실 Lr",      self.sp_lr)
        fl.addRow("주파수",            self.sp_fc)
        fl.addRow("대역폭",            self.sp_bw)
        fl.addRow("안테나 높이",       self.sp_hm)
        fl.addRow("최소 수신 레벨",    self.sp_rxm)
        fl.addRow("실내 투과 손실",    self.sp_indoor)
        lay.addWidget(grp)

        note = QLabel("실내 투과 손실: 실외=0dB, 목조=5~10dB, 콘크리트=10~25dB")
        note.setStyleSheet(f"color:{MUTED};font-size:10px;")
        lay.addWidget(note)

        btn_l = QHBoxLayout()
        ok  = QPushButton("확인"); ok.setProperty("role","ok")
        cxl = QPushButton("취소"); cxl.setProperty("role","cancel")
        ok.clicked.connect(self.accept)
        cxl.clicked.connect(self.reject)
        btn_l.addStretch(); btn_l.addWidget(cxl); btn_l.addWidget(ok)
        lay.addLayout(btn_l)

    def result_node(self, orig: NodeEntry) -> NodeEntry:
        return NodeEntry(
            callsign      = self.e_cs.text(),
            lon           = float(self.e_lon.text()),
            lat           = float(self.e_lat.text()),
            gr_dbi        = self.sp_gr.value(),
            lr_db         = self.sp_lr.value(),
            hm_m          = self.sp_hm.value(),
            min_rx_dbm    = self.sp_rxm.value(),
            indoor_loss_db= self.sp_indoor.value(),
        )


# ── 커버리지 분석 설정 ───────────────────────────────────────
class CoverageSettingsDialog(QDialog):
    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("커버리지 분석 설정")
        self.setStyleSheet(STYLE_DLG)
        self.setMinimumWidth(420)
        self._build(settings)

    def _build(self, s: dict):
        lay = QVBoxLayout(self); lay.setSpacing(10)

        g1 = QGroupBox("분석 파라미터")
        fl = QFormLayout(g1); fl.setSpacing(8)
        self.sp_rxh  = _dspin(0.1, 50,   s.get('rx_height', 1.5), 1, " m")
        self.sp_rad  = _dspin(0.1, 50,   s.get('radius_km', 10.0),1, " km")
        self.sp_mrx  = _dspin(-160, -50, s.get('min_rx', -126.6),  1, " dBm")
        fl.addRow("수신 안테나 높이", self.sp_rxh)
        fl.addRow("분석 반경",        self.sp_rad)
        fl.addRow("최소 수신 레벨",   self.sp_mrx)
        lay.addWidget(g1)

        g2 = QGroupBox("커버리지 색상 (dBm 레벨별)")
        g2l = QVBoxLayout(g2)

        self.color_table = QTableWidget(0, 3)
        self.color_table.setHorizontalHeaderLabels(
            ['Pr 이상 (dBm)', '색상', ''])
        self.color_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.color_table.setStyleSheet(
            f"QTableWidget{{background:{PANEL};color:{TEXT};"
            f"gridline-color:{BORDER};}}"
            f"QHeaderView::section{{background:{DARK};color:{MUTED};"
            f"border:none;padding:4px;}}")
        self.color_table.setEditTriggers(QAbstractItemView.DoubleClicked)
        g2l.addWidget(self.color_table)

        defaults = s.get('color_levels', [
            {'pr': -90,  'color': '#FF2020'},
            {'pr': -100, 'color': '#FF8C00'},
            {'pr': -110, 'color': '#FFD700'},
            {'pr': -120, 'color': '#00C94A'},
            # {'pr': -130, 'color': '#4f8ef7'},                             
        ])
        for lv in defaults:
            self._add_color_row(lv['pr'], lv['color'])

        add_btn = QPushButton("+ 레벨 추가")
        add_btn.clicked.connect(lambda: self._add_color_row(-110, '#888888'))
        g2l.addWidget(add_btn)
        lay.addWidget(g2)

        btn_l = QHBoxLayout()
        ok  = QPushButton("확인"); ok.setProperty("role","ok")
        cxl = QPushButton("취소"); cxl.setProperty("role","cancel")
        ok.clicked.connect(self.accept)
        cxl.clicked.connect(self.reject)
        btn_l.addStretch(); btn_l.addWidget(cxl); btn_l.addWidget(ok)
        lay.addLayout(btn_l)

    def _add_color_row(self, pr: float, color: str):
        r = self.color_table.rowCount()
        self.color_table.insertRow(r)

        pr_item = QTableWidgetItem(str(pr))
        self.color_table.setItem(r, 0, pr_item)

        col_btn = QPushButton()
        col_btn.setStyleSheet(
            f"background:{color};border:1px solid {BORDER};"
            f"border-radius:3px;min-height:22px;")
        col_btn.setProperty("color", color)
        col_btn.clicked.connect(lambda _, b=col_btn: self._pick_color(b))
        self.color_table.setCellWidget(r, 1, col_btn)

        del_btn = QPushButton("✕")
        del_btn.setStyleSheet(
            f"background:#3a1a1a;color:#ff6060;"
            f"border:none;font-size:11px;")
        del_btn.clicked.connect(lambda _, row=r: self._del_row(row))
        self.color_table.setCellWidget(r, 2, del_btn)

    def _pick_color(self, btn: QPushButton):
        from PyQt5.QtWidgets import QColorDialog
        cur = btn.property("color") or "#ffffff"
        col = QColorDialog.getColor(QColor(cur), self, "색상 선택")
        if col.isValid():
            hex_c = col.name()
            btn.setProperty("color", hex_c)
            btn.setStyleSheet(
                f"background:{hex_c};border:1px solid {BORDER};"
                f"border-radius:3px;min-height:22px;")

    def _del_row(self, row: int):
        for r in range(self.color_table.rowCount()):
            w = self.color_table.cellWidget(r, 2)
            if w and not w.isEnabled():
                self.color_table.removeRow(r)
                return
        self.color_table.removeRow(row)

    def get_settings(self) -> dict:
        levels = []
        for r in range(self.color_table.rowCount()):
            try:
                pr  = float(self.color_table.item(r,0).text())
                btn = self.color_table.cellWidget(r, 1)
                col = btn.property("color") if btn else "#888888"
                levels.append({'pr': pr, 'color': col})
            except Exception:
                continue
        levels.sort(key=lambda x: -x['pr'])
        return {
            'rx_height'   : self.sp_rxh.value(),
            'radius_km'   : self.sp_rad.value(),
            'min_rx'      : self.sp_mrx.value(),
            'color_levels': levels,
        }