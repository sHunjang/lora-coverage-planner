# ui/linkbudget_window.py — 링크 버짓 상세 창
from __future__ import annotations
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget,
    QTableWidgetItem, QHeaderView, QLabel, QComboBox,
    QPushButton, QAbstractItemView, QGroupBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from ui.dialogs import DARK, PANEL, TEXT, MUTED, BORDER, STYLE_DLG
from core.coverage import GWEntry, NodeEntry

STYLE = STYLE_DLG

# SF별 최소 수신 감도 (dBm) — LoRa 125kHz 기준
SF_SENS = {
    7:  -123.0,
    8:  -126.0,
    9:  -129.0,
    10: -132.0,
    11: -134.5,
    12: -137.0,
}


def calc_link_budget(gw: GWEntry, nd: NodeEntry,
                     pl: float) -> dict:
    """
    링크 버짓 전체 계산.
    pl: PathLossModel로 계산된 경로 손실 (dB)
    """
    # 수신 전력
    pr = gw.pt_dbm + gw.gt_dbi - gw.lt_db - pl + nd.gr_dbi - nd.lr_db

    # SF별 마진
    margins = {sf: pr - sens for sf, sens in SF_SENS.items()}

    # 최대 커버 가능 SF
    max_sf = None
    for sf in sorted(SF_SENS.keys()):
        if margins[sf] >= 0:
            max_sf = sf

    return {
        'pt'     : gw.pt_dbm,
        'gt'     : gw.gt_dbi,
        'lt'     : gw.lt_db,
        'pl'     : pl,
        'gr'     : nd.gr_dbi,
        'lr'     : nd.lr_db,
        'pr'     : pr,
        'margins': margins,
        'max_sf' : max_sf,
        'min_rx' : nd.min_rx_dbm,
        'margin_total': pr - nd.min_rx_dbm,
    }


class LinkBudgetWindow(QDialog):
    """GW ↔ Node 링크 버짓 상세 창."""

    def __init__(self, spatial, gws: list[GWEntry],
                 nodes: list[NodeEntry], parent=None):
        super().__init__(parent)
        self.setWindowTitle("링크 버짓 (Link Budget)")
        self.setStyleSheet(STYLE)
        self.resize(680, 580)
        self.setWindowFlag(Qt.Window)

        self.spatial = spatial
        self.gws     = [g for g in gws if g.enabled]
        self.nodes   = nodes
        self._cache  = {}   # (gi, ni) → budget dict
        self._build()
        self._calc_all()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        # 선택
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("GW:"))
        self.cb_gw = QComboBox()
        for g in self.gws: self.cb_gw.addItem(g.callsign)
        ctrl.addWidget(self.cb_gw)

        ctrl.addWidget(QLabel("  Node:"))
        self.cb_nd = QComboBox()
        for n in self.nodes: self.cb_nd.addItem(n.callsign)
        ctrl.addWidget(self.cb_nd)

        btn = QPushButton("계산")
        btn.setStyleSheet(
            "QPushButton{background:#1c3a5a;color:#7ab8e8;"
            "border:1px solid #2a5a8a;border-radius:4px;"
            "padding:5px 14px;font-size:11px;}"
            "QPushButton:hover{background:#254d78;}")
        btn.clicked.connect(self._show_selected)
        ctrl.addWidget(btn)
        ctrl.addStretch()
        lay.addLayout(ctrl)

        # 링크 버짓 테이블
        grp1 = QGroupBox("링크 버짓 상세")
        grp1.setStyleSheet(
            f"QGroupBox{{color:{MUTED};border:1px solid {BORDER};"
            f"border-radius:6px;margin-top:6px;padding-top:8px;}}"
            f"QGroupBox::title{{subcontrol-origin:margin;left:8px;}}")
        g1l = QVBoxLayout(grp1)

        self.tbl_budget = QTableWidget(0, 3)
        self.tbl_budget.setHorizontalHeaderLabels(['항목', '값', '단위'])
        self.tbl_budget.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.tbl_budget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_budget.verticalHeader().setVisible(False)
        self.tbl_budget.setStyleSheet(
            f"QTableWidget{{background:{PANEL};color:{TEXT};"
            f"gridline-color:{BORDER};}}"
            f"QHeaderView::section{{background:{DARK};color:{MUTED};"
            f"border:none;padding:4px;}}")
        g1l.addWidget(self.tbl_budget)
        lay.addWidget(grp1, 2)

        # SF 마진 테이블
        grp2 = QGroupBox("SF별 수신 마진")
        grp2.setStyleSheet(grp1.styleSheet())
        g2l = QVBoxLayout(grp2)

        self.tbl_sf = QTableWidget(1, len(SF_SENS))
        self.tbl_sf.setHorizontalHeaderLabels(
            [f"SF{sf}" for sf in sorted(SF_SENS.keys())])
        self.tbl_sf.setVerticalHeaderLabels(["마진 (dB)"])
        self.tbl_sf.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.tbl_sf.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_sf.setFixedHeight(72)
        self.tbl_sf.setStyleSheet(self.tbl_budget.styleSheet())
        g2l.addWidget(self.tbl_sf)
        lay.addWidget(grp2)

        # 전체 Matrix 테이블 (GW × Node)
        grp3 = QGroupBox("전체 링크 매트릭스 (Pr dBm)")
        grp3.setStyleSheet(grp1.styleSheet())
        g3l = QVBoxLayout(grp3)

        self.tbl_matrix = QTableWidget(
            len(self.nodes), len(self.gws))
        self.tbl_matrix.setHorizontalHeaderLabels(
            [g.callsign for g in self.gws])
        self.tbl_matrix.setVerticalHeaderLabels(
            [n.callsign for n in self.nodes])
        self.tbl_matrix.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.tbl_matrix.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_matrix.setStyleSheet(self.tbl_budget.styleSheet())
        self.tbl_matrix.cellClicked.connect(self._on_matrix_click)
        g3l.addWidget(self.tbl_matrix)
        lay.addWidget(grp3, 3)

        self.lbl = QLabel("")
        self.lbl.setStyleSheet(f"color:{MUTED};font-size:11px;")
        lay.addWidget(self.lbl)

    def _calc_all(self):
        """모든 GW × Node 조합 Pr 계산."""
        from core.propagation import PathLossModel
        self.lbl.setText("계산 중...")
        QApplication_processEvents()

        for gi, gw in enumerate(self.gws):
            gx, gy = self.spatial.lonlat_to_xy(gw.lon, gw.lat)
            for ni, nd in enumerate(self.nodes):
                nx, ny = self.spatial.lonlat_to_xy(nd.lon, nd.lat)
                model  = PathLossModel(
                    self.spatial, h_station=nd.hm_m,
                    hb_gw=gw.hb_m, env=2, fc=915.0, n_samples=100)
                pl = model.path_loss(
                    float(gx), float(gy), float(nx), float(ny))
                bud = calc_link_budget(gw, nd, pl)
                self._cache[(gi, ni)] = bud

                # 매트릭스 셀
                pr  = bud['pr']
                cov = pr >= nd.min_rx_dbm
                itm = QTableWidgetItem(f"{pr:.1f}")
                itm.setTextAlignment(Qt.AlignCenter)
                if cov:
                    itm.setBackground(QColor('#1d3a1d'))
                    itm.setForeground(QColor('#7ae87a'))
                else:
                    itm.setBackground(QColor('#3a1a1a'))
                    itm.setForeground(QColor('#e87a7a'))
                self.tbl_matrix.setItem(ni, gi, itm)

        self.lbl.setText("계산 완료. 매트릭스 셀 클릭 시 상세 보기.")
        self._show_selected()

    def _show_selected(self):
        gi = self.cb_gw.currentIndex()
        ni = self.cb_nd.currentIndex()
        bud = self._cache.get((gi, ni))
        if not bud:
            return
        self._fill_budget(bud)
        self._fill_sf(bud)

    def _on_matrix_click(self, row, col):
        bud = self._cache.get((col, row))
        if bud:
            self._fill_budget(bud)
            self._fill_sf(bud)
            self.cb_gw.setCurrentIndex(col)
            self.cb_nd.setCurrentIndex(row)

    def _fill_budget(self, bud: dict):
        rows = [
            ("송신 출력 Pt",        f"{bud['pt']:+.1f}", "dBm"),
            ("GW 안테나 이득 Gt",   f"{bud['gt']:+.2f}", "dBi"),
            ("GW 케이블 손실 Lt",   f"{-bud['lt']:.2f}", "dB"),
            ("경로 손실 PL",        f"{-bud['pl']:.1f}", "dB"),
            ("Node 안테나 이득 Gr", f"{bud['gr']:+.2f}", "dBi"),
            ("Node 수신 손실 Lr",   f"{-bud['lr']:.2f}", "dB"),
            ("─" * 18,             "─" * 8,             "─" * 4),
            ("수신 전력 Pr",        f"{bud['pr']:+.1f}", "dBm"),
            ("최소 수신 레벨",      f"{bud['min_rx']:+.1f}", "dBm"),
            ("링크 마진",           f"{bud['margin_total']:+.1f}", "dB"),
            ("최대 커버 가능 SF",   f"SF{bud['max_sf']}" if bud['max_sf'] else "불가", ""),
        ]
        self.tbl_budget.setRowCount(len(rows))
        for r, (name, val, unit) in enumerate(rows):
            self.tbl_budget.setItem(r, 0, QTableWidgetItem(name))
            vi = QTableWidgetItem(val)
            vi.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.tbl_budget.setItem(r, 1, vi)
            self.tbl_budget.setItem(r, 2, QTableWidgetItem(unit))

            # 마진 색상
            if name == "링크 마진":
                m = bud['margin_total']
                color = '#7ae87a' if m >= 6 else '#FFD700' if m >= 0 else '#e87a7a'
                vi.setForeground(QColor(color))
            if name == "수신 전력 Pr":
                vi.setForeground(QColor('#7ab8e8'))

    def _fill_sf(self, bud: dict):
        for c, sf in enumerate(sorted(SF_SENS.keys())):
            m   = bud['margins'][sf]
            itm = QTableWidgetItem(f"{m:+.1f}")
            itm.setTextAlignment(Qt.AlignCenter)
            if m >= 6:
                itm.setBackground(QColor('#1d3a1d'))
                itm.setForeground(QColor('#7ae87a'))
            elif m >= 0:
                itm.setBackground(QColor('#2a2a1a'))
                itm.setForeground(QColor('#FFD700'))
            else:
                itm.setBackground(QColor('#3a1a1a'))
                itm.setForeground(QColor('#e87a7a'))
            self.tbl_sf.setItem(0, c, itm)


def QApplication_processEvents():
    from PyQt5.QtWidgets import QApplication
    QApplication.processEvents()
