# ui/distance_window.py
from __future__ import annotations
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QLabel, QComboBox, QAbstractItemView,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from ui.dialogs import DARK, PANEL, TEXT, MUTED, BORDER, STYLE_DLG
from core.coverage import GWEntry, NodeEntry

COLS = ['Node', '거리 (km)', '방위각 (°)', 'Pr (dBm)', '커버']

from core.utils import haversine, bearing


class NumericItem(QTableWidgetItem):
    """숫자 기준 정렬을 지원하는 QTableWidgetItem."""
    def __init__(self, text: str, sort_val=None):
        super().__init__(text)
        if sort_val is not None:
            self._val = sort_val
        else:
            try:
                self._val = float(text.replace(',','').replace('°','').strip())
            except (ValueError, AttributeError):
                self._val = text

    def __lt__(self, other):
        if isinstance(other, NumericItem):
            try:
                return float(self._val) < float(other._val)
            except (TypeError, ValueError):
                return str(self._val) < str(other._val)
        return super().__lt__(other)


class DistanceWindow(QDialog):
    """GW ↔ Node 거리 분석 창."""

    def __init__(self, gws: list[GWEntry], nodes: list[NodeEntry],
                 result=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GW ↔ Node 거리 분석")
        self.setStyleSheet(STYLE_DLG)
        self.resize(700, 500)
        self.setWindowFlag(Qt.Window)

        self.gws    = gws
        self.nodes  = nodes
        self.result = result
        self._build()
        self._update_table()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        # GW 선택
        top = QHBoxLayout()
        top.addWidget(QLabel("GW 선택:"))
        self.cb_gw = QComboBox()
        self.cb_gw.setStyleSheet(
            f"QComboBox{{background:{PANEL};color:{TEXT};"
            f"border:1px solid {BORDER};border-radius:4px;"
            f"padding:4px 8px;min-height:26px;}}"
            f"QComboBox QAbstractItemView{{background:{PANEL};"
            f"color:{TEXT};selection-background-color:#253a5a;}}")
        for gw in self.gws:
            self.cb_gw.addItem(gw.callsign)
        top.addWidget(self.cb_gw)

        # 정렬 선택
        top.addWidget(QLabel("  정렬:"))
        self.cb_sort = QComboBox()
        self.cb_sort.setStyleSheet(self.cb_gw.styleSheet())
        self.cb_sort.addItems(['거리 순', 'Pr 높은 순', '커버 먼저', 'Node 이름 순'])
        top.addWidget(self.cb_sort)

        btn_ref = QPushButton("🔄 갱신")
        btn_ref.setStyleSheet(
            f"QPushButton{{background:#1c2a3a;color:#7ab8e8;"
            f"border:1px solid #2a4a6a;border-radius:4px;"
            f"padding:5px 12px;font-size:11px;}}"
            f"QPushButton:hover{{background:#254d78;}}")
        btn_ref.clicked.connect(self._update_table)
        top.addWidget(btn_ref)
        top.addStretch()
        lay.addLayout(top)

        # 요약 라벨
        self.lbl_summary = QLabel("")
        self.lbl_summary.setStyleSheet(
            f"color:{MUTED};font-size:11px;padding:2px 4px;")
        lay.addWidget(self.lbl_summary)

        # 테이블
        self.tbl = QTableWidget(0, len(COLS))
        self.tbl.setHorizontalHeaderLabels(COLS)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl.setAlternatingRowColors(True)
        self.tbl.setStyleSheet(
            f"QTableWidget{{background:{PANEL};color:{TEXT};"
            f"gridline-color:{BORDER};"
            f"alternate-background-color:#1a1d28;"
            f"selection-background-color:#253a5a;}}"
            f"QHeaderView::section{{background:{DARK};color:{MUTED};"
            f"border:none;padding:4px;}}")
        lay.addWidget(self.tbl)

        self.cb_gw.currentIndexChanged.connect(self._update_table)
        self.cb_sort.currentIndexChanged.connect(self._update_table)

    def update_data(self, gws, nodes, result=None):
        """외부에서 데이터 갱신."""
        self.gws    = gws
        self.nodes  = nodes
        self.result = result
        cur = self.cb_gw.currentText()
        self.cb_gw.clear()
        for gw in gws:
            self.cb_gw.addItem(gw.callsign)
        # 이전 선택 복원
        idx = self.cb_gw.findText(cur)
        if idx >= 0:
            self.cb_gw.setCurrentIndex(idx)
        self._update_table()

    def _update_table(self):
        gi = self.cb_gw.currentIndex()
        if gi < 0 or gi >= len(self.gws):
            return
        gw = self.gws[gi]

        # 각 Node 데이터 계산
        rows = []
        for ni, nd in enumerate(self.nodes):
            dist = haversine(gw.lon, gw.lat, nd.lon, nd.lat)
            brg  = bearing(gw.lon, gw.lat, nd.lon, nd.lat)

            # 커버 여부 및 Pr
            cov = False
            pr  = -999.0
            if self.result and ni < len(self.result.nodes):
                info = self.result.nodes[ni]
                cov  = info.covered
                pr   = info.gw_prs.get(gw.callsign, info.best_pr)

            rows.append({
                'callsign': nd.callsign,
                'dist'    : dist,
                'brg'     : brg,
                'pr'      : pr,
                'cov'     : cov,
            })

        # 정렬
        sort_idx = self.cb_sort.currentIndex()
        if sort_idx == 0:
            rows.sort(key=lambda x: x['dist'])
        elif sort_idx == 1:
            rows.sort(key=lambda x: -x['pr'])
        elif sort_idx == 2:
            rows.sort(key=lambda x: (not x['cov'], x['dist']))
        else:
            rows.sort(key=lambda x: x['callsign'])

        # 테이블 채우기
        self.tbl.setRowCount(0)
        n_cov = sum(1 for r in rows if r['cov'])

        for row in rows:
            r = self.tbl.rowCount()
            self.tbl.insertRow(r)

            items = [
                row['callsign'],
                f"{row['dist']:.3f}",
                f"{row['brg']:.1f}°",
                f"{row['pr']:.1f}" if row['pr'] > -999 else "─",
                "✓ 커버" if row['cov'] else "✗ 미커버",
            ]
            _num_cols = {1, 2, 3}  # 거리, 방위각, Pr
            for c, v in enumerate(items):
                if c in _num_cols:
                    it = NumericItem(str(v))
                else:
                    it = QTableWidgetItem(str(v))
                it.setTextAlignment(Qt.AlignCenter)
                self.tbl.setItem(r, c, it)

            # 색상
            if row['cov']:
                self.tbl.item(r, 4).setForeground(QColor('#00C94A'))
                self.tbl.item(r, 4).setBackground(QColor('#0d2010'))
            else:
                self.tbl.item(r, 4).setForeground(QColor('#FF4444'))
                self.tbl.item(r, 4).setBackground(QColor('#200d0d'))

            # Pr 색상
            if row['pr'] > -999:
                pr_col = ('#00C94A' if row['pr'] >= -110
                          else '#FFD700' if row['pr'] >= -126.6
                          else '#FF4444')
                self.tbl.item(r, 3).setForeground(QColor(pr_col))

        # 요약
        total = len(rows)
        if total > 0:
            avg_dist = np.mean([r['dist'] for r in rows])
            min_dist = min(r['dist'] for r in rows)
            max_dist = max(r['dist'] for r in rows)
            self.lbl_summary.setText(
                f"GW: {gw.callsign}  |  "
                f"Node {total}개  |  "
                f"커버: {n_cov}/{total} ({n_cov/total*100:.1f}%)  |  "
                f"거리 최소 {min_dist:.2f}km  평균 {avg_dist:.2f}km  최대 {max_dist:.2f}km")