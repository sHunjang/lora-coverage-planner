# ui/gw_node_detail_window.py
from __future__ import annotations
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QGroupBox, QPushButton,
    QFileDialog,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
import csv
from ui.dialogs import DARK, PANEL, TEXT, MUTED, BORDER, STYLE_DLG
from core.coverage import GWEntry, NodeEntry

COLS = ['Callsign', '거리(km)', '방위각(°)', 'Pr(dBm)',
        '커버', '경도', '위도', '높이(m)', '최소수신(dBm)']


class NumericItem(QTableWidgetItem):
    """숫자 기준 정렬을 지원하는 QTableWidgetItem."""
    def __init__(self, text: str, sort_val=None):
        super().__init__(text)
        # sort_val이 없으면 텍스트에서 숫자 추출 시도
        if sort_val is not None:
            self._val = sort_val
        else:
            try:
                self._val = float(text.replace(',', '').strip())
            except (ValueError, AttributeError):
                self._val = text  # 문자열 그대로

    def __lt__(self, other):
        if isinstance(other, NumericItem):
            try:
                return float(self._val) < float(other._val)
            except (TypeError, ValueError):
                return str(self._val) < str(other._val)
        return super().__lt__(other)


def haversine(lon1, lat1, lon2, lat2) -> float:
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def bearing(lon1, lat1, lon2, lat2) -> float:
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlam = np.radians(lon2 - lon1)
    x = np.sin(dlam) * np.cos(phi2)
    y = np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(dlam)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


class GWNodeDetailWindow(QDialog):
    """GW에 연결된 단말기 상세 정보 창."""

    def __init__(self, gw: GWEntry, nodes: list[NodeEntry],
                 result=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{gw.callsign} — 연결 단말기 상세")
        self.setStyleSheet(STYLE_DLG)
        self.resize(900, 580)
        self.setWindowFlag(Qt.Window)

        self.gw     = gw
        self.nodes  = nodes
        self.result = result
        self._build()
        self._fill()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        # ── GW 정보 ─────────────────────────────────────────
        grp_gw = QGroupBox(f"Gateway 정보 — {self.gw.callsign}")
        grp_gw.setStyleSheet(
            f"QGroupBox{{color:#7ab8e8;border:1px solid {BORDER};"
            f"border-radius:6px;margin-top:6px;padding-top:8px;"
            f"font-weight:bold;}}"
            f"QGroupBox::title{{subcontrol-origin:margin;left:8px;}}")
        gw_lay = QHBoxLayout(grp_gw)

        def _info(label, val):
            w = QVBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color:{MUTED};font-size:10px;")
            val_lbl = QLabel(str(val))
            val_lbl.setStyleSheet(f"color:{TEXT};font-size:13px;font-weight:bold;")
            w.addWidget(lbl)
            w.addWidget(val_lbl)
            return w

        gw_lay.addLayout(_info("위도", f"{self.gw.lat:.6f}"))
        gw_lay.addLayout(_info("경도", f"{self.gw.lon:.6f}"))
        gw_lay.addLayout(_info("송신 출력", f"{self.gw.pt_dbm} dBm"))
        gw_lay.addLayout(_info("안테나 이득", f"{self.gw.gt_dbi} dBi"))
        gw_lay.addLayout(_info("케이블 손실", f"{self.gw.lt_db} dB"))
        gw_lay.addLayout(_info("안테나 높이", f"{self.gw.hb_m} m"))
        gw_lay.addStretch()
        lay.addWidget(grp_gw)

        # ── 요약 라벨 ────────────────────────────────────────
        self.lbl_summary = QLabel("")
        self.lbl_summary.setStyleSheet(
            f"color:{MUTED};font-size:12px;padding:4px 0;")
        lay.addWidget(self.lbl_summary)

        # ── 테이블 ───────────────────────────────────────────
        self.tbl = QTableWidget(0, len(COLS))
        self.tbl.setHorizontalHeaderLabels(COLS)
        hh = self.tbl.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Stretch)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl.setAlternatingRowColors(True)
        self.tbl.setSortingEnabled(True)
        self.tbl.setStyleSheet(
            f"QTableWidget{{background:{PANEL};color:{TEXT};"
            f"gridline-color:{BORDER};"
            f"alternate-background-color:#1a1d28;"
            f"selection-background-color:#253a5a;}}"
            f"QHeaderView::section{{background:{DARK};color:{MUTED};"
            f"border:none;padding:4px;}}")
        lay.addWidget(self.tbl)

        # ── 하단 버튼 ────────────────────────────────────────
        bot = QHBoxLayout()
        btn_exp = QPushButton("CSV 내보내기")
        btn_exp.setStyleSheet(
            f"QPushButton{{background:#1c2a3a;color:#7ab8e8;"
            f"border:1px solid #2a4a6a;border-radius:4px;"
            f"padding:5px 14px;font-size:11px;}}"
            f"QPushButton:hover{{background:#254d78;}}")
        btn_exp.clicked.connect(self._export_csv)
        bot.addStretch()
        bot.addWidget(btn_exp)
        lay.addLayout(bot)

    def _fill(self):
        """GW에 연결된 Node 목록 채우기."""
        gw    = self.gw
        nodes = self.nodes
        result = self.result

        # 이 GW에 연결된 Node만 필터
        connected = []
        for ni, nd in enumerate(nodes):
            cov = False
            pr  = -999.0
            is_mine = False  # 이 GW가 best_gw인지

            if result and ni < len(result.nodes):
                info    = result.nodes[ni]
                cov     = info.covered
                pr      = info.gw_prs.get(gw.callsign, -999.0)
                is_mine = info.best_gw == gw.callsign

            if is_mine or (cov and pr > -999):
                dist = haversine(gw.lon, gw.lat, nd.lon, nd.lat)
                brg  = bearing(gw.lon, gw.lat, nd.lon, nd.lat)
                connected.append({
                    'nd'     : nd,
                    'dist'   : dist,
                    'brg'    : brg,
                    'pr'     : pr,
                    'cov'    : cov,
                    'is_mine': is_mine,
                })

        # 거리 순 정렬
        connected.sort(key=lambda x: x['dist'])

        # 테이블 채우기
        self.tbl.setSortingEnabled(False)
        self.tbl.setRowCount(0)

        for row in connected:
            nd = row['nd']
            r  = self.tbl.rowCount()
            self.tbl.insertRow(r)

            pr_str = f"{row['pr']:.1f}" if row['pr'] > -999 else "─"
            items  = [
                nd.callsign,
                f"{row['dist']:.3f}",
                f"{row['brg']:.1f}",
                pr_str,
                "✓ 커버" if row['cov'] else "✗ 미커버",
                f"{nd.lon:.6f}",
                f"{nd.lat:.6f}",
                f"{nd.hm_m:.1f}",
                f"{nd.min_rx_dbm:.1f}",
            ]
            _num_cols = {1, 2, 3, 5, 6, 7, 8}  # 숫자 정렬 컬럼
            for c, v in enumerate(items):
                if c in _num_cols:
                    it = NumericItem(v)
                else:
                    it = QTableWidgetItem(v)
                it.setTextAlignment(Qt.AlignCenter)
                self.tbl.setItem(r, c, it)

            # 커버 색상
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

            # 이 GW가 담당 GW인 행은 강조
            if row['is_mine']:
                for c in range(len(COLS)):
                    it = self.tbl.item(r, c)
                    if it:
                        it.setBackground(QColor('#1a2535'))

        self.tbl.setSortingEnabled(True)

        # 요약
        n_total   = len(connected)
        n_covered = sum(1 for r in connected if r['cov'])
        n_mine    = sum(1 for r in connected if r['is_mine'])
        if connected:
            avg_dist = np.mean([r['dist'] for r in connected])
            max_dist = max(r['dist'] for r in connected)
            min_dist = min(r['dist'] for r in connected)
        else:
            avg_dist = max_dist = min_dist = 0

        self.lbl_summary.setText(
            f"담당 단말기: {n_mine}개  |  "
            f"수신 가능 단말기: {n_total}개  |  "
            f"커버: {n_covered}개  |  "
            f"거리 최소 {min_dist:.2f}km  "
            f"평균 {avg_dist:.2f}km  "
            f"최대 {max_dist:.2f}km")

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "CSV 내보내기",
            f"{self.gw.callsign}_nodes.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(COLS)
            for r in range(self.tbl.rowCount()):
                w.writerow([self.tbl.item(r, c).text()
                            for c in range(len(COLS))])