# ui/node_gw_detail_window.py
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

COLS = ['GW Callsign', '거리(km)', '방위각(°)', 'Pr(dBm)',
        '상태', '경도', '위도', '높이(m)', 'Pt(dBm)']


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


class NodeGWDetailWindow(QDialog):
    """Node에 연결 가능한 GW 상세 정보 창."""

    def __init__(self, nd: NodeEntry, gws: list[GWEntry],
                 node_idx: int, result=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{nd.callsign} — 연결 GW 상세")
        self.setStyleSheet(STYLE_DLG)
        self.resize(900, 520)
        self.setWindowFlag(Qt.Window)

        self.nd       = nd
        self.gws      = gws
        self.node_idx = node_idx
        self.result   = result
        self._build()
        self._fill()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        # ── Node 정보 ────────────────────────────────────────
        grp_nd = QGroupBox(f"Node 정보 — {self.nd.callsign}")
        grp_nd.setStyleSheet(
            f"QGroupBox{{color:#7ae87a;border:1px solid {BORDER};"
            f"border-radius:6px;margin-top:6px;padding-top:8px;"
            f"font-weight:bold;}}"
            f"QGroupBox::title{{subcontrol-origin:margin;left:8px;}}")
        nd_lay = QHBoxLayout(grp_nd)

        def _info(label, val):
            w = QVBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color:{MUTED};font-size:10px;")
            val_lbl = QLabel(str(val))
            val_lbl.setStyleSheet(
                f"color:{TEXT};font-size:13px;font-weight:bold;")
            w.addWidget(lbl)
            w.addWidget(val_lbl)
            return w

        nd_lay.addLayout(_info("위도",        f"{self.nd.lat:.6f}"))
        nd_lay.addLayout(_info("경도",        f"{self.nd.lon:.6f}"))
        nd_lay.addLayout(_info("수신 이득 Gr", f"{self.nd.gr_dbi} dBi"))
        nd_lay.addLayout(_info("수신 손실 Lr", f"{self.nd.lr_db} dB"))
        nd_lay.addLayout(_info("안테나 높이",  f"{self.nd.hm_m} m"))
        nd_lay.addLayout(_info("최소 수신",    f"{self.nd.min_rx_dbm} dBm"))
        nd_lay.addStretch()
        lay.addWidget(grp_nd)

        # ── 요약 ─────────────────────────────────────────────
        self.lbl_summary = QLabel("")
        self.lbl_summary.setStyleSheet(
            f"color:{MUTED};font-size:12px;padding:4px 0;")
        lay.addWidget(self.lbl_summary)

        # ── 테이블 ───────────────────────────────────────────
        self.tbl = QTableWidget(0, len(COLS))
        self.tbl.setHorizontalHeaderLabels(COLS)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
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
        nd     = self.nd
        result = self.result

        # 커버리지 결과에서 이 Node 정보 가져오기
        node_info = None
        if result and self.node_idx < len(result.nodes):
            node_info = result.nodes[self.node_idx]

        rows = []
        for gi, gw in enumerate(self.gws):
            dist = haversine(nd.lon, nd.lat, gw.lon, gw.lat)
            brg  = bearing(nd.lon, nd.lat, gw.lon, gw.lat)

            # Pr 값
            pr       = -999.0
            is_best  = False  # 이 GW가 best_gw인지
            if node_info:
                pr      = node_info.gw_prs.get(gw.callsign, -999.0)
                is_best = node_info.best_gw == gw.callsign

            rows.append({
                'gw'     : gw,
                'dist'   : dist,
                'brg'    : brg,
                'pr'     : pr,
                'is_best': is_best,
            })

        # 거리 순 정렬
        rows.sort(key=lambda x: x['dist'])

        self.tbl.setSortingEnabled(False)
        self.tbl.setRowCount(0)

        for row in rows:
            gw = row['gw']
            r  = self.tbl.rowCount()
            self.tbl.insertRow(r)

            pr_str = f"{row['pr']:.1f}" if row['pr'] > -999 else "─"

            # 상태
            if row['is_best']:
                status = "★ 주 연결 GW"
            elif row['pr'] > -999 and row['pr'] >= nd.min_rx_dbm:
                status = "○ 수신 가능"
            elif row['pr'] > -999:
                status = "△ 신호 약함"
            else:
                status = "─ 미분석"

            items = [
                gw.callsign,
                f"{row['dist']:.3f}",
                f"{row['brg']:.1f}",
                pr_str,
                status,
                f"{gw.lon:.6f}",
                f"{gw.lat:.6f}",
                f"{gw.hb_m:.1f}",
                f"{gw.pt_dbm:.1f}",
            ]
            for c, v in enumerate(items):
                it = QTableWidgetItem(v)
                it.setTextAlignment(Qt.AlignCenter)
                self.tbl.setItem(r, c, it)

            # 상태 색상
            status_item = self.tbl.item(r, 4)
            if row['is_best']:
                status_item.setForeground(QColor('#FFD700'))
                status_item.setBackground(QColor('#1a1500'))
                for c in range(len(COLS)):
                    it = self.tbl.item(r, c)
                    if it: it.setBackground(QColor('#1a1a00'))
            elif row['pr'] > -999 and row['pr'] >= nd.min_rx_dbm:
                status_item.setForeground(QColor('#00C94A'))
                status_item.setBackground(QColor('#0d2010'))
            elif row['pr'] > -999:
                status_item.setForeground(QColor('#FF8C00'))
                status_item.setBackground(QColor('#201000'))

            # Pr 색상
            if row['pr'] > -999:
                pr_col = ('#00C94A' if row['pr'] >= -110
                          else '#FFD700' if row['pr'] >= nd.min_rx_dbm
                          else '#FF4444')
                self.tbl.item(r, 3).setForeground(QColor(pr_col))

        self.tbl.setSortingEnabled(True)

        # 요약
        n_total    = len(rows)
        n_reachable = sum(1 for r in rows
                         if r['pr'] > -999 and r['pr'] >= nd.min_rx_dbm)
        best_gw    = next((r['gw'].callsign for r in rows if r['is_best']), "없음")
        best_pr    = next((r['pr'] for r in rows if r['is_best']), -999)
        best_dist  = next((r['dist'] for r in rows if r['is_best']), 0)

        self.lbl_summary.setText(
            f"주 연결 GW: {best_gw}  |  "
            f"수신 전력: {best_pr:.1f} dBm  |  "
            f"거리: {best_dist:.2f} km  |  "
            f"수신 가능 GW: {n_reachable}/{n_total}개")

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "CSV 내보내기",
            f"{self.nd.callsign}_gw_info.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(COLS)
            for r in range(self.tbl.rowCount()):
                w.writerow([self.tbl.item(r, c).text()
                            for c in range(len(COLS))])