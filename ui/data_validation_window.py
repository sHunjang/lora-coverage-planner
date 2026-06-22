# ui/data_validation_window.py — 데이터 검증 결과 창
from __future__ import annotations
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from ui.dialogs import DARK, PANEL, TEXT, MUTED, BORDER, STYLE_DLG
from core.data_validator import validate, summarize

COLS = ['심각도', '항목', '대상', '내용']

LEVEL_STYLE = {
    'ERROR': ('#FF6464', '✗ ERROR'),
    'WARN' : ('#FFD700', '△ WARN'),
}


class DataValidationWindow(QDialog):
    """GW/Node 데이터 정합성 검증 결과를 보여주는 창."""

    def __init__(self, gws, nodes, spatial=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("데이터 검증")
        self.setStyleSheet(STYLE_DLG)
        self.resize(820, 480)
        self.setWindowFlag(Qt.Window)

        self._gws     = gws
        self._nodes   = nodes
        self._spatial = spatial

        self._build()
        self._run()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        top = QHBoxLayout()
        self.lbl_summary = QLabel("검증 중...")
        self.lbl_summary.setStyleSheet("font-size:13px; font-weight:bold;")
        top.addWidget(self.lbl_summary)
        top.addStretch()

        btn_rerun = QPushButton("🔄 다시 검사")
        btn_rerun.clicked.connect(self._run)
        btn_rerun.setStyleSheet(
            f"QPushButton{{background:{PANEL};color:#7ab8e8;"
            f"border:1px solid #2a4a6a;border-radius:5px;"
            f"padding:6px 14px;font-size:11px;}}"
            f"QPushButton:hover{{background:#254d78;}}")
        top.addWidget(btn_rerun)
        lay.addLayout(top)

        self.tbl = QTableWidget(0, len(COLS))
        self.tbl.setHorizontalHeaderLabels(COLS)
        hh = self.tbl.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Fixed)
        hh.setSectionResizeMode(1, QHeaderView.Fixed)
        hh.setSectionResizeMode(2, QHeaderView.Fixed)
        hh.setSectionResizeMode(3, QHeaderView.Stretch)
        self.tbl.setColumnWidth(0, 90)
        self.tbl.setColumnWidth(1, 110)
        self.tbl.setColumnWidth(2, 120)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl.setStyleSheet(
            f"QTableWidget{{background:{PANEL};color:{TEXT};"
            f"gridline-color:{BORDER};}}"
            f"QHeaderView::section{{background:{DARK};color:{MUTED};"
            f"border:none;padding:4px;}}")
        lay.addWidget(self.tbl)

        hint = QLabel(
            "ERROR는 분석 결과 신뢰도에 직접 영향을 줄 수 있는 문제, "
            "WARN은 참고용 권고 사항입니다.")
        hint.setStyleSheet(f"color:{MUTED};font-size:10px;")
        lay.addWidget(hint)

    def _run(self):
        issues = validate(self._gws, self._nodes, self._spatial)
        summary = summarize(issues)

        if summary['n_total'] == 0:
            self.lbl_summary.setText("✓ 문제 없음 — GW/Node 데이터가 정상입니다.")
            self.lbl_summary.setStyleSheet(
                "color:#7ae87a; font-size:13px; font-weight:bold;")
        else:
            self.lbl_summary.setText(
                f"ERROR {summary['n_error']}건, WARN {summary['n_warn']}건 발견")
            color = "#FF6464" if summary['n_error'] else "#FFD700"
            self.lbl_summary.setStyleSheet(
                f"color:{color}; font-size:13px; font-weight:bold;")

        self.tbl.setRowCount(0)
        # ERROR 먼저, 그다음 WARN
        for issue in sorted(issues, key=lambda i: i.level != 'ERROR'):
            r = self.tbl.rowCount()
            self.tbl.insertRow(r)
            color, label = LEVEL_STYLE.get(issue.level, ('#a0a8be', issue.level))

            it_level = QTableWidgetItem(label)
            it_level.setForeground(QColor(color))
            self.tbl.setItem(r, 0, it_level)
            self.tbl.setItem(r, 1, QTableWidgetItem(issue.category))
            self.tbl.setItem(r, 2, QTableWidgetItem(issue.target))
            self.tbl.setItem(r, 3, QTableWidgetItem(issue.message))
