# ui/legend_window.py — 히트맵 dBm 범례 설정 창
from __future__ import annotations
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTableWidget,
    QHeaderView, QAbstractItemView, QDoubleSpinBox,
    QColorDialog,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from ui.dialogs import DARK, PANEL, TEXT, MUTED, BORDER, STYLE_DLG

STYLE = STYLE_DLG + f"""
QTableWidget {{
    background:{PANEL}; color:{TEXT};
    gridline-color:{BORDER};
    border:1px solid {BORDER}; border-radius:4px;
}}
QTableWidget::item {{
    padding:4px 8px;
}}
QHeaderView::section {{
    background:{DARK}; color:{MUTED};
    border:none; border-bottom:1px solid {BORDER};
    padding:4px 8px; font-size:11px;
}}
"""

DEFAULT_LEVELS = [
    {'pr': -90,  'color': '#FF2020'},
    {'pr': -100, 'color': '#FF8C00'},
    {'pr': -110, 'color': '#FFD700'},
    {'pr': -120, 'color': '#00C94A'},
    {'pr': -130, 'color': '#4f8ef7'},
]


class LegendWindow(QDialog):
    """히트맵 dBm 범례 설정 창."""
    sig_levels_changed = pyqtSignal(list)

    def __init__(self, levels=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("히트맵 dBm 범례 설정")
        self.setStyleSheet(STYLE)
        self.resize(420, 440)
        self.setWindowFlag(Qt.Window)
        self._levels  = [dict(lv) for lv in (levels or DEFAULT_LEVELS)]
        self._new_color = '#888888'
        self._build()
        self._load_table()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        # 설명
        info = QLabel(
            "각 등고선의 Pr 기준값(dBm)과 색상을 설정합니다.\n"
            "적용 버튼을 누르면 다음 히트맵 계산에 반영됩니다.")
        info.setStyleSheet(f"color:{MUTED};font-size:10px;")
        lay.addWidget(info)

        # 테이블
        self.tbl = QTableWidget(0, 3)
        self.tbl.setHorizontalHeaderLabels(['Pr 기준 (dBm)', '색상', '삭제'])
        self.tbl.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        self.tbl.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Fixed)
        self.tbl.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Fixed)
        self.tbl.setColumnWidth(1, 80)
        self.tbl.setColumnWidth(2, 50)
        self.tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl.setSelectionMode(QAbstractItemView.NoSelection)
        self.tbl.verticalHeader().setVisible(False)
        lay.addWidget(self.tbl)

        # 추가 행
        add_row = QHBoxLayout()

        add_lbl = QLabel("새 레벨:")
        add_lbl.setStyleSheet(f"color:{TEXT};font-size:11px;")

        self.sp_new_pr = QDoubleSpinBox()
        self.sp_new_pr.setRange(-200, 0)
        self.sp_new_pr.setValue(-115.0)
        self.sp_new_pr.setSuffix(" dBm")
        self.sp_new_pr.setDecimals(1)
        self.sp_new_pr.setStyleSheet(
            f"background:{PANEL};color:{TEXT};"
            f"border:1px solid {BORDER};border-radius:4px;"
            f"padding:4px;min-height:26px;")

        self.btn_new_color = QPushButton()
        self.btn_new_color.setFixedWidth(40)
        self.btn_new_color.setToolTip("색상 선택")
        self.btn_new_color.setStyleSheet(
            f"background:{self._new_color};"
            f"border:1px solid {BORDER};border-radius:4px;"
            f"min-height:26px;")
        self.btn_new_color.clicked.connect(self._pick_new_color)

        btn_add = QPushButton("+ 추가")
        btn_add.setStyleSheet(
            f"background:#1d4a1d;color:{TEXT};"
            f"border:1px solid #2a6a2a;border-radius:4px;"
            f"padding:4px 12px;min-height:26px;")
        btn_add.clicked.connect(self._add_level)

        add_row.addWidget(add_lbl)
        add_row.addWidget(self.sp_new_pr)
        add_row.addWidget(self.btn_new_color)
        add_row.addWidget(btn_add)
        add_row.addStretch()
        lay.addLayout(add_row)

        # 하단 버튼
        btn_row = QHBoxLayout()

        btn_reset = QPushButton("기본값 복원")
        btn_reset.setStyleSheet(
            f"QPushButton{{background:{PANEL};color:{MUTED};"
            f"border:1px solid {BORDER};border-radius:5px;"
            f"padding:6px 14px;font-size:11px;}}"
            f"QPushButton:hover{{color:{TEXT};}}")
        btn_reset.clicked.connect(self._reset)

        btn_close = QPushButton("닫기")
        btn_close.setProperty("role", "cancel")
        btn_close.clicked.connect(self.close)

        btn_apply = QPushButton("적용")
        btn_apply.setProperty("role", "ok")
        btn_apply.clicked.connect(self._apply)

        btn_row.addWidget(btn_reset)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        btn_row.addWidget(btn_apply)
        lay.addLayout(btn_row)

    def _load_table(self):
        self.tbl.setRowCount(0)
        for lv in sorted(self._levels, key=lambda x: -x['pr']):
            self._add_row(lv['pr'], lv['color'])

    def _add_row(self, pr: float, color: str):
        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        self.tbl.setRowHeight(r, 36)

        # Pr 스핀박스
        pr_spin = QDoubleSpinBox()
        pr_spin.setRange(-200, 0)
        pr_spin.setValue(pr)
        pr_spin.setSuffix(" dBm")
        pr_spin.setDecimals(1)
        pr_spin.setStyleSheet(
            f"background:{PANEL};color:{TEXT};"
            f"border:none;padding:2px 4px;")
        self.tbl.setCellWidget(r, 0, pr_spin)

        # 색상 버튼
        col_btn = QPushButton()
        col_btn.setStyleSheet(
            f"background:{color};"
            f"border:1px solid {BORDER};border-radius:3px;"
            f"margin:3px;")
        col_btn.setProperty("color", color)
        col_btn.clicked.connect(lambda _, b=col_btn: self._pick_color(b))
        self.tbl.setCellWidget(r, 1, col_btn)

        # 삭제 버튼
        del_btn = QPushButton("✕")
        del_btn.setStyleSheet(
            f"background:#3a1a1a;color:#ff6060;"
            f"border:none;border-radius:3px;"
            f"margin:3px;font-size:12px;font-weight:bold;")
        del_btn.clicked.connect(lambda _, row=r: self._del_row(row))
        self.tbl.setCellWidget(r, 2, del_btn)

    def _del_row(self, row: int):
        self.tbl.removeRow(row)
        # 삭제 후 나머지 행의 삭제 버튼 row index 재연결
        for r in range(self.tbl.rowCount()):
            del_btn = self.tbl.cellWidget(r, 2)
            if del_btn:
                try:
                    del_btn.clicked.disconnect()
                except Exception:
                    pass
                del_btn.clicked.connect(
                    lambda _, row=r: self._del_row(row))

    def _pick_color(self, btn: QPushButton):
        cur = btn.property("color") or "#ffffff"
        col = QColorDialog.getColor(QColor(cur), self, "색상 선택")
        if col.isValid():
            hex_c = col.name()
            btn.setProperty("color", hex_c)
            btn.setStyleSheet(
                f"background:{hex_c};"
                f"border:1px solid {BORDER};border-radius:3px;"
                f"margin:3px;")

    def _pick_new_color(self):
        col = QColorDialog.getColor(
            QColor(self._new_color), self, "색상 선택")
        if col.isValid():
            self._new_color = col.name()
            self.btn_new_color.setStyleSheet(
                f"background:{self._new_color};"
                f"border:1px solid {BORDER};border-radius:4px;"
                f"min-height:26px;")

    def _add_level(self):
        self._add_row(self.sp_new_pr.value(), self._new_color)

    def _collect(self) -> list:
        levels = []
        for r in range(self.tbl.rowCount()):
            spin = self.tbl.cellWidget(r, 0)
            btn  = self.tbl.cellWidget(r, 1)
            if spin and btn:
                levels.append({
                    'pr'   : spin.value(),
                    'color': btn.property("color") or '#888888',
                })
        levels.sort(key=lambda x: -x['pr'])
        return levels

    def _apply(self):
        self._levels = self._collect()
        self.sig_levels_changed.emit(self._levels)
        self.status_flash("적용 완료")

    def _reset(self):
        self._levels = [dict(lv) for lv in DEFAULT_LEVELS]
        self._load_table()

    def status_flash(self, msg: str):
        """창 타이틀에 잠깐 메시지 표시."""
        self.setWindowTitle(f"히트맵 dBm 범례 설정 — {msg}")
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(
            2000,
            lambda: self.setWindowTitle("히트맵 dBm 범례 설정"))

    def get_levels(self) -> list:
        return [dict(lv) for lv in self._levels]