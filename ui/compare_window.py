# ui/compare_window.py
# 커버리지 분석 결과 비교 창
# - 히스토리: 분석할 때마다 자동 저장 (최대 10개)
# - 비교: 히스토리에서 2개 선택 → 수치 비교 테이블 + 차이값 표시

from __future__ import annotations
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QListWidget, QListWidgetItem, QSplitter,
    QAbstractItemView, QWidget, QScrollArea,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from ui.dialogs import DARK, PANEL, TEXT, MUTED, BORDER, STYLE_DLG

BTN = ("QPushButton{background:#1c2a3a;color:#7ab8e8;"
       "border:1px solid #2a4a6a;border-radius:4px;"
       "padding:5px 14px;font-size:11px;}"
       "QPushButton:hover{background:#254d78;}"
       "QPushButton:disabled{color:#3a5a6a;border-color:#1a2a3a;}")
BTN_RED = ("QPushButton{background:#3a1a1a;color:#e87a7a;"
           "border:1px solid #5a2a2a;border-radius:4px;"
           "padding:5px 14px;font-size:11px;}"
           "QPushButton:hover{background:#5a2020;}")
BTN_GREEN = ("QPushButton{background:#1d3a1d;color:#7ae87a;"
             "border:1px solid #2a5a2a;border-radius:4px;"
             "padding:5px 14px;font-size:11px;}"
             "QPushButton:hover{background:#256a25;}")

# 비교 테이블 색상
GREEN  = "#00C94A"
YELLOW = "#FFD700"
RED    = "#FF4444"
BLUE   = "#4f8ef7"
PURPLE = "#9B59B6"
ORANGE = "#FF8C00"


def _diff_color(diff: float, higher_is_better: bool = True) -> str:
    """차이값에 따른 색상 반환."""
    if abs(diff) < 0.1:
        return MUTED
    if higher_is_better:
        return GREEN if diff > 0 else RED
    else:
        return GREEN if diff < 0 else RED


def _fmt_diff(diff: float, unit: str = "", higher_is_better: bool = True) -> str:
    """차이값 포맷팅."""
    if abs(diff) < 0.01:
        return "─"
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff:.1f}{unit}"


class Snapshot:
    """분석 결과 스냅샷."""
    def __init__(self, label: str, result, gws: list, nodes: list):
        self.label     = label
        self.result    = result
        self.gws       = gws
        self.nodes     = nodes
        self.timestamp = datetime.now()
        self.ts_str    = self.timestamp.strftime("%H:%M:%S")

    def summary(self) -> dict:
        """비교에 사용할 주요 지표 딕셔너리."""
        r = self.result
        nodes = r.nodes

        n_total   = r.n_total
        n_covered = r.n_covered
        n_gws     = len(self.gws)

        # 중첩도
        n_multi  = sum(1 for nd in nodes if nd.n_rx_gw >= 2)
        n_single = sum(1 for nd in nodes if nd.covered and nd.n_rx_gw == 1)
        n_uncov  = sum(1 for nd in nodes if not nd.covered)
        overlap_pct = n_multi / n_covered * 100 if n_covered > 0 else 0
        single_pct  = n_single / n_total * 100  if n_total  > 0 else 0

        # SNR
        avg_snr    = getattr(r, 'avg_snr', 0.0)
        avg_margin = getattr(r, 'avg_snr_margin', 0.0)
        cell_rate  = getattr(r, 'cell_success_rate', 0.0)
        edge_rate  = getattr(r, 'edge_success_rate', 0.0)

        # 매크로 다이버시티
        macro_gain = getattr(r, 'macro_diversity_gain', 0.0)
        avg_rx_gw  = getattr(r, 'avg_n_rx_gw', 0.0)

        # ADR
        adr_dist  = getattr(r, 'adr_sf_distribution', {})
        avg_toa   = getattr(r, 'avg_toa_ms', 0.0)
        avg_nodes_per_gw = n_covered / n_gws if n_gws > 0 else 0

        return {
            'coverage_pct'     : r.coverage_pct,
            'n_covered'        : n_covered,
            'n_total'          : n_total,
            'n_gws'            : n_gws,
            'avg_nodes_per_gw' : avg_nodes_per_gw,
            'overlap_pct'      : overlap_pct,
            'single_pct'       : single_pct,
            'n_uncovered'      : n_uncov,
            'n_multi_gw'       : n_multi,
            'avg_snr'          : avg_snr,
            'avg_snr_margin'   : avg_margin,
            'cell_success_rate': cell_rate,
            'edge_success_rate': edge_rate,
            'macro_gain'       : macro_gain,
            'avg_rx_gw'        : avg_rx_gw,
            'avg_toa_ms'       : avg_toa,
            'sf7_pct'  : adr_dist.get(7,  0) / n_covered * 100 if n_covered else 0,
            'sf12_pct' : adr_dist.get(12, 0) / n_covered * 100 if n_covered else 0,
        }


# ── 비교 지표 정의 ────────────────────────────────────────
# (표시명, 키, 단위, 높을수록 좋음 여부)
COMPARE_METRICS = [
    # 커버리지
    ("── 커버리지",        None,                  "",    True),
    ("전체 커버리지",      'coverage_pct',         "%",   True),
    ("커버 Node 수",       'n_covered',            "개",  True),
    ("전체 Node 수",       'n_total',              "개",  True),
    ("활성 GW 수",         'n_gws',                "개",  False),
    ("GW당 평균 Node",     'avg_nodes_per_gw',     "개",  True),
    # 중첩도
    ("── 중첩도",          None,                  "",    True),
    ("중첩 커버",          'overlap_pct',          "%",   True),
    ("단독 커버",          'single_pct',           "%",   True),
    ("음영 지역",          'n_uncovered',          "개",  False),
    ("다중 GW 연결",       'n_multi_gw',           "개",  True),
    # 통신 성공율
    ("── 통신 성공율",     None,                  "",    True),
    ("셀 전체 성공율",     'cell_success_rate',    "%",   True),
    ("셀 경계 성공율",     'edge_success_rate',    "%",   True),
    ("평균 SNR",           'avg_snr',              "dB",  True),
    ("평균 SNR 마진",      'avg_snr_margin',       "dB",  True),
    # 매크로 다이버시티
    ("── 매크로 다이버시티", None,                 "",    True),
    ("평균 이득",          'macro_gain',           "dB",  True),
    ("평균 수신 GW",       'avg_rx_gw',            "개",  True),
    # ADR
    ("── ADR SF",          None,                  "",    True),
    ("SF7 비율 (고속)",    'sf7_pct',              "%",   True),
    ("SF12 비율 (장거리)", 'sf12_pct',             "%",   False),
    ("평균 ToA",           'avg_toa_ms',           "ms",  False),
]


class CompareWindow(QDialog):
    """분석 결과 히스토리 및 비교 창."""

    sig_load_snapshot = pyqtSignal(object)   # 스냅샷 불러오기 시그널

    def __init__(self, history: list[Snapshot], parent=None):
        super().__init__(parent)
        self.setWindowTitle("분석 결과 비교")
        self.setStyleSheet(STYLE_DLG)
        self.resize(1000, 700)
        self.setMinimumSize(800, 500)
        self.setWindowFlag(Qt.Window)

        self._history = history   # Snapshot 목록 (최신순)
        self._build()
        self._refresh_list()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        # ── 상단: 히스토리 목록 ──────────────────────────
        grp_hist = QGroupBox("분석 히스토리 (2개 선택 후 비교)")
        grp_hist.setStyleSheet(
            f"QGroupBox{{color:{MUTED};border:1px solid {BORDER};"
            f"border-radius:6px;margin-top:6px;padding-top:8px;}}"
            f"QGroupBox::title{{subcontrol-origin:margin;left:8px;}}")
        hist_lay = QVBoxLayout(grp_hist)
        hist_lay.setSpacing(6)

        self.lst = QListWidget()
        self.lst.setSelectionMode(QAbstractItemView.MultiSelection)
        self.lst.setFixedHeight(160)
        self.lst.setStyleSheet(f"""
            QListWidget {{
                background:{PANEL}; color:{TEXT};
                border:1px solid {BORDER}; border-radius:4px;
                font-size:11px;
            }}
            QListWidget::item {{
                padding:5px 8px; border-radius:3px;
            }}
            QListWidget::item:selected {{
                background:#253a5a; color:#7ab8e8;
            }}
            QListWidget::item:hover {{
                background:#1e2d40;
            }}
        """)
        hist_lay.addWidget(self.lst)

        # 히스토리 버튼
        hist_btn_row = QHBoxLayout()
        self.btn_compare = QPushButton("📊  선택 항목 비교")
        self.btn_compare.setStyleSheet(BTN_GREEN)
        self.btn_compare.clicked.connect(self._compare_selected)

        self.btn_load = QPushButton("⬆  이 결과로 복원")
        self.btn_load.setStyleSheet(BTN)
        self.btn_load.clicked.connect(self._load_selected)

        self.btn_rename = QPushButton("✏  이름 변경")
        self.btn_rename.setStyleSheet(BTN)
        self.btn_rename.clicked.connect(self._rename_selected)

        self.btn_del = QPushButton("✕  삭제")
        self.btn_del.setStyleSheet(BTN_RED)
        self.btn_del.clicked.connect(self._del_selected)

        self.btn_clear = QPushButton("전체 삭제")
        self.btn_clear.setStyleSheet(BTN_RED)
        self.btn_clear.clicked.connect(self._clear_history)

        hist_btn_row.addWidget(self.btn_compare)
        hist_btn_row.addWidget(self.btn_load)
        hist_btn_row.addWidget(self.btn_rename)
        hist_btn_row.addWidget(self.btn_del)
        hist_btn_row.addStretch()
        hist_btn_row.addWidget(self.btn_clear)
        hist_lay.addLayout(hist_btn_row)
        lay.addWidget(grp_hist)

        # ── 비교 결과 테이블 ──────────────────────────────
        self.grp_cmp = QGroupBox("비교 결과")
        self.grp_cmp.setStyleSheet(grp_hist.styleSheet())
        cmp_lay = QVBoxLayout(self.grp_cmp)

        self.tbl = QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(
            ["지표", "시나리오 A", "시나리오 B", "차이 (B-A)"])
        hh = self.tbl.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl.setSelectionMode(QAbstractItemView.NoSelection)
        self.tbl.setAlternatingRowColors(False)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setStyleSheet(f"""
            QTableWidget {{
                background:{PANEL}; color:{TEXT};
                gridline-color:{BORDER};
                border:none; font-size:11px;
            }}
            QHeaderView::section {{
                background:{DARK}; color:{MUTED};
                border:none; padding:6px;
                font-size:11px;
            }}
        """)
        cmp_lay.addWidget(self.tbl)

        # 비교 요약 라벨
        self.lbl_summary = QLabel("히스토리에서 2개를 선택하고 [비교] 버튼을 클릭하세요.")
        self.lbl_summary.setStyleSheet(
            f"color:{MUTED};font-size:11px;padding:4px;")
        self.lbl_summary.setWordWrap(True)
        cmp_lay.addWidget(self.lbl_summary)
        lay.addWidget(self.grp_cmp, 1)

        # ── 하단 버튼 ─────────────────────────────────────
        bot_row = QHBoxLayout()
        btn_close = QPushButton("닫기")
        btn_close.setStyleSheet(BTN)
        btn_close.clicked.connect(self.close)
        bot_row.addStretch()
        bot_row.addWidget(btn_close)
        lay.addLayout(bot_row)

    def _refresh_list(self):
        """히스토리 목록 갱신."""
        self.lst.clear()
        for i, snap in enumerate(self._history):
            n_gws = len(snap.gws)
            pct   = snap.result.coverage_pct
            item  = QListWidgetItem(
                f"[{snap.ts_str}]  {snap.label}  "
                f"| GW {n_gws}개 | 커버리지 {pct:.1f}%")
            item.setData(Qt.UserRole, i)
            # 최신 항목 강조
            if i == 0:
                item.setForeground(QColor('#7ab8e8'))
            self.lst.addItem(item)

    def add_snapshot(self, snap: Snapshot):
        """외부에서 스냅샷 추가."""
        self._history.insert(0, snap)
        if len(self._history) > 10:
            self._history.pop()
        self._refresh_list()

    def _selected_indices(self) -> list[int]:
        return [item.data(Qt.UserRole)
                for item in self.lst.selectedItems()]

    def _compare_selected(self):
        """선택된 2개 스냅샷을 비교."""
        idxs = self._selected_indices()
        if len(idxs) != 2:
            self.lbl_summary.setText("정확히 2개를 선택하세요.")
            return
        a = self._history[idxs[0]]
        b = self._history[idxs[1]]
        # 시간순으로 A=이전, B=최신
        if a.timestamp > b.timestamp:
            a, b = b, a
        self._draw_compare(a, b)

    def _draw_compare(self, a: Snapshot, b: Snapshot):
        """비교 테이블 렌더링."""
        sa = a.summary()
        sb = b.summary()

        # 헤더 업데이트
        self.tbl.setHorizontalHeaderLabels([
            "지표",
            f"A: {a.label} ({a.ts_str})",
            f"B: {b.label} ({b.ts_str})",
            "차이 (B − A)",
        ])

        self.tbl.setRowCount(0)

        for metric_name, key, unit, higher_better in COMPARE_METRICS:
            r = self.tbl.rowCount()
            self.tbl.insertRow(r)

            # 구분선 행 (key=None)
            if key is None:
                item = QTableWidgetItem(metric_name)
                item.setBackground(QColor('#1a1e2a'))
                item.setForeground(QColor(MUTED))
                font = QFont(); font.setBold(True)
                item.setFont(font)
                self.tbl.setItem(r, 0, item)
                for col in [1, 2, 3]:
                    empty = QTableWidgetItem("")
                    empty.setBackground(QColor('#1a1e2a'))
                    self.tbl.setItem(r, col, empty)
                self.tbl.setRowHeight(r, 22)
                continue

            val_a = sa.get(key, 0.0)
            val_b = sb.get(key, 0.0)
            diff  = val_b - val_a

            # 지표명
            name_item = QTableWidgetItem(f"  {metric_name}")
            name_item.setForeground(QColor(TEXT))
            self.tbl.setItem(r, 0, name_item)

            # 값 A
            def _fmt_val(v, u):
                if isinstance(v, int) or u == "개":
                    return f"{int(v)}{u}"
                return f"{v:.1f}{u}"

            item_a = QTableWidgetItem(_fmt_val(val_a, unit))
            item_a.setTextAlignment(Qt.AlignCenter)
            item_a.setForeground(QColor(TEXT))
            self.tbl.setItem(r, 1, item_a)

            # 값 B
            item_b = QTableWidgetItem(_fmt_val(val_b, unit))
            item_b.setTextAlignment(Qt.AlignCenter)
            item_b.setForeground(QColor(TEXT))
            self.tbl.setItem(r, 2, item_b)

            # 차이값
            diff_str  = _fmt_diff(diff, unit, higher_better)
            diff_color = _diff_color(diff, higher_better)
            item_d = QTableWidgetItem(diff_str)
            item_d.setTextAlignment(Qt.AlignCenter)
            item_d.setForeground(QColor(diff_color))
            font_d = QFont(); font_d.setBold(abs(diff) >= 1.0)
            item_d.setFont(font_d)
            self.tbl.setItem(r, 3, item_d)

            self.tbl.setRowHeight(r, 26)

        # 요약 문구
        cov_diff = sb.get('coverage_pct', 0) - sa.get('coverage_pct', 0)
        gw_diff  = sb.get('n_gws', 0)        - sa.get('n_gws', 0)
        snr_diff = sb.get('avg_snr', 0)      - sa.get('avg_snr', 0)

        parts = []
        if abs(cov_diff) >= 0.1:
            sign = "+" if cov_diff > 0 else ""
            col  = "🟢" if cov_diff > 0 else "🔴"
            parts.append(f"{col} 커버리지 {sign}{cov_diff:.1f}%")
        if gw_diff != 0:
            sign = "+" if gw_diff > 0 else ""
            col  = "🔴" if gw_diff > 0 else "🟢"
            parts.append(f"{col} GW 수 {sign}{gw_diff}개")
        if abs(snr_diff) >= 0.5:
            sign = "+" if snr_diff > 0 else ""
            col  = "🟢" if snr_diff > 0 else "🔴"
            parts.append(f"{col} 평균 SNR {sign}{snr_diff:.1f}dB")

        if parts:
            summary = f"B가 A 대비: " + " | ".join(parts)
        else:
            summary = "두 시나리오의 차이가 거의 없습니다."

        self.lbl_summary.setText(
            f"📊 {a.label} ({a.ts_str})  vs  {b.label} ({b.ts_str})\n{summary}")

    def _load_selected(self):
        """선택한 스냅샷의 GW/Node 결과를 메인 창에 복원."""
        idxs = self._selected_indices()
        if len(idxs) != 1:
            self.lbl_summary.setText("복원할 항목 1개를 선택하세요.")
            return
        snap = self._history[idxs[0]]
        self.sig_load_snapshot.emit(snap)
        self.lbl_summary.setText(
            f"'{snap.label}' 결과를 메인 창에 복원했습니다.")

    def _rename_selected(self):
        """선택한 스냅샷 이름 변경."""
        from PyQt5.QtWidgets import QInputDialog
        idxs = self._selected_indices()
        if len(idxs) != 1:
            self.lbl_summary.setText("이름을 변경할 항목 1개를 선택하세요.")
            return
        snap = self._history[idxs[0]]
        name, ok = QInputDialog.getText(
            self, "이름 변경", "새 이름:", text=snap.label)
        if ok and name.strip():
            snap.label = name.strip()
            self._refresh_list()

    def _del_selected(self):
        """선택한 스냅샷 삭제."""
        idxs = sorted(self._selected_indices(), reverse=True)
        if not idxs:
            return
        for i in idxs:
            self._history.pop(i)
        self._refresh_list()
        self.tbl.setRowCount(0)
        self.lbl_summary.setText("삭제 완료.")

    def _clear_history(self):
        """전체 히스토리 삭제."""
        from PyQt5.QtWidgets import QMessageBox
        ret = QMessageBox.question(
            self, "전체 삭제",
            f"히스토리 {len(self._history)}개를 모두 삭제하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No)
        if ret == QMessageBox.Yes:
            self._history.clear()
            self._refresh_list()
            self.tbl.setRowCount(0)
            self.lbl_summary.setText("전체 삭제 완료.")