# ui/result_panel.py — 커버리지 분석 결과 요약 패널
from __future__ import annotations
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QGroupBox, QGridLayout, QProgressBar, QFrame,
)
from PyQt5.QtCore import Qt

DARK   = "#181b22"
PANEL  = "#1e2130"
TEXT   = "#e0e4ef"
MUTED  = "#7a8099"
BORDER = "#2a2f3b"
GREEN  = "#00C94A"
YELLOW = "#FFD700"
RED    = "#FF4444"
BLUE   = "#4f8ef7"

SF_SENS = {
    7:-123.0, 8:-126.0, 9:-129.0,
    10:-132.0, 11:-134.5, 12:-137.0,
}
SF_COLORS_ADR = {
    7:'#FF4444', 8:'#FF8C00', 9:'#FFD700',
    10:'#00C94A', 11:'#4f8ef7', 12:'#9B59B6',
}


def _color_for_pct(pct: float) -> str:
    if pct >= 90: return GREEN
    if pct >= 70: return YELLOW
    return RED


class StatCard(QFrame):
    def __init__(self, title, value="─", unit="", color=TEXT, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame{{background:{PANEL};border:1px solid {BORDER};"
            f"border-radius:8px;padding:4px;}}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(2)
        self._lbl_title = QLabel(title)
        self._lbl_title.setStyleSheet(f"color:{MUTED};font-size:10px;border:none;")
        self._lbl_value = QLabel(value)
        self._lbl_value.setStyleSheet(
            f"color:{color};font-size:18px;font-weight:bold;border:none;")
        self._lbl_unit = QLabel(unit)
        self._lbl_unit.setStyleSheet(f"color:{MUTED};font-size:10px;border:none;")
        lay.addWidget(self._lbl_title)
        lay.addWidget(self._lbl_value)
        if unit:
            lay.addWidget(self._lbl_unit)

    def update(self, value, color=TEXT, unit=""):
        self._lbl_value.setText(value)
        self._lbl_value.setStyleSheet(
            f"color:{color};font-size:18px;font-weight:bold;border:none;")
        if unit:
            self._lbl_unit.setText(unit)


class ResultPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"QWidget{{background:{DARK};}}")
        self.setMinimumWidth(220)
        self._build()
        self.clear()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        title = QLabel("📊 분석 결과")
        title.setStyleSheet(f"color:{TEXT};font-size:13px;font-weight:bold;")
        lay.addWidget(title)

        # ── 기본 카드 ─────────────────────────────────────────
        grid = QGridLayout(); grid.setSpacing(6)
        self.card_cov   = StatCard("전체 커버리지", "─", "%")
        self.card_nodes = StatCard("커버 Node",    "─", "개")
        self.card_gws   = StatCard("활성 GW",      "─", "개")
        self.card_avg   = StatCard("GW당 평균",     "─", "개/GW")
        grid.addWidget(self.card_cov,   0, 0)
        grid.addWidget(self.card_nodes, 0, 1)
        grid.addWidget(self.card_gws,   1, 0)
        grid.addWidget(self.card_avg,   1, 1)
        lay.addLayout(grid)

        # ── 커버리지 바 ──────────────────────────────────────
        grp_bar = self._grp("커버리지")
        bar_lay = QVBoxLayout(grp_bar); bar_lay.setSpacing(4)
        self.prog_total = self._make_bar(GREEN)
        bar_lay.addWidget(self.prog_total)
        lay.addWidget(grp_bar)

        # ── SF별 커버리지 ────────────────────────────────────
        grp_sf = self._grp("SF별 커버리지 (Pr 기준)")
        sf_lay = QVBoxLayout(grp_sf); sf_lay.setSpacing(4)
        self._sf_bars = {}; self._sf_lbls = {}
        for sf, sens in SF_SENS.items():
            row = QHBoxLayout()
            lbl = QLabel(f"SF{sf}")
            lbl.setStyleSheet(
                f"color:{SF_COLORS_ADR[sf]};font-size:10px;min-width:28px;")
            bar = self._make_bar(SF_COLORS_ADR[sf])
            pct_lbl = QLabel("─")
            pct_lbl.setStyleSheet(
                f"color:{MUTED};font-size:10px;min-width:36px;"
                f"qproperty-alignment:AlignRight;")
            row.addWidget(lbl); row.addWidget(bar, 1); row.addWidget(pct_lbl)
            sf_lay.addLayout(row)
            self._sf_bars[sf] = bar; self._sf_lbls[sf] = pct_lbl
        lay.addWidget(grp_sf)

        # ── 매크로 다이버시티 ────────────────────────────────
        grp_md = self._grp("매크로 다이버시티")
        md_lay = QGridLayout(grp_md); md_lay.setSpacing(6)
        self.card_macro_gain = StatCard("평균 이득",    "─", "dB", BLUE)
        self.card_avg_rx_gw  = StatCard("평균 수신 GW", "─", "개", BLUE)
        md_lay.addWidget(self.card_macro_gain, 0, 0)
        md_lay.addWidget(self.card_avg_rx_gw,  0, 1)
        lay.addWidget(grp_md)

        # ── ADR SF 분포 ──────────────────────────────────────
        grp_adr = self._grp("ADR SF 분포")
        adr_lay = QVBoxLayout(grp_adr); adr_lay.setSpacing(3)
        self._adr_bars = {}; self._adr_lbls = {}
        for sf in range(7, 13):
            row = QHBoxLayout()
            lbl = QLabel(f"SF{sf}")
            lbl.setStyleSheet(
                f"color:{SF_COLORS_ADR[sf]};font-size:10px;min-width:28px;")
            bar = self._make_bar(SF_COLORS_ADR[sf])
            cnt_lbl = QLabel("─")
            cnt_lbl.setStyleSheet(
                f"color:{MUTED};font-size:10px;min-width:40px;"
                f"qproperty-alignment:AlignRight;")
            row.addWidget(lbl); row.addWidget(bar, 1); row.addWidget(cnt_lbl)
            adr_lay.addLayout(row)
            self._adr_bars[sf] = bar; self._adr_lbls[sf] = cnt_lbl
        self.card_avg_toa = StatCard("평균 ToA", "─", "ms", YELLOW)
        adr_lay.addWidget(self.card_avg_toa)
        lay.addWidget(grp_adr)

        # ── GW별 담당 Node ───────────────────────────────────
        grp_gw = self._grp("GW별 담당 Node")
        gw_lay = QVBoxLayout(grp_gw)
        self._gw_lbl = QLabel("─")
        self._gw_lbl.setStyleSheet(f"color:{MUTED};font-size:10px;")
        self._gw_lbl.setWordWrap(True)
        gw_lay.addWidget(self._gw_lbl)
        lay.addWidget(grp_gw)

        lay.addStretch()

    def _grp(self, title):
        g = QGroupBox(title)
        g.setStyleSheet(
            f"QGroupBox{{color:{MUTED};border:1px solid {BORDER};"
            f"border-radius:6px;margin-top:6px;padding-top:8px;}}"
            f"QGroupBox::title{{subcontrol-origin:margin;left:8px;}}")
        return g

    @staticmethod
    def _make_bar(color):
        bar = QProgressBar()
        bar.setRange(0, 100); bar.setValue(0)
        bar.setFixedHeight(12); bar.setTextVisible(False)
        bar.setStyleSheet(
            f"QProgressBar{{background:{DARK};border:1px solid {BORDER};"
            f"border-radius:5px;}}"
            f"QProgressBar::chunk{{background:{color};border-radius:5px;}}")
        return bar

    def clear(self):
        self.card_cov.update("─", MUTED)
        self.card_nodes.update("─", MUTED)
        self.card_gws.update("─", MUTED)
        self.card_avg.update("─", MUTED)
        self.prog_total.setValue(0)
        for sf in SF_SENS:
            self._sf_bars[sf].setValue(0)
            self._sf_lbls[sf].setText("─")
        self.card_macro_gain.update("─", MUTED)
        self.card_avg_rx_gw.update("─", MUTED)
        for sf in range(7, 13):
            self._adr_bars[sf].setValue(0)
            self._adr_lbls[sf].setText("─")
        self.card_avg_toa.update("─", MUTED)
        self._gw_lbl.setText("─")

    def update_result(self, result, gws=None):
        if result is None:
            self.clear(); return

        n_total   = result.n_total
        n_covered = result.n_covered
        pct       = result.coverage_pct
        col       = _color_for_pct(pct)

        self.card_cov.update(f"{pct:.1f}", col, "%")
        self.card_nodes.update(f"{n_covered}/{n_total}", col, "개")
        n_gws = len(gws) if gws else len(result.gw_counts)
        self.card_gws.update(str(n_gws), TEXT, "개")
        avg = n_covered / n_gws if n_gws else 0
        self.card_avg.update(f"{avg:.1f}", TEXT, "개/GW")
        self.prog_total.setValue(int(pct))

        # SF별 커버리지
        nodes = result.nodes
        for sf, sens in SF_SENS.items():
            n_sf   = sum(1 for nd in nodes if nd.best_pr >= sens)
            pct_sf = n_sf / n_total * 100 if n_total else 0
            self._sf_bars[sf].setValue(int(pct_sf))
            self._sf_lbls[sf].setText(f"{pct_sf:.0f}%")

        # 매크로 다이버시티
        macro_gain = getattr(result, 'macro_diversity_gain', 0.0)
        avg_rx_gw  = getattr(result, 'avg_n_rx_gw', 0.0)
        self.card_macro_gain.update(
            f"+{macro_gain:.1f}" if macro_gain > 0 else "─",
            BLUE if macro_gain > 0.5 else MUTED, "dB")
        self.card_avg_rx_gw.update(
            f"{avg_rx_gw:.1f}" if avg_rx_gw > 0 else "─",
            BLUE if avg_rx_gw > 1 else MUTED, "개")

        # ADR SF 분포
        adr_dist = getattr(result, 'adr_sf_distribution', {})
        avg_toa  = getattr(result, 'avg_toa_ms', 0.0)
        max_cnt  = max(adr_dist.values()) if adr_dist else 1
        for sf in range(7, 13):
            cnt = adr_dist.get(sf, 0)
            self._adr_bars[sf].setValue(int(cnt / max(max_cnt,1) * 100))
            self._adr_lbls[sf].setText(f"{cnt}개")
        self.card_avg_toa.update(
            f"{avg_toa:.0f}" if avg_toa > 0 else "─",
            YELLOW if avg_toa > 0 else MUTED, "ms")

        # GW별 담당 Node
        gw_counts = result.gw_counts
        if gw_counts:
            lines = []
            for cs, cnt in sorted(gw_counts.items(), key=lambda x: -x[1])[:8]:
                bar_len = int(cnt / max(gw_counts.values()) * 10)
                lines.append(f"{cs}: {'█'*bar_len} {cnt}개")
            if len(gw_counts) > 8:
                lines.append(f"… 외 {len(gw_counts)-8}개")
            self._gw_lbl.setText("\n".join(lines))
        else:
            self._gw_lbl.setText("─")