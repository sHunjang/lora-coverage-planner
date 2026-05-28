# ui/result_panel.py — 커버리지 분석 결과 요약 패널
from __future__ import annotations
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QGroupBox, QGridLayout, QProgressBar, QFrame,
    QScrollArea,
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
PURPLE = "#9B59B6"
ORANGE = "#FF8C00"

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

def _color_for_rate(rate: float) -> str:
    if rate >= 95: return GREEN
    if rate >= 80: return YELLOW
    return RED


class StatCard(QFrame):
    def __init__(self, title, value="─", unit="", color=TEXT, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame{{background:{PANEL};border:1px solid {BORDER};"
            f"border-radius:8px;padding:4px;}}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(2)
        self._lbl_title = QLabel(title)
        self._lbl_title.setStyleSheet(
            f"color:{MUTED};font-size:11px;border:none;")  # 9 → 11
        self._lbl_title.setWordWrap(True)
        self._lbl_value = QLabel(value)
        self._lbl_value.setStyleSheet(
            f"color:{color};font-size:18px;font-weight:bold;border:none;")  # 15 → 18
        self._lbl_unit = QLabel(unit)
        self._lbl_unit.setStyleSheet(
            f"color:{MUTED};font-size:11px;border:none;")  # 9 → 11
        lay.addWidget(self._lbl_title)
        lay.addWidget(self._lbl_value)
        if unit:
            lay.addWidget(self._lbl_unit)

    def update(self, value, color=TEXT, unit=""):
        self._lbl_value.setText(value)
        self._lbl_value.setStyleSheet(
            f"color:{color};font-size:18px;font-weight:bold;border:none;")  # 15 → 18
        if unit:
            self._lbl_unit.setText(unit)


class ResultPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"QWidget{{background:{DARK};}}")
        self.setMinimumWidth(220)   # 200 → 220
        self._build()
        self.clear()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet(f"""
            QScrollArea {{ background:{DARK}; border:none; }}
            QScrollBar:vertical {{
                background:{PANEL}; width:6px; border-radius:3px;
            }}
            QScrollBar::handle:vertical {{
                background:#3a4060; border-radius:3px; min-height:20px;
            }}
            QScrollBar::handle:vertical:hover {{ background:#4f8ef7; }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{ height:0px; }}
        """)

        content = QWidget()
        content.setStyleSheet(f"QWidget{{background:{DARK};}}")
        lay = QVBoxLayout(content)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        title = QLabel("📊 분석 결과")
        title.setStyleSheet(
            f"color:{TEXT};font-size:14px;font-weight:bold;")  # 12 → 14
        lay.addWidget(title)

        # ── 기본 카드 ─────────────────────────────────────────
        grid = QGridLayout(); grid.setSpacing(4)
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
        bar_lay = QVBoxLayout(grp_bar); bar_lay.setSpacing(3)
        self.prog_total = self._make_bar(GREEN)
        bar_lay.addWidget(self.prog_total)
        lay.addWidget(grp_bar)

        # ── 통신 성공율 ───────────────────────────────────────
        grp_succ = self._grp("통신 성공율 (SNR 마진 기준)")
        succ_lay = QVBoxLayout(grp_succ); succ_lay.setSpacing(5)

        row_cell = QHBoxLayout()
        lbl_cell = QLabel("셀 전체")
        lbl_cell.setStyleSheet(
            f"color:{MUTED};font-size:11px;min-width:46px;")  # 9 → 11
        self.bar_cell_succ = self._make_bar(GREEN)
        self.lbl_cell_succ = QLabel("─")
        self.lbl_cell_succ.setStyleSheet(
            f"color:{GREEN};font-size:11px;min-width:40px;"  # 9 → 11
            f"qproperty-alignment:AlignRight;")
        row_cell.addWidget(lbl_cell)
        row_cell.addWidget(self.bar_cell_succ, 1)
        row_cell.addWidget(self.lbl_cell_succ)
        succ_lay.addLayout(row_cell)

        row_edge = QHBoxLayout()
        lbl_edge = QLabel("셀 경계")
        lbl_edge.setStyleSheet(
            f"color:{MUTED};font-size:11px;min-width:46px;")
        self.bar_edge_succ = self._make_bar(ORANGE)
        self.lbl_edge_succ = QLabel("─")
        self.lbl_edge_succ.setStyleSheet(
            f"color:{ORANGE};font-size:11px;min-width:40px;"
            f"qproperty-alignment:AlignRight;")
        row_edge.addWidget(lbl_edge)
        row_edge.addWidget(self.bar_edge_succ, 1)
        row_edge.addWidget(self.lbl_edge_succ)
        succ_lay.addLayout(row_edge)

        snr_grid = QGridLayout(); snr_grid.setSpacing(4)
        self.card_avg_snr    = StatCard("평균 SNR",      "─", "dB", BLUE)
        self.card_snr_margin = StatCard("평균 SNR 마진", "─", "dB", GREEN)
        snr_grid.addWidget(self.card_avg_snr,    0, 0)
        snr_grid.addWidget(self.card_snr_margin, 0, 1)
        succ_lay.addLayout(snr_grid)

        note = QLabel(
            "· 셀 전체: 커버 Node 중 SNR 마진 > 0dB 비율\n"
            "· 셀 경계: 마진 < 3dB Node 중 > 0dB 비율")
        note.setStyleSheet(f"color:{MUTED};font-size:10px;")  # 8 → 10
        note.setWordWrap(True)
        succ_lay.addWidget(note)
        lay.addWidget(grp_succ)

        # ── 중첩도 분석 ──────────────────────────────────────
        grp_ovlp = self._grp("중첩도 분석")
        ovlp_grid = QGridLayout(grp_ovlp); ovlp_grid.setSpacing(4)
        self.card_overlap    = StatCard("중첩 커버",    "─", "%",  PURPLE)
        self.card_single_cov = StatCard("단독 커버",    "─", "%",  BLUE)
        self.card_uncovered  = StatCard("음영 지역",    "─", "개", RED)
        self.card_multi_gw   = StatCard("다중 GW 연결", "─", "개", GREEN)
        ovlp_grid.addWidget(self.card_overlap,    0, 0)
        ovlp_grid.addWidget(self.card_single_cov, 0, 1)
        ovlp_grid.addWidget(self.card_uncovered,  1, 0)
        ovlp_grid.addWidget(self.card_multi_gw,   1, 1)
        lay.addWidget(grp_ovlp)

        # ── SF별 커버리지 ────────────────────────────────────
        grp_sf = self._grp("SF별 커버리지 (Pr 기준)")
        sf_lay = QVBoxLayout(grp_sf); sf_lay.setSpacing(4)
        self._sf_bars = {}; self._sf_lbls = {}
        for sf, sens in SF_SENS.items():
            row = QHBoxLayout()
            lbl = QLabel(f"SF{sf}")
            lbl.setStyleSheet(
                f"color:{SF_COLORS_ADR[sf]};font-size:11px;"  # 9 → 11
                f"min-width:28px;max-width:28px;")
            bar = self._make_bar(SF_COLORS_ADR[sf])
            pct_lbl = QLabel("─")
            pct_lbl.setStyleSheet(
                f"color:{MUTED};font-size:11px;min-width:36px;"  # 9 → 11
                f"qproperty-alignment:AlignRight;")
            row.addWidget(lbl)
            row.addWidget(bar, 1)
            row.addWidget(pct_lbl)
            sf_lay.addLayout(row)
            self._sf_bars[sf] = bar
            self._sf_lbls[sf] = pct_lbl
        lay.addWidget(grp_sf)

        # ── 매크로 다이버시티 ────────────────────────────────
        grp_md = self._grp("매크로 다이버시티")
        md_lay = QGridLayout(grp_md); md_lay.setSpacing(4)
        self.card_macro_gain = StatCard("평균 이득",    "─", "dB", BLUE)
        self.card_avg_rx_gw  = StatCard("평균 수신 GW", "─", "개", BLUE)
        md_lay.addWidget(self.card_macro_gain, 0, 0)
        md_lay.addWidget(self.card_avg_rx_gw,  0, 1)
        lay.addWidget(grp_md)

        # ── ADR SF 분포 ──────────────────────────────────────
        grp_adr = self._grp("ADR SF 분포")
        adr_lay = QVBoxLayout(grp_adr); adr_lay.setSpacing(4)
        self._adr_bars = {}; self._adr_lbls = {}
        for sf in range(7, 13):
            row = QHBoxLayout()
            lbl = QLabel(f"SF{sf}")
            lbl.setStyleSheet(
                f"color:{SF_COLORS_ADR[sf]};font-size:11px;"
                f"min-width:28px;max-width:28px;")
            bar = self._make_bar(SF_COLORS_ADR[sf])
            cnt_lbl = QLabel("─")
            cnt_lbl.setStyleSheet(
                f"color:{MUTED};font-size:11px;min-width:38px;"
                f"qproperty-alignment:AlignRight;")
            row.addWidget(lbl)
            row.addWidget(bar, 1)
            row.addWidget(cnt_lbl)
            adr_lay.addLayout(row)
            self._adr_bars[sf] = bar
            self._adr_lbls[sf] = cnt_lbl
        self.card_avg_toa = StatCard("평균 ToA", "─", "ms", YELLOW)
        adr_lay.addWidget(self.card_avg_toa)
        lay.addWidget(grp_adr)

        # ── 트래픽 용량 분석 ─────────────────────────────────
        grp_trf = self._grp("트래픽 용량 분석 (ALOHA)")
        trf_lay = QVBoxLayout(grp_trf); trf_lay.setSpacing(5)

        row_load = QHBoxLayout()
        lbl_load = QLabel("평균 부하")
        lbl_load.setStyleSheet(
            f"color:{MUTED};font-size:11px;min-width:46px;")
        self.bar_load = self._make_bar(BLUE)
        self.lbl_load = QLabel("─")
        self.lbl_load.setStyleSheet(
            f"color:{BLUE};font-size:11px;min-width:40px;"
            f"qproperty-alignment:AlignRight;")
        row_load.addWidget(lbl_load)
        row_load.addWidget(self.bar_load, 1)
        row_load.addWidget(self.lbl_load)
        trf_lay.addLayout(row_load)

        row_pdr = QHBoxLayout()
        lbl_pdr = QLabel("평균 PDR")
        lbl_pdr.setStyleSheet(
            f"color:{MUTED};font-size:11px;min-width:46px;")
        self.bar_pdr = self._make_bar(GREEN)
        self.lbl_pdr = QLabel("─")
        self.lbl_pdr.setStyleSheet(
            f"color:{GREEN};font-size:11px;min-width:40px;"
            f"qproperty-alignment:AlignRight;")
        row_pdr.addWidget(lbl_pdr)
        row_pdr.addWidget(self.bar_pdr, 1)
        row_pdr.addWidget(self.lbl_pdr)
        trf_lay.addLayout(row_pdr)

        trf_grid = QGridLayout(); trf_grid.setSpacing(4)
        self.card_overload_gw = StatCard("과부하 GW", "─", "개", RED)
        self.card_avg_load    = StatCard("평균 부하", "─", "%",  BLUE)
        trf_grid.addWidget(self.card_overload_gw, 0, 0)
        trf_grid.addWidget(self.card_avg_load,    0, 1)
        trf_lay.addLayout(trf_grid)

        self._trf_lbl = QLabel("─")
        self._trf_lbl.setStyleSheet(f"color:{MUTED};font-size:11px;")
        self._trf_lbl.setWordWrap(True)
        trf_lay.addWidget(self._trf_lbl)

        note_trf = QLabel(
            "· 부하: GW 채널 용량 대비 ToA 사용률\n"
            "· PDR: Pure ALOHA 기반 패킷 성공률\n"
            "· 과부하 기준: 부하 ≥ 80%")
        note_trf.setStyleSheet(f"color:{MUTED};font-size:10px;")
        note_trf.setWordWrap(True)
        trf_lay.addWidget(note_trf)
        lay.addWidget(grp_trf)

        # ── GW별 담당 Node ───────────────────────────────────
        grp_gw = self._grp("GW별 담당 Node")
        gw_lay = QVBoxLayout(grp_gw)
        self._gw_lbl = QLabel("─")
        self._gw_lbl.setStyleSheet(f"color:{MUTED};font-size:11px;")
        self._gw_lbl.setWordWrap(True)
        gw_lay.addWidget(self._gw_lbl)
        lay.addWidget(grp_gw)

        lay.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)

    def _grp(self, title):
        g = QGroupBox(title)
        g.setStyleSheet(
            f"QGroupBox{{color:{MUTED};border:1px solid {BORDER};"
            f"border-radius:6px;margin-top:6px;padding-top:8px;"
            f"font-size:11px;}}"   # 10 → 11
            f"QGroupBox::title{{subcontrol-origin:margin;left:8px;}}")
        return g

    @staticmethod
    def _make_bar(color):
        bar = QProgressBar()
        bar.setRange(0, 100); bar.setValue(0)
        bar.setFixedHeight(12); bar.setTextVisible(False)  # 10 → 12
        bar.setStyleSheet(
            f"QProgressBar{{background:{DARK};border:1px solid {BORDER};"
            f"border-radius:4px;}}"
            f"QProgressBar::chunk{{background:{color};border-radius:4px;}}")
        return bar

    def clear(self):
        self.card_cov.update("─", MUTED)
        self.card_nodes.update("─", MUTED)
        self.card_gws.update("─", MUTED)
        self.card_avg.update("─", MUTED)
        self.prog_total.setValue(0)
        self.bar_cell_succ.setValue(0)
        self.lbl_cell_succ.setText("─")
        self.bar_edge_succ.setValue(0)
        self.lbl_edge_succ.setText("─")
        self.card_avg_snr.update("─", MUTED)
        self.card_snr_margin.update("─", MUTED)
        self.card_overlap.update("─", MUTED)
        self.card_single_cov.update("─", MUTED)
        self.card_uncovered.update("─", MUTED)
        self.card_multi_gw.update("─", MUTED)
        for sf in SF_SENS:
            self._sf_bars[sf].setValue(0)
            self._sf_lbls[sf].setText("─")
        self.card_macro_gain.update("─", MUTED)
        self.card_avg_rx_gw.update("─", MUTED)
        for sf in range(7, 13):
            self._adr_bars[sf].setValue(0)
            self._adr_lbls[sf].setText("─")
        self.card_avg_toa.update("─", MUTED)
        self.bar_load.setValue(0)
        self.lbl_load.setText("─")
        self.bar_pdr.setValue(0)
        self.lbl_pdr.setText("─")
        self.card_overload_gw.update("─", MUTED)
        self.card_avg_load.update("─", MUTED)
        self._trf_lbl.setText("─")
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

        # ── 통신 성공율 ──────────────────────────────────────
        cell_rate  = getattr(result, 'cell_success_rate', 0.0)
        edge_rate  = getattr(result, 'edge_success_rate', 0.0)
        avg_snr    = getattr(result, 'avg_snr',           0.0)
        avg_margin = getattr(result, 'avg_snr_margin',    0.0)

        cell_col = _color_for_rate(cell_rate)
        edge_col = _color_for_rate(edge_rate)

        self.bar_cell_succ.setValue(int(cell_rate))
        self.lbl_cell_succ.setText(f"{cell_rate:.1f}%")
        self.lbl_cell_succ.setStyleSheet(
            f"color:{cell_col};font-size:11px;min-width:40px;"
            f"qproperty-alignment:AlignRight;")

        self.bar_edge_succ.setValue(int(edge_rate))
        self.lbl_edge_succ.setText(f"{edge_rate:.1f}%")
        self.lbl_edge_succ.setStyleSheet(
            f"color:{edge_col};font-size:11px;min-width:40px;"
            f"qproperty-alignment:AlignRight;")

        margin_col = GREEN if avg_margin >= 0 else RED
        self.card_avg_snr.update(
            f"{avg_snr:.1f}" if avg_snr > -999 else "─",
            BLUE if avg_snr > -999 else MUTED, "dB")
        self.card_snr_margin.update(
            f"{avg_margin:+.1f}" if avg_margin > -999 else "─",
            margin_col if avg_margin > -999 else MUTED, "dB")

        # ── 중첩도 계산 ──────────────────────────────────────
        nodes        = result.nodes
        n_uncovered  = sum(1 for nd in nodes if not nd.covered)
        n_multi_gw   = sum(1 for nd in nodes if nd.n_rx_gw >= 2)
        n_single_cov = sum(1 for nd in nodes
                           if nd.covered and nd.n_rx_gw == 1)
        overlap_pct  = (n_multi_gw / n_covered * 100) if n_covered > 0 else 0.0
        single_pct   = (n_single_cov / n_total * 100)  if n_total > 0  else 0.0

        self.card_overlap.update(
            f"{overlap_pct:.1f}", PURPLE if overlap_pct > 0 else MUTED, "%")
        self.card_single_cov.update(
            f"{single_pct:.1f}", BLUE if single_pct > 0 else MUTED, "%")
        self.card_uncovered.update(
            str(n_uncovered), RED if n_uncovered > 0 else GREEN, "개")
        self.card_multi_gw.update(
            str(n_multi_gw), GREEN if n_multi_gw > 0 else MUTED, "개")

        # ── SF별 커버리지 ────────────────────────────────────
        for sf, sens in SF_SENS.items():
            n_sf   = sum(1 for nd in nodes if nd.best_pr >= sens)
            pct_sf = n_sf / n_total * 100 if n_total else 0
            self._sf_bars[sf].setValue(int(pct_sf))
            self._sf_lbls[sf].setText(f"{pct_sf:.0f}%")

        # ── 매크로 다이버시티 ────────────────────────────────
        macro_gain = getattr(result, 'macro_diversity_gain', 0.0)
        avg_rx_gw  = getattr(result, 'avg_n_rx_gw',         0.0)
        self.card_macro_gain.update(
            f"+{macro_gain:.1f}" if macro_gain > 0 else "─",
            BLUE if macro_gain > 0.5 else MUTED, "dB")
        self.card_avg_rx_gw.update(
            f"{avg_rx_gw:.1f}" if avg_rx_gw > 0 else "─",
            BLUE if avg_rx_gw > 1 else MUTED, "개")

        # ── ADR SF 분포 ──────────────────────────────────────
        adr_dist = getattr(result, 'adr_sf_distribution', {})
        avg_toa  = getattr(result, 'avg_toa_ms',           0.0)
        max_cnt  = max(adr_dist.values()) if adr_dist else 1
        for sf in range(7, 13):
            cnt = adr_dist.get(sf, 0)
            self._adr_bars[sf].setValue(int(cnt / max(max_cnt, 1) * 100))
            self._adr_lbls[sf].setText(f"{cnt}개")
        self.card_avg_toa.update(
            f"{avg_toa:.0f}" if avg_toa > 0 else "─",
            YELLOW if avg_toa > 0 else MUTED, "ms")

        # ── 트래픽 용량 분석 ─────────────────────────────────
        gw_traffic = getattr(result, 'gw_traffic',      {})
        avg_pdr    = getattr(result, 'avg_pdr',          100.0)
        n_overload = getattr(result, 'n_overloaded_gw',  0)
        avg_load   = getattr(result, 'avg_load_pct',     0.0)

        load_bar_val = min(int(avg_load), 100)
        load_col     = (RED    if avg_load >= 80
                        else YELLOW if avg_load >= 50 else BLUE)
        self.bar_load.setValue(load_bar_val)
        self.lbl_load.setText(f"{avg_load:.1f}%")
        self.lbl_load.setStyleSheet(
            f"color:{load_col};font-size:11px;min-width:40px;"
            f"qproperty-alignment:AlignRight;")

        pdr_col = (GREEN if avg_pdr >= 90
                   else YELLOW if avg_pdr >= 70 else RED)
        self.bar_pdr.setValue(int(avg_pdr))
        self.lbl_pdr.setText(f"{avg_pdr:.1f}%")
        self.lbl_pdr.setStyleSheet(
            f"color:{pdr_col};font-size:11px;min-width:40px;"
            f"qproperty-alignment:AlignRight;")

        self.card_overload_gw.update(
            str(n_overload),
            RED if n_overload > 0 else GREEN, "개")
        self.card_avg_load.update(
            f"{avg_load:.1f}", load_col, "%")

        if gw_traffic:
            lines = []
            sorted_gws = sorted(
                gw_traffic.items(),
                key=lambda x: -x[1]['load_pct'])[:8]
            for cs, info in sorted_gws:
                load = info['load_pct']
                pdr  = info['pdr']
                n_nd = info['n_nodes']
                flag = "⚠" if info['overloaded'] else " "
                col  = "🔴" if load >= 80 else "🟡" if load >= 50 else "🟢"
                lines.append(
                    f"{flag}{col} {cs}: {load:.0f}% | PDR {pdr:.0f}% | {n_nd}개")
            if len(gw_traffic) > 8:
                lines.append(f"… 외 {len(gw_traffic)-8}개")
            self._trf_lbl.setText("\n".join(lines))
        else:
            self._trf_lbl.setText("─")

        # ── GW별 담당 Node ───────────────────────────────────
        gw_counts = result.gw_counts
        if gw_counts:
            lines   = []
            max_cnt = max(gw_counts.values()) if max(gw_counts.values()) > 0 else 1
            for cs, cnt in sorted(gw_counts.items(), key=lambda x: -x[1])[:8]:
                bar_len = int(cnt / max_cnt * 10)
                lines.append(f"{cs}: {'█'*bar_len} {cnt}개")
            if len(gw_counts) > 8:
                lines.append(f"… 외 {len(gw_counts)-8}개")
            self._gw_lbl.setText("\n".join(lines))
        else:
            self._gw_lbl.setText("─")