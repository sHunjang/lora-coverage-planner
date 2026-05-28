# ui/graph_window.py
# RSSI 분포 / 중첩도 분포 / SF 분포 그래프 창
# matplotlib 기반, 커버리지 분석 결과를 시각화

from __future__ import annotations
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QTabWidget, QWidget, QLabel, QSizePolicy,
)
from PyQt5.QtCore import Qt
from ui.dialogs import DARK, PANEL, TEXT, MUTED, BORDER, STYLE_DLG

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ── matplotlib 한글 폰트 설정 (Windows Malgun Gothic) ────────
import matplotlib.font_manager as fm
import platform

def _set_korean_font():
    """운영체제별 한글 폰트 설정."""
    if platform.system() == 'Windows':
        # Windows: Malgun Gothic (맑은 고딕)
        font_path = 'C:/Windows/Fonts/malgun.ttf'
        if not __import__('os').path.exists(font_path):
            font_path = 'C:/Windows/Fonts/gulim.ttc'  # 굴림 폴백
    elif platform.system() == 'Darwin':
        # macOS: Apple Gothic
        font_path = '/System/Library/Fonts/AppleGothic.ttf'
    else:
        # Linux: NanumGothic 등
        font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'

    try:
        font_prop = fm.FontProperties(fname=font_path)
        font_name = font_prop.get_name()
        matplotlib.rcParams['font.family'] = font_name
        matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
        fm.fontManager.addfont(font_path)
    except Exception as e:
        print(f"[GRAPH] 한글 폰트 설정 실패: {e}")

_set_korean_font()

# ── 그래프 스타일 ────────────────────────────────────────────
STYLE_MPL = {
    'figure.facecolor'  : '#181b22',
    'axes.facecolor'    : '#1e2130',
    'axes.edgecolor'    : '#2a2f3b',
    'axes.labelcolor'   : '#e0e4ef',
    'axes.titlecolor'   : '#e0e4ef',
    'xtick.color'       : '#7a8099',
    'ytick.color'       : '#7a8099',
    'grid.color'        : '#2a2f3b',
    'grid.linestyle'    : '--',
    'grid.alpha'        : 0.5,
    'text.color'        : '#e0e4ef',
    'legend.facecolor'  : '#1e2130',
    'legend.edgecolor'  : '#2a2f3b',
    'legend.labelcolor' : '#e0e4ef',
    'axes.unicode_minus': False,
}

BTN = ("QPushButton{background:#1c2a3a;color:#7ab8e8;"
       "border:1px solid #2a4a6a;border-radius:4px;"
       "padding:5px 14px;font-size:11px;}"
       "QPushButton:hover{background:#254d78;}")


class MplCanvas(FigureCanvas):
    """matplotlib Figure를 Qt 위젯으로 래핑."""
    def __init__(self, parent=None, figsize=(8, 5)):
        with plt.rc_context(STYLE_MPL):
            self.fig = Figure(figsize=figsize, tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


class GraphWindow(QDialog):
    """RSSI 분포 / 중첩도 분포 / SF 분포 그래프 창."""

    def __init__(self, result, gws=None, nodes=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("분석 결과 그래프")
        self.setStyleSheet(STYLE_DLG)
        self.resize(860, 560)
        self.setWindowFlag(Qt.Window)

        self._result = result
        self._gws    = gws   or []
        self._nodes  = nodes or []

        self._build()
        self._draw_all()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        # ── 탭 ───────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border:1px solid {BORDER}; background:{DARK};
            }}
            QTabBar::tab {{
                background:{PANEL}; color:{MUTED};
                border:1px solid {BORDER}; border-bottom:none;
                border-radius:4px 4px 0 0;
                padding:6px 18px; margin-right:2px; font-size:11px;
            }}
            QTabBar::tab:selected {{ background:{DARK}; color:{TEXT}; }}
            QTabBar::tab:hover    {{ color:{TEXT}; }}
        """)
        lay.addWidget(self.tabs)

        # 탭 1: RSSI 분포
        self.tab_rssi = QWidget()
        rssi_lay = QVBoxLayout(self.tab_rssi)
        rssi_lay.setContentsMargins(0, 0, 0, 0)
        self.canvas_rssi = MplCanvas(figsize=(8, 4.5))
        rssi_lay.addWidget(self.canvas_rssi)
        self.tabs.addTab(self.tab_rssi, "📶 RSSI 분포")

        # 탭 2: 중첩도 분포
        self.tab_ovlp = QWidget()
        ovlp_lay = QVBoxLayout(self.tab_ovlp)
        ovlp_lay.setContentsMargins(0, 0, 0, 0)
        self.canvas_ovlp = MplCanvas(figsize=(8, 4.5))
        ovlp_lay.addWidget(self.canvas_ovlp)
        self.tabs.addTab(self.tab_ovlp, "📊 중첩도 분포")

        # 탭 3: SF 분포
        self.tab_sf = QWidget()
        sf_lay = QVBoxLayout(self.tab_sf)
        sf_lay.setContentsMargins(0, 0, 0, 0)
        self.canvas_sf = MplCanvas(figsize=(8, 4.5))
        sf_lay.addWidget(self.canvas_sf)
        self.tabs.addTab(self.tab_sf, "📡 SF 분포")

        # 탭 4: SNR 분포
        self.tab_snr = QWidget()
        snr_lay = QVBoxLayout(self.tab_snr)
        snr_lay.setContentsMargins(0, 0, 0, 0)
        self.canvas_snr = MplCanvas(figsize=(8, 4.5))
        snr_lay.addWidget(self.canvas_snr)
        self.tabs.addTab(self.tab_snr, "📈 SNR 분포")

        # ── 하단 버튼 ────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_save = QPushButton("💾 이미지 저장")
        btn_save.setStyleSheet(BTN)
        btn_save.clicked.connect(self._save_current)
        btn_close = QPushButton("닫기")
        btn_close.setStyleSheet(BTN)
        btn_close.clicked.connect(self.close)
        btn_row.addStretch()
        btn_row.addWidget(btn_save)
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)

    def _draw_all(self):
        """모든 탭 그래프를 한 번에 그립니다."""
        self._draw_rssi()
        self._draw_overlap()
        self._draw_sf()
        self._draw_snr()

    # ── 탭 1: RSSI 분포 히스토그램 ──────────────────────────

    def _draw_rssi(self):
        """
        RSSI(수신전력) 분포 히스토그램.
        커버된 Node(초록)와 미커버 Node(빨강)를 겹쳐서 표시.
        세로 점선으로 평균값 표시.
        """
        result = self._result
        if not result or not result.nodes:
            return

        covered_pr   = [nd.best_pr for nd in result.nodes
                        if nd.covered and nd.best_pr > -999]
        uncovered_pr = [nd.best_pr for nd in result.nodes
                        if not nd.covered and nd.best_pr > -999]

        with plt.rc_context(STYLE_MPL):
            fig = self.canvas_rssi.fig
            fig.clear()
            ax  = fig.add_subplot(111)

            bins = np.arange(-140, -50, 2)   # 2dBm 간격 빈

            # 커버 Node RSSI
            if covered_pr:
                ax.hist(covered_pr, bins=bins,
                        color='#00C94A', alpha=0.7,
                        label=f'커버 ({len(covered_pr)}개)',
                        edgecolor='none')
                ax.axvline(np.mean(covered_pr),
                           color='#00C94A', linestyle='--',
                           linewidth=1.5,
                           label=f'평균 {np.mean(covered_pr):.1f}dBm')

            # 미커버 Node RSSI
            if uncovered_pr:
                ax.hist(uncovered_pr, bins=bins,
                        color='#FF4444', alpha=0.7,
                        label=f'미커버 ({len(uncovered_pr)}개)',
                        edgecolor='none')
                ax.axvline(np.mean(uncovered_pr),
                           color='#FF4444', linestyle='--',
                           linewidth=1.5,
                           label=f'평균 {np.mean(uncovered_pr):.1f}dBm')

            # 최소 수신 감도 기준선
            if self._nodes:
                min_rx_vals = [nd.min_rx_dbm for nd in self._nodes]
                min_rx_avg  = np.mean(min_rx_vals)
                ax.axvline(min_rx_avg,
                           color='#FFD700', linestyle=':',
                           linewidth=2.0,
                           label=f'감도 기준 {min_rx_avg:.1f}dBm')

            ax.set_xlabel('수신전력 RSSI (dBm)', fontsize=10)
            ax.set_ylabel('Node 수 (개)',        fontsize=10)
            ax.set_title('RSSI 분포',            fontsize=12,
                         fontweight='bold', pad=10)
            ax.legend(fontsize=9)
            ax.grid(True, axis='y')
            ax.set_xlim(-145, -50)

            # SF별 감도 기준선 배경색
            SF_SENS = {
                7:-123.0, 8:-126.0, 9:-129.0,
                10:-132.0, 11:-134.5, 12:-137.0,
            }
            SF_COLORS = {
                7:'#FF4444', 8:'#FF8C00', 9:'#FFD700',
                10:'#00C94A', 11:'#4f8ef7', 12:'#9B59B6',
            }
            for sf, sens in SF_SENS.items():
                ax.axvline(sens, color=SF_COLORS[sf],
                           alpha=0.3, linewidth=0.8,
                           linestyle='-.')

            self.canvas_rssi.draw()

    # ── 탭 2: 중첩도 분포 ───────────────────────────────────

    def _draw_overlap(self):
        """
        중첩도 분포 그래프.
        상단: GW별 담당 Node 수 가로 막대 그래프
        하단: 수신 GW 수별 Node 분포 (1개/2개/3개+)
        """
        result = self._result
        if not result or not result.nodes:
            return

        with plt.rc_context(STYLE_MPL):
            fig = self.canvas_ovlp.fig
            fig.clear()

            ax1 = fig.add_subplot(211)   # GW별 담당 Node
            ax2 = fig.add_subplot(212)   # 수신 GW 수별 분포

            # ── 상단: GW별 담당 Node 수 ──────────────────────
            gw_counts = result.gw_counts
            if gw_counts:
                sorted_gws = sorted(gw_counts.items(),
                                    key=lambda x: -x[1])[:15]  # 최대 15개
                labels = [cs for cs, _ in sorted_gws]
                values = [cnt for _, cnt in sorted_gws]

                # GW별 색상 팔레트
                GW_HEX = [
                    '#e74c3c','#3498db','#2ecc71','#9b59b6','#e67e22',
                    '#c0392b','#2980b9','#27ae60','#8e44ad','#17a589',
                    '#e91e8c','#5dade2','#58d68d','#f0e68c','#2c3e50',
                ]
                colors = [GW_HEX[i % len(GW_HEX)]
                          for i in range(len(labels))]

                bars = ax1.barh(labels[::-1], values[::-1],
                                color=colors[::-1],
                                height=0.6, edgecolor='none')

                # 막대 끝에 수치 표시
                for bar, val in zip(bars, values[::-1]):
                    ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                             f'{val}개', va='center', ha='left',
                             fontsize=8, color=TEXT)

                ax1.set_xlabel('담당 Node 수 (개)', fontsize=9)
                ax1.set_title('GW별 담당 Node 수', fontsize=11,
                              fontweight='bold', pad=8)
                ax1.grid(True, axis='x')
                ax1.set_xlim(0, max(values) * 1.2 if values else 10)

            # ── 하단: 수신 GW 수별 Node 분포 ─────────────────
            n_rx_counts = {}
            for nd in result.nodes:
                n_rx = nd.n_rx_gw
                key  = min(n_rx, 4)   # 4개 이상은 "4+" 으로 묶음
                n_rx_counts[key] = n_rx_counts.get(key, 0) + 1

            rx_labels  = []
            rx_values  = []
            rx_colors  = ['#FF4444','#4f8ef7','#00C94A','#FFD700','#9B59B6']
            rx_lblnames = ['0개 (미커버)','1개','2개','3개','4개+']

            for i in range(5):
                cnt = n_rx_counts.get(i, 0)
                if cnt > 0:
                    rx_labels.append(rx_lblnames[i])
                    rx_values.append(cnt)

            if rx_values:
                colors_used = [rx_colors[rx_lblnames.index(l)]
                               for l in rx_labels]
                x = np.arange(len(rx_labels))
                bars2 = ax2.bar(x, rx_values,
                                color=colors_used,
                                width=0.5, edgecolor='none')
                # 막대 위에 수치 표시
                for bar, val in zip(bars2, rx_values):
                    ax2.text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 0.5,
                             f'{val}개', ha='center', va='bottom',
                             fontsize=9, color=TEXT)

                ax2.set_xticks(x)
                ax2.set_xticklabels(rx_labels, fontsize=9)
                ax2.set_ylabel('Node 수 (개)', fontsize=9)
                ax2.set_title('수신 GW 수별 Node 분포 (중첩도)',
                              fontsize=11, fontweight='bold', pad=8)
                ax2.grid(True, axis='y')

            fig.tight_layout(pad=1.5)
            self.canvas_ovlp.draw()

    # ── 탭 3: SF 분포 ────────────────────────────────────────

    def _draw_sf(self):
        """
        SF(Spreading Factor) 분포.
        좌: ADR SF 분포 파이 차트
        우: SF별 커버리지 비율 가로 막대 그래프
        """
        result = self._result
        if not result or not result.nodes:
            return

        SF_COLORS = {
            7:'#FF4444', 8:'#FF8C00', 9:'#FFD700',
            10:'#00C94A', 11:'#4f8ef7', 12:'#9B59B6',
        }
        SF_SENS = {
            7:-123.0, 8:-126.0, 9:-129.0,
            10:-132.0, 11:-134.5, 12:-137.0,
        }

        with plt.rc_context(STYLE_MPL):
            fig = self.canvas_sf.fig
            fig.clear()

            ax1 = fig.add_subplot(121)   # 파이 차트
            ax2 = fig.add_subplot(122)   # 막대 그래프

            # ── 좌: ADR SF 분포 파이 차트 ────────────────────
            adr_dist = getattr(result, 'adr_sf_distribution', {})
            sf_labels = []
            sf_values = []
            sf_colors = []

            for sf in range(7, 13):
                cnt = adr_dist.get(sf, 0)
                if cnt > 0:
                    sf_labels.append(f'SF{sf}\n({cnt}개)')
                    sf_values.append(cnt)
                    sf_colors.append(SF_COLORS[sf])

            if sf_values:
                wedges, texts, autotexts = ax1.pie(
                    sf_values,
                    labels=sf_labels,
                    colors=sf_colors,
                    autopct='%1.1f%%',
                    startangle=90,
                    pctdistance=0.75,
                    wedgeprops={'edgecolor': '#181b22', 'linewidth': 1.5},
                )
                for at in autotexts:
                    at.set_fontsize(8)
                    at.set_color('#181b22')
                for t in texts:
                    t.set_fontsize(8)

                ax1.set_title('ADR SF 분포\n(수신전력 기준)',
                              fontsize=11, fontweight='bold', pad=10)

            # ── 우: SF별 커버리지 비율 막대 ──────────────────
            n_total = result.n_total
            sf_list = list(range(7, 13))
            pct_list = []

            for sf, sens in SF_SENS.items():
                n_sf = sum(1 for nd in result.nodes
                           if nd.best_pr >= sens)
                pct_list.append(n_sf / n_total * 100 if n_total else 0)

            colors_bar = [SF_COLORS[sf] for sf in sf_list]
            y = np.arange(len(sf_list))

            bars = ax2.barh([f'SF{sf}' for sf in sf_list[::-1]],
                            pct_list[::-1],
                            color=colors_bar[::-1],
                            height=0.5, edgecolor='none')

            for bar, pct in zip(bars, pct_list[::-1]):
                ax2.text(bar.get_width() + 0.5,
                         bar.get_y() + bar.get_height()/2,
                         f'{pct:.1f}%', va='center', ha='left',
                         fontsize=9, color=TEXT)

            ax2.set_xlim(0, 110)
            ax2.set_xlabel('커버리지 비율 (%)', fontsize=9)
            ax2.set_title('SF별 커버리지 비율\n(Pr 기준 누적)',
                          fontsize=11, fontweight='bold', pad=10)
            ax2.grid(True, axis='x')

            fig.tight_layout(pad=1.5)
            self.canvas_sf.draw()

    # ── 탭 4: SNR 분포 ───────────────────────────────────────

    def _draw_snr(self):
        """
        SNR 및 SNR 마진 분포.
        상단: SNR 히스토그램 (커버 Node)
        하단: SNR 마진 히스토그램 + 마진=0 기준선
        """
        result = self._result
        if not result or not result.nodes:
            return

        snr_list    = [nd.best_snr    for nd in result.nodes
                       if nd.covered and nd.best_snr > -999]
        margin_list = [nd.snr_margin  for nd in result.nodes
                       if nd.covered and nd.snr_margin > -999]

        if not snr_list:
            return

        with plt.rc_context(STYLE_MPL):
            fig = self.canvas_snr.fig
            fig.clear()

            ax1 = fig.add_subplot(211)   # SNR
            ax2 = fig.add_subplot(212)   # SNR 마진

            # ── 상단: SNR 히스토그램 ─────────────────────────
            bins_snr = np.arange(-30, 50, 2)
            ax1.hist(snr_list, bins=bins_snr,
                     color='#4f8ef7', alpha=0.8,
                     edgecolor='none',
                     label=f'커버 Node ({len(snr_list)}개)')
            ax1.axvline(np.mean(snr_list),
                        color='#FFD700', linestyle='--',
                        linewidth=2.0,
                        label=f'평균 {np.mean(snr_list):.1f}dB')
            ax1.axvline(0, color='#FF4444', linestyle=':',
                        linewidth=1.5, label='SNR=0dB 기준')

            ax1.set_xlabel('SNR (dB)', fontsize=10)
            ax1.set_ylabel('Node 수 (개)', fontsize=10)
            ax1.set_title('SNR 분포 (커버 Node)',
                          fontsize=12, fontweight='bold', pad=8)
            ax1.legend(fontsize=9)
            ax1.grid(True, axis='y')

            # ── 하단: SNR 마진 히스토그램 ────────────────────
            if margin_list:
                bins_mg = np.arange(-15, 50, 2)

                # 마진 양수(성공) / 음수(실패) 분리
                mg_ok   = [m for m in margin_list if m >= 0]
                mg_fail = [m for m in margin_list if m <  0]

                if mg_ok:
                    ax2.hist(mg_ok, bins=bins_mg,
                             color='#00C94A', alpha=0.8,
                             edgecolor='none',
                             label=f'성공 ({len(mg_ok)}개, 마진≥0dB)')
                if mg_fail:
                    ax2.hist(mg_fail, bins=bins_mg,
                             color='#FF4444', alpha=0.8,
                             edgecolor='none',
                             label=f'실패 ({len(mg_fail)}개, 마진<0dB)')

                ax2.axvline(0, color='#FFD700',
                            linestyle='--', linewidth=2.0,
                            label='마진=0dB 기준')
                ax2.axvline(np.mean(margin_list),
                            color='#9B59B6', linestyle='--',
                            linewidth=1.5,
                            label=f'평균 {np.mean(margin_list):.1f}dB')

                ax2.set_xlabel('SNR 마진 (dB)', fontsize=10)
                ax2.set_ylabel('Node 수 (개)',   fontsize=10)
                ax2.set_title('SNR 마진 분포 (통신 성공/실패 기준)',
                              fontsize=12, fontweight='bold', pad=8)
                ax2.legend(fontsize=9)
                ax2.grid(True, axis='y')

            fig.tight_layout(pad=1.5)
            self.canvas_snr.draw()

    # ── 현재 탭 이미지 저장 ──────────────────────────────────

    def _save_current(self):
        """현재 탭의 그래프를 PNG로 저장."""
        from PyQt5.QtWidgets import QFileDialog
        tab_idx = self.tabs.currentIndex()
        canvas_map = {
            0: (self.canvas_rssi, "rssi_distribution"),
            1: (self.canvas_ovlp, "overlap_distribution"),
            2: (self.canvas_sf,   "sf_distribution"),
            3: (self.canvas_snr,  "snr_distribution"),
        }
        canvas, default_name = canvas_map.get(tab_idx, (None, "graph"))
        if canvas is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "그래프 저장", f"{default_name}.png",
            "PNG (*.png);;SVG (*.svg)")
        if not path:
            return

        try:
            canvas.fig.savefig(path, dpi=150,
                               bbox_inches='tight',
                               facecolor='#181b22')
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "완료",
                f"저장 완료:\n{path}")
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "오류", f"저장 실패:\n{e}")