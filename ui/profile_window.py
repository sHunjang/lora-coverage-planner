# ui/profile_window.py — GW↔Node 지형 단면도 창
from __future__ import annotations
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QGroupBox, QSizePolicy,
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# 한글 폰트 설정
import matplotlib.font_manager as fm
def _set_korean_font():
    candidates = ['Malgun Gothic', 'NanumGothic', 'AppleGothic', 'NotoSansCJK']
    available  = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams['font.family'] = name
            return
    # 한글 폰트 없으면 영문으로 대체
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
_set_korean_font()

from ui.dialogs import DARK, PANEL, TEXT, MUTED, BORDER, STYLE_DLG
from core.coverage import GWEntry, NodeEntry

STYLE = STYLE_DLG + f"""
QComboBox {{
    background:{PANEL}; color:{TEXT};
    border:1px solid {BORDER}; border-radius:4px;
    padding:4px 8px; min-height:26px; min-width:120px;
}}
QComboBox QAbstractItemView {{
    background:{PANEL}; color:{TEXT};
    selection-background-color:#253a5a;
}}
"""


def _fresnel_radius(d1, d2, d_total, fc_mhz):
    """1차 Fresnel 반경 (m)."""
    lam = 3e8 / (fc_mhz * 1e6)
    d1c = np.clip(d1, 1e-3, None)
    d2c = np.clip(d2, 1e-3, None)
    return np.sqrt(lam * d1c * d2c / (d1c + d2c))


class ProfileWindow(QDialog):
    """GW ↔ Node 지형 단면도."""

    def __init__(self, spatial, gws: list[GWEntry],
                 nodes: list[NodeEntry], fc_mhz=915.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("지형 단면도 (Profile)")
        self.setStyleSheet(STYLE)
        self.resize(860, 560)
        self.setWindowFlag(Qt.Window)

        self.spatial  = spatial
        self.gws      = [g for g in gws if g.enabled]
        self.nodes    = nodes
        self.fc       = fc_mhz
        self._build()
        self._draw()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        # 선택 컨트롤
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("GW:"))
        self.cb_gw = QComboBox()
        for g in self.gws:
            self.cb_gw.addItem(g.callsign)
        ctrl.addWidget(self.cb_gw)

        ctrl.addWidget(QLabel("  →  Node:"))
        self.cb_nd = QComboBox()
        for n in self.nodes:
            self.cb_nd.addItem(n.callsign)
        ctrl.addWidget(self.cb_nd)

        self.btn_draw = QPushButton("단면도 그리기")
        self.btn_draw.setStyleSheet(
            f"QPushButton{{background:#1c3a5a;color:#7ab8e8;"
            f"border:1px solid #2a5a8a;border-radius:4px;"
            f"padding:5px 14px;font-size:11px;}}"
            f"QPushButton:hover{{background:#254d78;}}")
        self.btn_draw.clicked.connect(self._draw)
        ctrl.addWidget(self.btn_draw)
        ctrl.addStretch()
        lay.addLayout(ctrl)

        # Matplotlib 캔버스
        self.fig = Figure(figsize=(10, 5), dpi=96,
                          facecolor='#1c1f26')
        self.ax  = self.fig.add_subplot(111)
        self.ax.set_facecolor('#252930')
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self.canvas)

        # 결과 라벨
        self.lbl = QLabel("")
        self.lbl.setStyleSheet(
            f"color:{MUTED};font-size:11px;padding:2px 4px;")
        lay.addWidget(self.lbl)

    def _draw(self):
        gi = self.cb_gw.currentIndex()
        ni = self.cb_nd.currentIndex()
        if gi < 0 or ni < 0 or gi >= len(self.gws) or ni >= len(self.nodes):
            return
        gw = self.gws[gi]
        nd = self.nodes[ni]
        self._plot(gw, nd)

    def _plot(self, gw: GWEntry, nd: NodeEntry):
        self.ax.cla()
        self.ax.set_facecolor('#252930')
        self.fig.patch.set_facecolor('#1c1f26')

        # 좌표 변환
        gx, gy = self.spatial.lonlat_to_xy(gw.lon, gw.lat)
        nx, ny = self.spatial.lonlat_to_xy(nd.lon, nd.lat)
        gx, gy, nx, ny = float(gx), float(gy), float(nx), float(ny)

        d_total = np.hypot(nx - gx, ny - gy)
        if d_total < 1:
            self.lbl.setText("GW와 Node가 너무 가깝습니다.")
            return

        # DEM 단면 샘플링 (200포인트)
        n_samples = 200
        xs = np.linspace(gx, nx, n_samples)
        ys = np.linspace(gy, ny, n_samples)
        dists = np.linspace(0, d_total, n_samples)

        dem   = self.spatial.dem
        ox, oy = self.spatial.ox, self.spatial.oy
        res    = self.spatial.res
        rows_  = self.spatial.dem_rows
        cols_  = self.spatial.dem_cols

        cols_idx = np.clip(((xs - ox) / res).astype(int), 0, cols_ - 1)
        rows_idx = np.clip(((oy - ys) / res).astype(int), 0, rows_ - 1)
        elevs    = dem[rows_idx, cols_idx]
        elevs    = np.where(np.isnan(elevs), 50.0, elevs)

        # GW/Node 안테나 높이 반영한 시선(LOS) 직선
        gw_elev  = elevs[0]  + gw.hb_m
        nd_elev  = elevs[-1] + nd.hm_m
        los_line = gw_elev + (nd_elev - gw_elev) * (dists / d_total)

        # Fresnel 1차 존 반경
        fres = _fresnel_radius(dists, d_total - dists, d_total, self.fc)

        # LOS 차단 구간 (지형이 시선을 막는 구간)
        blocked = elevs > los_line - 0.01

        # ── 그리기 ────────────────────────────────────────────
        dist_km = dists / 1000

        # 지형 채움
        self.ax.fill_between(
            dist_km, 0, elevs,
            color='#3a4a2a', alpha=0.85, linewidth=0, zorder=2,
            label='Terrain')
        self.ax.plot(dist_km, elevs,
                     color='#6a9a4a', linewidth=1.2, zorder=3)

        # Fresnel 존 (시선 ± 반경)
        fres_upper = los_line + fres
        fres_lower = los_line - fres
        self.ax.fill_between(
            dist_km, fres_lower, fres_upper,
            color='#4f8ef7', alpha=0.12, zorder=1,
            label='Fresnel Zone 1')
        self.ax.plot(dist_km, fres_upper,
                     color='#4f8ef7', linewidth=0.6,
                     linestyle='--', alpha=0.5, zorder=1)
        self.ax.plot(dist_km, fres_lower,
                     color='#4f8ef7', linewidth=0.6,
                     linestyle='--', alpha=0.5, zorder=1)

        # 시선(LOS)
        self.ax.plot(dist_km, los_line,
                     color='#00C94A', linewidth=1.5,
                     zorder=4, label='시선(LOS)')

        # LOS 차단 구간 강조
        if blocked.any():
            self.ax.fill_between(
                dist_km, los_line, elevs,
                where=blocked,
                color='#FF4444', alpha=0.5, zorder=5,
                label='LOS Blocked')
            self.ax.plot(
                dist_km[blocked], elevs[blocked],
                color='#FF4444', linewidth=2.0, zorder=6)

        # Fresnel 침범 구간 (지형이 Fresnel 존 안에 있지만 LOS는 통과)
        fresnel_violated = (elevs > fres_lower) & ~blocked
        if fresnel_violated.any():
            self.ax.fill_between(
                dist_km, fres_lower, elevs,
                where=fresnel_violated,
                color='#FF8C00', alpha=0.35, zorder=4,
                label='Fresnel Violated')

        # GW / Node 마커
        self.ax.plot(0, gw_elev, marker='^',
                     color='#FFD700', markersize=10, zorder=7)
        self.ax.plot(dist_km[-1], nd_elev, marker='o',
                     color='#FF69B4', markersize=9, zorder=7)

        # 라벨
        self.ax.annotate(
            f'{gw.callsign}\n({gw_elev:.0f}m)',
            xy=(0, gw_elev), xytext=(dist_km[-1]*0.03, gw_elev + fres.max()*0.3),
            color='#FFD700', fontsize=9,
            arrowprops=dict(arrowstyle='->', color='#FFD700', lw=0.8))
        self.ax.annotate(
            f'{nd.callsign}\n({nd_elev:.0f}m)',
            xy=(dist_km[-1], nd_elev),
            xytext=(dist_km[-1]*0.88, nd_elev + fres.max()*0.3),
            color='#FF69B4', fontsize=9,
            arrowprops=dict(arrowstyle='->', color='#FF69B4', lw=0.8))

        # 축 스타일
        self.ax.set_xlabel('Distance (km)', color='#a0a8be', fontsize=10)
        self.ax.set_ylabel('Elevation (m)', color='#a0a8be', fontsize=10)
        self.ax.tick_params(colors='#a0a8be', labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_edgecolor('#2a2f3b')
        self.ax.grid(True, color='#2a2f3b',
                     linestyle='--', alpha=0.5, linewidth=0.5)
        self.ax.legend(
            loc='upper right', fontsize=8,
            facecolor='#1e2130', edgecolor='#2a2f3b',
            labelcolor='#a0a8be')

        # 결과 요약 라벨
        los_ok   = not blocked.any()
        n_block  = int(blocked.sum())
        max_pen  = float(np.max(elevs - fres_lower)) if fresnel_violated.any() else 0
        fres_ok  = max_pen <= 0

        status = "✓ LOS Clear" if los_ok else f"✗ LOS Blocked ({n_block} pts)"
        fstatus= "✓ Fresnel OK" if fres_ok else f"△ Fresnel Violated {max_pen:.1f}m"
        self.lbl.setText(
            f"Distance: {d_total/1000:.2f}km  |  {status}  |  {fstatus}  |  "
            f"GW: {elevs[0]:.0f}m + {gw.hb_m:.0f}m  |  "
            f"Node: {elevs[-1]:.0f}m + {nd.hm_m:.0f}m")

        self.fig.tight_layout(pad=1.2)
        self.canvas.draw()
