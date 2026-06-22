# ui/main_window.py
from __future__ import annotations
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QStatusBar, QLabel,
    QToolBar, QAction, QApplication, QSizePolicy,
)
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QThread, QSize
from PyQt5.QtGui import QCursor

import json, os

from ui.map_widget       import MapWidget
from ui.gw_list_window   import GWListWindow
from ui.node_list_window import NodeListWindow
from ui.result_panel     import ResultPanel
from core.coverage       import CoverageEngine, GWEntry
from core.utils import SF_SENS

DARK  = "#181b22"
PANEL = "#1e2130"
TEXT  = "#e0e4ef"
MUTED = "#7a8099"

TOOLBAR_STYLE = f"""
QToolBar {{
    background:{PANEL};
    border-bottom:1px solid #2a2f3b;
    spacing:4px; padding:4px 8px;
}}
QToolButton {{
    background:#252930; color:{TEXT};
    border:1px solid #2a2f3b; border-radius:6px;
    padding:8px 20px; font-size:12px; min-width:110px;
}}
QToolButton:hover  {{ background:#2e3545; border-color:#4f8ef7; }}
QToolButton:pressed {{ background:#1c2535; }}
QToolButton:checked {{
    background:#1c3a5a; color:#7ab8e8; border-color:#4f8ef7;
}}
"""

# GW별 히트맵 색상 팔레트 (hex)
GW_HEX_COLORS = [
    '#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22',
    '#c0392b', '#2980b9', '#27ae60', '#8e44ad', '#17a589',
    '#e91e8c', '#5dade2', '#58d68d', '#f0e68c', '#2c3e50',
]


class CoverageWorker(QObject):
    sig_done = pyqtSignal(object)
    sig_err  = pyqtSignal(str)

    def __init__(self, spatial, gws, nodes, settings=None):
        super().__init__()
        self.spatial  = spatial
        self.gws      = gws
        self.nodes    = nodes
        self.settings = settings or {}

    def run(self):
        try:
            from core.coverage import CoverageEngine
            env  = self.settings.get('env', 2)
            if env is None:
                env = 2

            fc   = self.settings.get('fc_mhz', 915.0)
            nsmp = self.settings.get('n_samples', 100)
            eng    = CoverageEngine(self.spatial, env=env,
                                    fc=fc, n_samples=nsmp,
                                    settings=self.settings)
            result = eng.run(self.gws, self.nodes)
            self.sig_done.emit(result)
        except Exception:
            import traceback
            self.sig_err.emit(traceback.format_exc())


class HeatmapWorker(QObject):
    sig_log  = pyqtSignal(str)
    sig_done = pyqtSignal(list)
    sig_err  = pyqtSignal(str)

    def __init__(self, spatial, gws, settings, env=2, fc=915.0):
        super().__init__()
        self.spatial  = spatial
        self.gws      = gws
        self.settings = settings
        self.env      = env
        self.fc       = fc

    def run(self):
        try:
            import numpy as np
            from core.coverage import CoverageEngine

            eng    = CoverageEngine(self.spatial, self.env, self.fc,
                                    n_samples=self.settings.get('n_samples', 100),
                                    settings=self.settings)
            min_rx = self.settings.get('min_rx', -126.6)
            step   = float(self.settings.get('heatmap_step', 0.0015))

            color_levels = self.settings.get('color_levels', [
                {'pr': -90,  'color': '#FF2020'},
                {'pr': -100, 'color': '#FF8C00'},
                {'pr': -110, 'color': '#FFD700'},
                {'pr': -120, 'color': '#00C94A'},
                {'pr': -130, 'color': '#4f8ef7'},
            ])
            SF_COLORS = {
                7: '#FF2020', 8: '#FF8C00', 9: '#FFD700',
                10: '#00C94A', 11: '#4f8ef7', 12: '#9B59B6',
            }

            radius_km = float(self.settings.get('radius_km', 25.0))
            
            hms = []

            # ── 히트맵 계산 (단일/다중 공통) ─────────────────
            # _nodes, _result는 settings에 담겨 eng 내부에서 자동 사용
            if len(self.gws) > 1:
                self.sig_log.emit(
                    f"{len(self.gws)}개 GW 합성 히트맵 계산 중...")
                hm = eng.heatmap_combined(
                    self.gws, min_rx, step=step,
                    cb=self.sig_log.emit,
                    radius_km=radius_km)
            else:
                gw = self.gws[0]
                self.sig_log.emit(f"{gw.callsign} 히트맵 계산 중...")
                hm = eng.heatmap(
                    gw, min_rx, step=step,
                    cb=self.sig_log.emit,
                    radius_km=radius_km)

            # ── 등고선 / SF 레이어 계산 (공통) ───────────────
            ps = hm.get('ps')
            cm = hm.get('cm')
            contours = []

            if ps is not None and cm is not None:
                lmin     = hm.get('lon_min', 0)
                latmin   = hm.get('lat_min', 0)
                act_step = hm.get('step', step)
                lon_ax   = np.linspace(lmin,
                                    lmin + act_step * ps.shape[1],
                                    ps.shape[1])
                lat_ax   = np.linspace(latmin,
                                    latmin + act_step * ps.shape[0],
                                    ps.shape[0])
                pr_m     = np.where(cm, ps, np.nan)
                ps_in_cm = ps[cm]

                if len(ps_in_cm) > 0:
                    pr_min_in = float(ps_in_cm.min())
                    pr_max_in = float(ps_in_cm.max())

                    # 등고선
                    for lv in color_levels:
                        pv = float(lv['pr'])
                        if pv < pr_min_in or pv > pr_max_in:
                            continue
                        allsegs = _calc_contour_segments(
                            lon_ax, lat_ax, pr_m, pv)
                        if not allsegs:
                            continue
                        segs, lpts = [], []
                        for col_segs in allsegs:
                            for seg in col_segs:
                                if len(seg) < 4:
                                    continue
                                d = np.diff(seg, axis=0)
                                if float(np.sqrt(
                                        (d**2).sum(axis=1)).sum()) < act_step:
                                    continue
                                pts = [[float(p[1]), float(p[0])]
                                    for p in seg]
                                segs.append(pts)
                                mid = len(pts) // 2
                                lpts.append({
                                    'lat' : pts[mid][0],
                                    'lon' : pts[mid][1],
                                    'text': f'{pv:.0f} dBm',
                                })
                        if segs:
                            contours.append({
                                'color'    : lv['color'],
                                'weight'   : 2.0,
                                'label'    : f'{pv:.0f} dBm',
                                'segments' : segs,
                                'label_pts': lpts,
                            })

                    # SF 레이어
                    sf_layers = []
                    for sf, sens in SF_SENS.items():
                        if not (ps[cm] >= sens).any():
                            continue
                        pr_sf      = np.where(cm, ps, np.nan)
                        allsegs_sf = _calc_contour_segments(
                            lon_ax, lat_ax, pr_sf, sens)
                        if not allsegs_sf:
                            continue
                        segs_sf = []
                        for col_segs in allsegs_sf:
                            for seg in col_segs:
                                if len(seg) < 4:
                                    continue
                                d = np.diff(seg, axis=0)
                                if float(np.sqrt(
                                        (d**2).sum(axis=1)).sum()) < act_step:
                                    continue
                                segs_sf.append(
                                    [[float(p[1]), float(p[0])]
                                    for p in seg])
                        if segs_sf:
                            sf_layers.append({
                                'sf'      : sf,
                                'color'   : SF_COLORS[sf],
                                'segments': segs_sf,
                                'label'   : f'SF{sf} ({sens:.1f} dBm)',
                            })
                    hm['sf_layers'] = sf_layers

            hm['contours'] = contours
            hms.append(hm)
            self.sig_done.emit(hms)

        except Exception:
            import traceback
            self.sig_err.emit(traceback.format_exc())


def _calc_contour_segments(lon_ax, lat_ax, pr_m, level):
    """스레드 안전한 등고선 계산."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    try:
        cs = ax.contour(lon_ax, lat_ax, pr_m, levels=[level])
        return cs.allsegs
    except Exception:
        return []
    finally:
        plt.close(fig)


class MainWindow(QMainWindow):
    def __init__(self, shp_path, dem_path, spatial=None):
        super().__init__()
        self.setWindowTitle("LoRa Coverage Planner")
        geo = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(int(geo.width()*0.03), int(geo.height()*0.03),
                         int(geo.width()*0.94), int(geo.height()*0.94))
        self.setStyleSheet(f"QMainWindow{{background:{DARK};}}")

        self.spatial        = None
        self._thread        = None
        self._cov_thread    = None
        self._cov_worker    = None
        self._legend_levels = None
        self._legend_win    = None
        self._result        = None
        self._heatmaps      = []
        self._shp           = shp_path
        self._dem           = dem_path
        self._gw_win        = None
        self._node_win      = None
        self._opt_win       = None
        self._history      = []    # Snapshot 히스토리 (최대 10개)
        self._compare_win  = None  # CompareWindow 인스턴스
        self._field_data    = []     # 실측 데이터 [{lat, lon, rssi, snr}, ...]

        from ui.settings_window import load_settings
        self._settings = load_settings()

        self._build_ui()
        if spatial is not None:
            self.spatial = spatial
            self.map_w.set_bounds(tuple(spatial.bounds))
            self.map_w.refresh()
        else:
            self._load_spatial()
        self._load_session()

    def closeEvent(self, event):
        # 실행 중인 스레드 안전하게 종료
        for t in (self._thread, self._cov_thread):
            try:
                if t and t.isRunning():
                    t.quit()
                    t.wait(3000)
            except RuntimeError:
                pass
        self._save_session()
        event.accept()

    def _save_session(self):
        """종료 시 GW/Node 위치만 session.json에 저장."""
        try:
            gws   = self._gw_win.get_gws()    if self._gw_win   else []
            nodes = self._node_win.get_nodes() if self._node_win else []

            data = {
                'gws': [
                    {
                        'callsign': g.callsign,
                        'lon'     : float(g.lon),
                        'lat'     : float(g.lat),
                        'pt_dbm'  : float(g.pt_dbm),
                        'gt_dbi'  : float(g.gt_dbi),
                        'lt_db'   : float(g.lt_db),
                        'hb_m'    : float(g.hb_m),
                        'enabled' : bool(g.enabled),
                    }
                    for g in gws
                ],
                'nodes': [
                    {
                        'callsign'      : n.callsign,
                        'lon'           : float(n.lon),
                        'lat'           : float(n.lat),
                        'gr_dbi'        : float(n.gr_dbi),
                        'lr_db'         : float(n.lr_db),
                        'hm_m'          : float(n.hm_m),
                        'min_rx_dbm'    : float(n.min_rx_dbm),
                        'indoor_loss_db': float(getattr(n, 'indoor_loss_db', 0.0)),
                    }
                    for n in nodes
                ],
            }

            session_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..', 'session.json')
            with open(session_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"[SESSION] 저장: GW {len(gws)}개, Node {len(nodes)}개")

        except Exception as e:
            print(f"[SESSION] 저장 실패: {e}")

    def _load_session(self):
        """시작 시 session.json에서 GW/Node 위치만 복원."""
        try:
            session_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..', 'session.json')

            if not os.path.exists(session_path):
                return

            with open(session_path, encoding='utf-8') as f:
                data = json.load(f)

            gws   = data.get('gws',   [])
            nodes = data.get('nodes', [])

            from core.coverage import GWEntry, NodeEntry

            if gws:
                if self._gw_win is None:
                    from ui.gw_list_window import GWListWindow
                    self._gw_win = GWListWindow(self)
                    self._gw_win.sig_coverage_requested.connect(self._run_heatmap)
                    self._gw_win.sig_coverage_clear.connect(self._clear_heatmap)
                    self._gw_win.sig_coverage_analyze.connect(self._run_coverage)
                    self._gw_win.sig_map_refresh.connect(self._refresh_map)
                    self._gw_win.sig_env_map_requested.connect(self._run_env_map)

                self._gw_win._gws = [
                    GWEntry(
                        callsign = g.get('callsign', 'GW'),
                        lon      = float(g.get('lon',    127.1)),
                        lat      = float(g.get('lat',    37.4)),
                        pt_dbm   = float(g.get('pt_dbm', 14.0)),
                        gt_dbi   = float(g.get('gt_dbi', 2.15)),
                        lt_db    = float(g.get('lt_db',  0.0)),
                        hb_m     = float(g.get('hb_m',   15.0)),
                        enabled  = bool (g.get('enabled', True)),
                    )
                    for g in gws
                ]
                self._gw_win._refresh_table(suppress_map=True)

            if nodes:
                if self._node_win is None:
                    from ui.node_list_window import NodeListWindow
                    self._node_win = NodeListWindow(self)
                    self._node_win.sig_map_refresh.connect(self._refresh_map)

                self._node_win._nodes = [
                    NodeEntry(
                        callsign       = n.get('callsign', 'Node'),
                        lon            = float(n.get('lon',           127.1)),
                        lat            = float(n.get('lat',           37.4)),
                        gr_dbi         = float(n.get('gr_dbi',        2.15)),
                        lr_db          = float(n.get('lr_db',         0.0)),
                        hm_m           = float(n.get('hm_m',          1.5)),
                        min_rx_dbm     = float(n.get('min_rx_dbm',   -126.6)),
                        indoor_loss_db = float(n.get('indoor_loss_db', 0.0)),
                    )
                    for n in nodes
                ]
                self._node_win._refresh_table(suppress_map=True)

            if gws or nodes:
                self._refresh_map()
                self.status.showMessage(
                    f"세션 복원: GW {len(gws)}개, Node {len(nodes)}개")

        except Exception as e:
            print(f"[SESSION] 복원 실패: {e}")

    def _build_ui(self):
        tb = QToolBar()
        tb.setMovable(False)
        tb.setIconSize(QSize(20, 20))
        tb.setStyleSheet(TOOLBAR_STYLE)
        self.addToolBar(Qt.TopToolBarArea, tb)

        act_gw     = QAction("📡  GW 목록",      self)
        act_node   = QAction("📶  단말 목록",     self)
        act_opt    = QAction("⚙   GW 최적 배치", self)
        act_legend = QAction("🎨  범례 설정",     self)
        act_graph = QAction("📈  그래프", self)
        act_cfg    = QAction("🔧  설정",           self)
        act_dist   = QAction("📏  거리 측정", self, checkable=True)
        act_proj_save = QAction("🗂  프로젝트 저장", self)
        act_proj_load = QAction("📁  프로젝트 열기", self) 
        act_save   = QAction("💾  결과 저장",      self)
        act_load   = QAction("📂  결과 불러오기",  self)
        act_field  = QAction("📡  실측 데이터", self)
        act_report = QAction("📄  리포트",  self)
        # act_manual  = QAction("📖  매뉴얼", self)
        act_compare = QAction("📊  결과 비교", self)
        act_log     = QAction("📜  로그",       self)
        act_perf    = QAction("⚡  성능",       self)
        act_validate= QAction("✔  데이터 검증", self)

        for a in [act_gw, act_node, act_opt, act_legend, act_graph, act_cfg,
                  act_dist,
                    act_proj_save, act_proj_load,
                  act_save, 
                  act_load, 
                  act_report, act_compare, act_field,
                  act_log, act_perf, act_validate]:
            tb.addAction(a)

        act_gw.triggered.connect(self._open_gw_list)
        act_node.triggered.connect(self._open_node_list)
        act_opt.triggered.connect(self._open_optimize)
        act_legend.triggered.connect(self._open_legend)
        act_graph.triggered.connect(self._open_graph)
        act_dist.triggered.connect(self._toggle_measure)
        self._measuring   = False
        self._measure_pts = []
        act_cfg.triggered.connect(self._open_settings)
        act_save.triggered.connect(self._save_result)
        act_load.triggered.connect(self._load_result)
        act_report.triggered.connect(self._open_report)
        # act_manual.triggered.connect(self._open_manual)
        act_compare.triggered.connect(self._open_compare)
        act_field.triggered.connect(self._open_field_data)
        act_proj_save.triggered.connect(self._save_project)
        act_proj_load.triggered.connect(self._load_project)
        act_log.triggered.connect(self._open_log_viewer)
        act_perf.triggered.connect(self._open_perf_monitor)
        act_validate.triggered.connect(self._open_data_validation)

        from PyQt5.QtWidgets import QSplitter
        splitter = QSplitter(Qt.Horizontal)
        self.map_w        = MapWidget()
        self.result_panel = ResultPanel()
        self.result_panel.setMaximumWidth(260)
        self.result_panel.setMinimumWidth(200)
        splitter.addWidget(self.map_w)
        splitter.addWidget(self.result_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1200, 240])
        splitter.setStyleSheet(
            f"QSplitter::handle{{background:#2a2f3b;width:2px;}}")
        self.setCentralWidget(splitter)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.lbl = QLabel("─")
        self.lbl.setStyleSheet(
            f"color:{MUTED};font-size:12px;padding:4px 12px;")
        self.status.addPermanentWidget(self.lbl)

        self.map_w.sig_map_clicked.connect(self._on_map_clicked)
        self.map_w.sig_gw_dragged.connect(self._on_gw_dragged)
        self.map_w.sig_nd_dragged.connect(self._on_nd_dragged)
        self.map_w.sig_map_right_clicked.connect(self._on_map_right_clicked)

    # ── 창 열기 ─────────────────────────────────────────────

    def _open_gw_list(self):
        self._ensure_gw_win()
        self._gw_win.show(); self._gw_win.raise_()

    def _open_node_list(self):
        self._ensure_node_win()
        self._node_win.show(); self._node_win.raise_()

    def _open_settings(self):
        from ui.settings_window import SettingsWindow
        dlg = SettingsWindow(self)
        dlg.sig_settings_changed.connect(self._on_settings_changed)
        dlg.exec_()

    def _on_settings_changed(self, settings: dict):
        self._settings = settings
        self._refresh_map()
        self.status.showMessage("설정이 적용되었습니다.")

    def _open_optimize(self):
        if self.spatial is None:
            self.status.showMessage("공간 데이터 로드 중..."); return
        nodes = self._node_win.get_nodes() if self._node_win else []
        if not nodes:
            self.status.showMessage("단말 목록에 Node를 먼저 추가하세요.")
            return
        if self._opt_win is None:
            from ui.gw_optimize_window import GWOptimizeWindow
            self._opt_win = GWOptimizeWindow(self.spatial, nodes, self)
            self._opt_win.sig_result_ready.connect(self._on_optimize_done)
        else:
            self._opt_win.set_nodes(nodes)
        self._opt_win.show(); self._opt_win.raise_()

    def _open_legend(self):
        from ui.legend_window import LegendWindow, DEFAULT_LEVELS
        levels = self._legend_levels or DEFAULT_LEVELS
        self._legend_win = LegendWindow(levels=levels, parent=self)
        self._legend_win.sig_levels_changed.connect(self._on_legend_changed)
        self._legend_win.show()
        self._legend_win.raise_()

    def _on_legend_changed(self, levels: list):
        self._legend_levels = levels
        self._settings['color_levels'] = levels
        if levels:
            self._settings['pr_min'] = min(lv['pr'] for lv in levels)
            self._settings['pr_max'] = max(lv['pr'] for lv in levels)

        if self._heatmaps and self._gw_win:
            gws = [g for g in self._gw_win.get_gws() if g.enabled]
            if gws:
                self.status.showMessage("범례 변경 — 히트맵 재계산 중...")
                self._run_heatmap(gws, {})
                return

        self.status.showMessage(
            f"범례 업데이트 완료 — {len(levels)}개 레벨 | "
            f"히트맵 재계산 시 반영됩니다.")
        
    def _open_report(self):
            from ui.report_window import ReportWindow
            gws   = self._gw_win.get_gws()    if self._gw_win   else []
            nodes = self._node_win.get_nodes() if self._node_win else []
            dlg   = ReportWindow(
                result   = self._result,
                gws      = gws,
                nodes    = nodes,
                heatmaps = self._heatmaps,
                settings = self._settings,
                parent   = self,
            )
            dlg.show()

    def _open_graph(self):
        if self._result is None:
            self.status.showMessage("커버리지 분석을 먼저 실행하세요.")
            return
        from ui.graph_window import GraphWindow
        gws   = self._gw_win.get_gws()    if self._gw_win   else []
        nodes = self._node_win.get_nodes() if self._node_win else []
        dlg   = GraphWindow(
            result = self._result,
            gws    = gws,
            nodes  = nodes,
            parent = self,
        )
        dlg.show()

    # def _open_manual(self):
    #     from ui.manual_window import ManualWindow
    #     gws   = self._gw_win.get_gws()    if self._gw_win   else []
    #     nodes = self._node_win.get_nodes() if self._node_win else []
    #     dlg   = ManualWindow(
    #         main_window = self,
    #         result      = self._result,
    #         gws         = gws,
    #         nodes       = nodes,
    #         settings    = self._settings,
    #         parent      = self,
    #     )
    #     dlg.show()

    # ── 프로젝트 파일 (.lorascape) ───────────────────────────

    def _save_project(self):
        """
        현재 작업 전체를 .lorascape 파일로 저장합니다.

        저장 내용:
        - 메타정보 (버전, 저장 일시)
        - GW 목록
        - Node 목록
        - 커버리지 분석 결과
        - 설정값 (전파 모델, 범례 등)
        - 분석 히스토리 스냅샷 (최대 10개)
        - 실측 데이터

        형식: JSON을 ZIP으로 압축한 단일 파일
        """
        import zipfile
        import io
        from datetime import datetime
        from PyQt5.QtWidgets import QFileDialog
        from ui.splash_screen import APP_VERSION

        path, _ = QFileDialog.getSaveFileName(
            self, "프로젝트 저장",
            "project.lorascape",
            "LoRaScape 프로젝트 (*.lorascape)")
        if not path:
            return

        try:
            gws   = self._gw_win.get_gws()    if self._gw_win   else []
            nodes = self._node_win.get_nodes() if self._node_win else []

            # ── 커버리지 결과 직렬화 ──────────────────────────
            def _ser_result(r):
                if r is None:
                    return None
                return {
                    'n_covered'           : int(r.n_covered),
                    'n_total'             : int(r.n_total),
                    'gw_counts'           : {k: int(v)
                                             for k, v in r.gw_counts.items()},
                    'macro_diversity_gain': float(getattr(r,'macro_diversity_gain',0.0)),
                    'avg_n_rx_gw'         : float(getattr(r,'avg_n_rx_gw',0.0)),
                    'adr_sf_distribution' : {str(k): int(v) for k, v in
                                             getattr(r,'adr_sf_distribution',{}).items()},
                    'avg_toa_ms'          : float(getattr(r,'avg_toa_ms',0.0)),
                    'cell_success_rate'   : float(getattr(r,'cell_success_rate',0.0)),
                    'edge_success_rate'   : float(getattr(r,'edge_success_rate',0.0)),
                    'avg_snr'             : float(getattr(r,'avg_snr',0.0)),
                    'avg_snr_margin'      : float(getattr(r,'avg_snr_margin',0.0)),
                    'gw_traffic'          : getattr(r,'gw_traffic',{}),
                    'avg_pdr'             : float(getattr(r,'avg_pdr',100.0)),
                    'n_overloaded_gw'     : int(getattr(r,'n_overloaded_gw',0)),
                    'avg_load_pct'        : float(getattr(r,'avg_load_pct',0.0)),
                    'nodes': [
                        {
                            'covered'    : bool(nd.covered),
                            'best_gw'    : str(nd.best_gw),
                            'best_pr'    : float(nd.best_pr),
                            'gw_prs'     : {k: float(v)
                                            for k, v in nd.gw_prs.items()},
                            'macro_pr'   : float(getattr(nd,'macro_pr',-999.0)),
                            'n_rx_gw'    : int(getattr(nd,'n_rx_gw',0)),
                            'best_snr'   : float(getattr(nd,'best_snr',-999.0)),
                            'snr_margin' : float(getattr(nd,'snr_margin',-999.0)),
                            'link_ok'    : bool(getattr(nd,'link_ok',False)),
                            'adr_sf'     : int(getattr(nd,'adr_sf',12)),
                            'toa_ms'     : float(getattr(nd,'toa_ms',0.0)),
                        }
                        for nd in r.nodes
                    ],
                }

            # ── 히스토리 직렬화 ───────────────────────────────
            history_data = []
            for snap in self._history:
                try:
                    history_data.append(snap.to_dict())
                except Exception as e:
                    print(f"[PROJECT] 스냅샷 직렬화 실패: {e}")

            # ── 전체 프로젝트 데이터 ─────────────────────────
            project = {
                'meta': {
                    'version'   : APP_VERSION,
                    'saved_at'  : datetime.now().isoformat(),
                    'app_name'  : 'LoRaScape',
                    'file_ver'  : '1',
                },
                'gws': [
                    {
                        'callsign': g.callsign,
                        'lon'     : float(g.lon),
                        'lat'     : float(g.lat),
                        'pt_dbm'  : float(g.pt_dbm),
                        'gt_dbi'  : float(g.gt_dbi),
                        'lt_db'   : float(g.lt_db),
                        'hb_m'    : float(g.hb_m),
                        'enabled' : bool(g.enabled),
                    }
                    for g in gws
                ],
                'nodes': [
                    {
                        'callsign'      : n.callsign,
                        'lon'           : float(n.lon),
                        'lat'           : float(n.lat),
                        'gr_dbi'        : float(n.gr_dbi),
                        'lr_db'         : float(n.lr_db),
                        'hm_m'          : float(n.hm_m),
                        'min_rx_dbm'    : float(n.min_rx_dbm),
                        'indoor_loss_db': float(getattr(n,'indoor_loss_db',0.0)),
                    }
                    for n in nodes
                ],
                'result'    : _ser_result(self._result),
                'settings'  : dict(self._settings),
                'history'   : history_data,
                'field_data': self._field_data or [],
            }

            # ── ZIP 압축 저장 ─────────────────────────────────
            json_bytes = json.dumps(
                project, ensure_ascii=False, indent=2
            ).encode('utf-8')

            with zipfile.ZipFile(path, 'w',
                                 compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=6) as zf:
                zf.writestr('project.json', json_bytes)

            size_kb = os.path.getsize(path) / 1024
            self.status.showMessage(
                f"프로젝트 저장 완료: {os.path.basename(path)} "
                f"({size_kb:.0f} KB) | "
                f"GW {len(gws)}개, Node {len(nodes)}개")
            print(f"[PROJECT] 저장: {path} ({size_kb:.0f} KB)")

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.status.showMessage(f"프로젝트 저장 실패: {e}")

    def _load_project(self):
        """
        .lorascape 파일을 열어 작업 전체를 복원합니다.

        복원 내용:
        - GW 목록 + Node 목록
        - 커버리지 분석 결과
        - 설정값
        - 분석 히스토리
        - 실측 데이터
        """
        import zipfile
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        from core.coverage import (
            GWEntry, NodeEntry, CoverageResult, LinkResult)

        path, _ = QFileDialog.getOpenFileName(
            self, "프로젝트 열기", "",
            "LoRaScape 프로젝트 (*.lorascape);;모든 파일 (*.*)")
        if not path:
            return

        try:
            # ── ZIP에서 project.json 추출 ─────────────────────
            with zipfile.ZipFile(path, 'r') as zf:
                if 'project.json' not in zf.namelist():
                    QMessageBox.warning(
                        self, "파일 오류",
                        "유효한 .lorascape 파일이 아닙니다.")
                    return
                json_bytes = zf.read('project.json')

            project = json.loads(json_bytes.decode('utf-8'))

            # ── 버전 확인 ─────────────────────────────────────
            meta = project.get('meta', {})
            file_ver = meta.get('file_ver', '1')
            saved_at = meta.get('saved_at', '알 수 없음')
            app_ver  = meta.get('version',  '알 수 없음')
            print(f"[PROJECT] 열기: {path}")
            print(f"[PROJECT] 저장 버전: {app_ver}, 저장 일시: {saved_at}")

            # ── GW 복원 ───────────────────────────────────────
            gws_data = project.get('gws', [])
            if gws_data:
                self._ensure_gw_win()
                self._gw_win._gws = [
                    GWEntry(
                        callsign = g.get('callsign', 'GW'),
                        lon      = float(g.get('lon',    127.1)),
                        lat      = float(g.get('lat',    37.4)),
                        pt_dbm   = float(g.get('pt_dbm', 14.0)),
                        gt_dbi   = float(g.get('gt_dbi', 2.15)),
                        lt_db    = float(g.get('lt_db',  0.0)),
                        hb_m     = float(g.get('hb_m',   15.0)),
                        enabled  = bool (g.get('enabled', True)),
                    )
                    for g in gws_data
                ]
                self._gw_win._refresh_table(suppress_map=True)

            # ── Node 복원 ─────────────────────────────────────
            nodes_data = project.get('nodes', [])
            if nodes_data:
                self._ensure_node_win()
                self._node_win._nodes = [
                    NodeEntry(
                        callsign       = n.get('callsign', 'Node'),
                        lon            = float(n.get('lon',           127.1)),
                        lat            = float(n.get('lat',           37.4)),
                        gr_dbi         = float(n.get('gr_dbi',        2.15)),
                        lr_db          = float(n.get('lr_db',         0.0)),
                        hm_m           = float(n.get('hm_m',          1.5)),
                        min_rx_dbm     = float(n.get('min_rx_dbm',   -126.6)),
                        indoor_loss_db = float(n.get('indoor_loss_db', 0.0)),
                    )
                    for n in nodes_data
                ]
                self._node_win._refresh_table(suppress_map=True)

            # ── 커버리지 결과 복원 ────────────────────────────
            rd = project.get('result')
            if rd:
                result = CoverageResult(
                    n_covered            = rd.get('n_covered', 0),
                    n_total              = rd.get('n_total',   0),
                    gw_counts            = rd.get('gw_counts', {}),
                    macro_diversity_gain = float(rd.get('macro_diversity_gain',0.0)),
                    avg_n_rx_gw          = float(rd.get('avg_n_rx_gw',         0.0)),
                    adr_sf_distribution  = {int(k): v for k, v in
                                            rd.get('adr_sf_distribution',{}).items()},
                    avg_toa_ms           = float(rd.get('avg_toa_ms',          0.0)),
                    cell_success_rate    = float(rd.get('cell_success_rate',   0.0)),
                    edge_success_rate    = float(rd.get('edge_success_rate',   0.0)),
                    avg_snr              = float(rd.get('avg_snr',             0.0)),
                    avg_snr_margin       = float(rd.get('avg_snr_margin',      0.0)),
                    gw_traffic           = rd.get('gw_traffic',                {}),
                    avg_pdr              = float(rd.get('avg_pdr',           100.0)),
                    n_overloaded_gw      = int  (rd.get('n_overloaded_gw',     0)),
                    avg_load_pct         = float(rd.get('avg_load_pct',        0.0)),
                )
                for nd in rd.get('nodes', []):
                    result.nodes.append(LinkResult(
                        covered    = bool (nd.get('covered',    False)),
                        best_gw    = str  (nd.get('best_gw',    '')),
                        best_pr    = float(nd.get('best_pr',    -999.0)),
                        gw_prs     = {k: float(v)
                                      for k, v in nd.get('gw_prs',{}).items()},
                        macro_pr   = float(nd.get('macro_pr',   -999.0)),
                        n_rx_gw    = int  (nd.get('n_rx_gw',    0)),
                        best_snr   = float(nd.get('best_snr',   -999.0)),
                        snr_margin = float(nd.get('snr_margin', -999.0)),
                        link_ok    = bool (nd.get('link_ok',    False)),
                        adr_sf     = int  (nd.get('adr_sf',     12)),
                        toa_ms     = float(nd.get('toa_ms',     0.0)),
                    ))
                self._result = result
            else:
                self._result = None

            # ── 설정 복원 ─────────────────────────────────────
            saved_settings = project.get('settings', {})
            if saved_settings:
                from ui.settings_window import load_settings, DEFAULT_SETTINGS
                merged = dict(DEFAULT_SETTINGS)
                merged.update(saved_settings)
                self._settings = merged

            # ── 히스토리 복원 ─────────────────────────────────
            self._history = []
            from ui.compare_window import Snapshot
            for d_snap in project.get('history', []):
                try:
                    self._history.append(Snapshot.from_dict(d_snap))
                except Exception as e:
                    print(f"[PROJECT] 스냅샷 복원 실패: {e}")

            # ── 실측 데이터 복원 ──────────────────────────────
            self._field_data = project.get('field_data', [])

            # ── 지도 갱신 ─────────────────────────────────────
            gws   = self._gw_win.get_gws()    if self._gw_win   else []
            nodes = self._node_win.get_nodes() if self._node_win else []
            self.map_w.refresh(
                gws=gws, nodes=nodes,
                result=self._result,
                heatmaps=[],
                selected_gws=[],
                map_tile=self._settings.get('map_tile', 'CartoDB Voyager'),
                field_data=self._field_data or None)

            if self._result:
                self.result_panel.update_result(self._result, gws)
                pct = self._result.coverage_pct
                self.lbl.setText(
                    f"커버리지: {self._result.n_covered}/"
                    f"{self._result.n_total} ({pct:.1f}%)")
            else:
                self.lbl.setText("─")

            # 타이틀 업데이트
            self.setWindowTitle(
                f"LoRa Coverage Planner — {os.path.basename(path)}")

            self.status.showMessage(
                f"프로젝트 열기 완료: {os.path.basename(path)} | "
                f"GW {len(gws)}개, Node {len(nodes)}개, "
                f"히스토리 {len(self._history)}개 | "
                f"저장: {saved_at[:16]}")

        except zipfile.BadZipFile:
            QMessageBox.warning(
                self, "파일 오류",
                "손상된 파일이거나 .lorascape 형식이 아닙니다.")
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.status.showMessage(f"프로젝트 열기 실패: {e}")

    def _add_snapshot(self, result, gws, nodes):
            """분석 결과를 히스토리에 자동 저장 (최대 10개)."""
            from ui.compare_window import Snapshot
            n_gws = len(gws)
            pct   = result.coverage_pct
            label = f"GW {n_gws}개 | {pct:.1f}%"
            snap  = Snapshot(label, result, list(gws), list(nodes))
            self._history.insert(0, snap)
            if len(self._history) > 10:
                self._history.pop()
            if self._compare_win and self._compare_win.isVisible():
                self._compare_win._history = self._history
                self._compare_win._refresh_list()

    def _open_compare(self):
        """비교 창 열기."""
        if not self._history:
            self.status.showMessage(
                "커버리지 분석을 먼저 실행하세요. "
                "분석할 때마다 히스토리에 자동 저장됩니다.")
            return
        if self._compare_win is None:
            from ui.compare_window import CompareWindow
            self._compare_win = CompareWindow(self._history, parent=self)
            self._compare_win.sig_load_snapshot.connect(
                self._on_load_snapshot)
        else:
            self._compare_win._history = self._history
            self._compare_win._refresh_list()
        self._compare_win.show()
        self._compare_win.raise_()

    def _open_log_viewer(self):
        from ui.log_viewer_window import LogViewerWindow
        dlg = LogViewerWindow(self)
        dlg.show()

    def _open_perf_monitor(self):
        from ui.perf_monitor_window import PerfMonitorWindow
        dlg = PerfMonitorWindow(self)
        dlg.show()

    def _open_data_validation(self):
        from ui.data_validation_window import DataValidationWindow
        gws   = self._gw_win.get_gws()    if self._gw_win   else []
        nodes = self._node_win.get_nodes() if self._node_win else []
        dlg = DataValidationWindow(gws, nodes, self.spatial, self)
        dlg.show()

    def _open_field_data(self):
        """실측 데이터 CSV 불러오기 / 초기화."""
        from PyQt5.QtWidgets import QMenu
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background:#1e2130; color:#e0e4ef;
                border:1px solid #2a2f3b; border-radius:6px;
                padding:4px;
            }
            QMenu::item { padding:7px 22px; font-size:12px; }
            QMenu::item:selected { background:#253a5a; color:#7ab8e8; }
            QMenu::separator { height:1px; background:#2a2f3b; margin:4px 8px; }
        """)

        n = len(self._field_data)
        lbl = menu.addAction(
            f"📡 실측 데이터: {n}개 로드됨" if n else "📡 실측 데이터: 없음")
        lbl.setEnabled(False)
        menu.addSeparator()

        act_load  = menu.addAction("📂  CSV 불러오기")
        act_clear = menu.addAction("✕   실측 데이터 초기화")
        act_clear.setEnabled(bool(self._field_data))

        from PyQt5.QtGui import QCursor
        action = menu.exec_(QCursor.pos())
        if action == act_load:
            self._load_field_csv()
        elif action == act_clear:
            self._field_data = []
            self._refresh_map()
            self.status.showMessage("실측 데이터 초기화 완료")
            self.lbl.setText("─")

    def _load_field_csv(self):
        """실측 데이터 CSV 파일을 불러와 지도에 표시합니다.

        CSV 형식:
            필수: lat, lon, rssi
            선택: snr
        """
        import csv
        from PyQt5.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self, "실측 데이터 CSV", "",
            "CSV 파일 (*.csv);;모든 파일 (*.*)")
        if not path:
            return

        field_data = []
        err_count  = 0
        err_msgs   = []

        try:
            with open(path, newline='', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)

                # 헤더 검사
                if reader.fieldnames is None:
                    self.status.showMessage("CSV 파일이 비어있습니다.")
                    return

                fields_lower = [f.lower().strip()
                                for f in reader.fieldnames]
                required = {'lat', 'lon', 'rssi'}
                missing  = required - set(fields_lower)
                if missing:
                    from PyQt5.QtWidgets import QMessageBox
                    QMessageBox.warning(
                        self, "CSV 형식 오류",
                        f"필수 컬럼이 없습니다: {', '.join(missing)}\n\n"
                        f"필수: lat, lon, rssi\n선택: snr")
                    return

                # 컬럼명 정규화 (대소문자 무관)
                col_map = {f.lower().strip(): f
                           for f in reader.fieldnames}

                for i, row in enumerate(reader, start=2):
                    try:
                        lat  = float(row[col_map['lat']])
                        lon  = float(row[col_map['lon']])
                        rssi = float(row[col_map['rssi']])

                        # 선택 컬럼
                        snr = None
                        if 'snr' in col_map:
                            v = row[col_map['snr']].strip()
                            if v:
                                snr = float(v)

                        # 범위 검사
                        if not (-90 <= lat <= 90):
                            raise ValueError(f"위도 범위 초과: {lat}")
                        if not (-180 <= lon <= 180):
                            raise ValueError(f"경도 범위 초과: {lon}")
                        if not (-200 <= rssi <= 0):
                            raise ValueError(f"RSSI 범위 초과: {rssi}")

                        field_data.append({
                            'lat' : lat,
                            'lon' : lon,
                            'rssi': rssi,
                            'snr' : snr,
                        })

                    except (ValueError, KeyError) as e:
                        err_count += 1
                        err_msgs.append(f"행 {i}: {e}")
                        if len(err_msgs) > 10:
                            break

        except Exception as e:
            self.status.showMessage(f"CSV 읽기 실패: {e}")
            return

        if not field_data:
            self.status.showMessage("유효한 실측 데이터가 없습니다.")
            return

        self._field_data = field_data
        self._refresh_map()

        msg = f"실측 데이터 로드 완료: {len(field_data)}개"
        if err_count:
            msg += f", {err_count}개 오류 건너뜀"
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "CSV 일부 오류",
                f"{msg}\n\n오류 내용:\n" + "\n".join(err_msgs[:10]))

        self.lbl.setText(f"📡 실측: {len(field_data)}개")
        self.status.showMessage(msg)

    def _on_load_snapshot(self, snap):
        """히스토리 스냅샷을 메인 창에 복원."""
        self._result = snap.result
        self._ensure_gw_win()
        self._gw_win._gws = list(snap.gws)
        self._gw_win._refresh_table(suppress_map=True)
        self._ensure_node_win()
        self._node_win._nodes = list(snap.nodes)
        self._node_win._refresh_table(suppress_map=True)
        tile = self._settings.get('map_tile', 'CartoDB Voyager')
        self.map_w.refresh(
            gws=snap.gws, nodes=snap.nodes,
            result=snap.result,
            heatmaps=[], selected_gws=[],
            map_tile=tile
            )
        self.result_panel.update_result(snap.result, snap.gws)
        if self._node_win:
            self._node_win.update_result(snap.result)
        pct = snap.result.coverage_pct
        self.lbl.setText(
            f"커버리지: {snap.result.n_covered}/{snap.result.n_total} "
            f"({pct:.1f}%)")
        self.status.showMessage(
            f"'{snap.label}' 복원 완료 — {pct:.1f}%")

    # ── 커버리지 분석 ────────────────────────────────────────
    def _run_coverage(self, gws):
        if self.spatial is None:
            self.status.showMessage("공간 데이터 로드 중..."); return
        nodes = self._node_win.get_nodes() if self._node_win else []
        if not nodes:
            self.status.showMessage("단말기를 먼저 추가하세요."); return
            
        # ── 이미 분석 중이면 무시 (크래시 방지) ──────────────
        if self._cov_thread and self._cov_thread.isRunning():
            self.status.showMessage(
                "이전 커버리지 분석이 진행 중입니다. 잠시 기다려주세요.")
            return

        # ── 이전 스레드 안전하게 종료 ────────────────────────────
        try:
            if self._cov_thread and self._cov_thread.isRunning():
                self._cov_thread.quit()
                self._cov_thread.wait(2000)
        except RuntimeError:
            pass  # 이미 삭제된 경우 무시
        finally:
            self._cov_thread = None
            self._cov_worker = None

        self.status.showMessage(
            f"커버리지 분석 중: GW {len(gws)}개 × Node {len(nodes)}개...")
        self.map_w.show_loading(
            f"커버리지 분석 중: GW {len(gws)}개 × Node {len(nodes)}개...")

        w = CoverageWorker(self.spatial, gws, nodes, settings=self._settings)
        t = QThread()
        w.moveToThread(t)
        t.started.connect(w.run)
        w.sig_done.connect(self._on_coverage_done)
        w.sig_done.connect(t.quit)
        w.sig_err.connect(t.quit)
        w.sig_err.connect(lambda m: print(f"[COV ERR] {m}"))
        w.sig_err.connect(lambda m: self.map_w.hide_loading())   # ← 에러 시 해제
        t.finished.connect(w.deleteLater)
        t.finished.connect(t.deleteLater)
        t.finished.connect(self._on_cov_thread_finished)
        self._cov_worker = w
        self._cov_thread = t
        t.start()

    def _on_cov_thread_finished(self):
        """커버리지 스레드 종료 시 참조 정리."""
        self._cov_thread = None
        self._cov_worker = None

    def _on_coverage_done(self, result):
        self.map_w.hide_loading()
        self._result = result
        gws   = self._gw_win.get_gws()    if self._gw_win   else []
        nodes = self._node_win.get_nodes() if self._node_win else []
        sel   = [h['callsign'] for h in self._heatmaps] if self._heatmaps else []
        tile  = self._settings.get('map_tile', 'CartoDB Voyager')  # ← 추가
        self.map_w.refresh(
            gws=gws, nodes=nodes,
            result=result,
            heatmaps=self._heatmaps,
            selected_gws=sel,
            map_tile=tile)
        self.result_panel.update_result(result, gws)
        if self._node_win:
            self._node_win.update_result(result)
        pct = result.coverage_pct
        self.lbl.setText(
            f"커버리지: {result.n_covered}/{result.n_total} ({pct:.1f}%)")
        self.status.showMessage(f"커버리지 분석 완료: {pct:.1f}%")
        
        self._add_snapshot(result, gws, nodes) 

    # ── 지도 갱신 ────────────────────────────────────────────

    def _refresh_map(self):
        gws   = self._gw_win.get_gws()    if self._gw_win   else []
        nodes = self._node_win.get_nodes() if self._node_win else []
        sel   = [h['callsign'] for h in self._heatmaps] if self._heatmaps else []
        tile  = self._settings.get('map_tile', 'CartoDB Voyager')
        pts   = self._measure_pts if self._measuring else []
        self.map_w.refresh(
            gws=gws, nodes=nodes,
            result=self._result,
            heatmaps=self._heatmaps,
            selected_gws=sel,
            map_tile=tile,
            measure_pts=pts,
            field_data=self._field_data or None,
            settings=self._settings)


    def _clear_heatmap(self):
        self._heatmaps = []
        # _result는 유지 — Node 색(커버리지 결과)은 남김
        gws   = self._gw_win.get_gws()    if self._gw_win   else []
        nodes = self._node_win.get_nodes() if self._node_win else []
        if callable(nodes): nodes = []
        if callable(gws):   gws   = []
        self.map_w.refresh(
            gws=gws, nodes=nodes,
            result=self._result,   # 결과 유지
            heatmaps=[], selected_gws=[],
            map_tile=self._settings.get('map_tile', 'CartoDB Voyager'))
        self.lbl.setText("─")
        self.status.showMessage("히트맵 초기화 (커버리지 결과 유지)")

    # ── 드래그 이벤트 ────────────────────────────────────────

    def _on_gw_dragged(self, callsign, lon, lat):
        if self._gw_win is None:
            return
        gws = self._gw_win.get_gws()
        for i, gw in enumerate(gws):
            if gw.callsign == callsign:
                from core.coverage import GWEntry
                gws[i] = GWEntry(gw.callsign, lon, lat,
                                  gw.pt_dbm, gw.gt_dbi,
                                  gw.lt_db, gw.hb_m, gw.enabled)
                self._gw_win._gws = gws
                self._gw_win._refresh_table(suppress_map=True)
                self.status.showMessage(
                    f"{callsign} 이동 → ({lat:.6f}, {lon:.6f})")
                self._run_coverage(gws)
                break

    def _on_nd_dragged(self, callsign, lon, lat):
        if self._node_win is None:
            return
        nodes = self._node_win.get_nodes()
        for i, nd in enumerate(nodes):
            if nd.callsign == callsign:
                from core.coverage import NodeEntry
                nodes[i] = NodeEntry(nd.callsign, lon, lat,
                                      nd.gr_dbi, nd.lr_db,
                                      nd.hm_m, nd.min_rx_dbm)
                self._node_win._nodes = nodes
                self._node_win._refresh_table(suppress_map=True)
                self.status.showMessage(
                    f"{callsign} 이동 → ({lat:.6f}, {lon:.6f})")
                break

    # ── 우클릭 컨텍스트 메뉴 ────────────────────────────────

    def _on_map_right_clicked(self, lon: float, lat: float):
            from PyQt5.QtWidgets import QMenu, QAction
            from PyQt5.QtGui import QCursor
            from PyQt5.QtWidgets import QApplication
            from core.coverage import GWEntry, NodeEntry

            menu = QMenu(self)
            menu.setStyleSheet("""
                QMenu {
                    background:#1e2130; color:#e0e4ef;
                    border:1px solid #2a2f3b; border-radius:6px;
                    padding:4px;
                }
                QMenu::item {
                    padding:7px 22px; border-radius:4px;
                    font-size:12px;
                }
                QMenu::item:selected { background:#253a5a; color:#7ab8e8; }
                QMenu::item:disabled { color:#3a4060; }
                QMenu::separator { height:1px; background:#2a2f3b; margin:4px 8px; }
            """)

            # ── 좌표 표시 (비활성) ────────────────────────────
            lbl = menu.addAction(f"📍  {lat:.5f}, {lon:.5f}")
            lbl.setEnabled(False)
            menu.addSeparator()

            # ── GW / Node 추가 ────────────────────────────────
            act_gw   = menu.addAction("📡  이 위치에 GW 추가")
            act_node = menu.addAction("📶  이 위치에 단말기 추가")
            menu.addSeparator()

            # ── 커버리지 분석 ─────────────────────────────────
            gws   = self._gw_win.get_gws()    if self._gw_win   else []
            nodes = self._node_win.get_nodes() if self._node_win else []
            act_cov = menu.addAction("🔄  커버리지 분석 실행")
            act_cov.setEnabled(bool(gws) and bool(nodes))

            # ── 히트맵 계산 ───────────────────────────────────
            act_hm = menu.addAction("🗺  히트맵 계산")
            act_hm.setEnabled(bool(gws))
            menu.addSeparator()

            # ── 거리 측정 ─────────────────────────────────────
            if self._measuring:
                act_measure = menu.addAction("📏  거리 측정 종료")
            else:
                act_measure = menu.addAction("📏  거리 측정 시작")

            act_measure_clear = menu.addAction("✕  측정 초기화")
            act_measure_clear.setEnabled(bool(self._measure_pts))
            menu.addSeparator()

            # ── 좌표 복사 ─────────────────────────────────────
            act_copy_latlon = menu.addAction(
                f"📋  좌표 복사 ({lat:.6f}, {lon:.6f})")
            act_copy_lonlat = menu.addAction(
                f"📋  GeoJSON 좌표 복사 ({lon:.6f}, {lat:.6f})")

            # ── 메뉴 실행 ─────────────────────────────────────
            action = menu.exec_(QCursor.pos())
            if action is None:
                return

            # ── GW 추가 ───────────────────────────────────────
            if action == act_gw:
                self._ensure_gw_win()
                s = self._settings
                n = len(self._gw_win._gws) + 1
                gw = GWEntry(
                    callsign = f"GW{n}",
                    lon=lon, lat=lat,
                    pt_dbm   = s.get('gw_pt_dbm', 14.0),
                    gt_dbi   = s.get('gw_gt_dbi', 2.15),
                    lt_db    = s.get('gw_lt_db',  0.0),
                    hb_m     = s.get('gw_hb_m',   15.0),
                )
                self._gw_win._gws.append(gw)
                self._gw_win._refresh_table(suppress_map=True)
                self._refresh_map()
                self.status.showMessage(
                    f"GW{n} 추가 → ({lat:.5f}, {lon:.5f})")

            # ── Node 추가 ─────────────────────────────────────
            elif action == act_node:
                self._ensure_node_win()
                s = self._settings
                n = len(self._node_win._nodes) + 1
                nd = NodeEntry(
                    callsign       = f"Node{n}",
                    lon=lon, lat=lat,
                    gr_dbi         = s.get('nd_gr_dbi', 2.15),
                    lr_db          = s.get('nd_lr_db',  0.0),
                    hm_m           = s.get('nd_hm_m',   1.5),
                    min_rx_dbm     = s.get('nd_min_rx', -126.6),
                    indoor_loss_db = s.get('nd_indoor_loss', 0.0),
                )
                self._node_win._nodes.append(nd)
                self._node_win._refresh_table(suppress_map=True)
                self._refresh_map()
                self.status.showMessage(
                    f"Node{n} 추가 → ({lat:.5f}, {lon:.5f})")

            # ── 커버리지 분석 ─────────────────────────────────
            elif action == act_cov:
                active_gws = [g for g in gws if g.enabled]
                if active_gws:
                    self._run_coverage(active_gws)
                else:
                    self.status.showMessage("활성 GW가 없습니다.")

            # ── 히트맵 계산 ───────────────────────────────────
            elif action == act_hm:
                active_gws = [g for g in gws if g.enabled]
                if active_gws:
                    self._run_heatmap(active_gws, {})
                else:
                    self.status.showMessage("활성 GW가 없습니다.")

            # ── 거리 측정 시작/종료 ───────────────────────────
            elif action == act_measure:
                if self._measuring:
                    # 종료
                    self._measuring = False
                    self._refresh_map()
                    self.status.showMessage("거리 측정 모드 종료")
                else:
                    # 시작 — 클릭한 위치를 첫 번째 점으로 추가
                    self._measuring = True
                    self._measure_pts = [(lon, lat)]
                    self._refresh_map()
                    self.status.showMessage(
                        f"거리 측정 시작: P1 ({lat:.5f}, {lon:.5f}) — "
                        f"다음 점을 클릭하세요.")

            # ── 측정 초기화 ───────────────────────────────────
            elif action == act_measure_clear:
                self._measure_pts = []
                self._refresh_map()
                self.status.showMessage("거리 측정 초기화")
                self.lbl.setText("─")

            # ── 좌표 복사 ─────────────────────────────────────
            elif action == act_copy_latlon:
                QApplication.clipboard().setText(f"{lat:.6f}, {lon:.6f}")
                self.status.showMessage(
                    f"클립보드 복사: {lat:.6f}, {lon:.6f}")

            elif action == act_copy_lonlat:
                QApplication.clipboard().setText(f"{lon:.6f}, {lat:.6f}")
                self.status.showMessage(
                    f"클립보드 복사: {lon:.6f}, {lat:.6f}")

    def _ensure_gw_win(self):
        """GWListWindow가 없으면 생성."""
        if self._gw_win is None:
            self._gw_win = GWListWindow(self)
            self._gw_win.sig_coverage_requested.connect(self._run_heatmap)
            self._gw_win.sig_coverage_clear.connect(self._clear_heatmap)
            self._gw_win.sig_coverage_analyze.connect(self._run_coverage)
            self._gw_win.sig_map_refresh.connect(self._refresh_map)
            self._gw_win.sig_env_map_requested.connect(self._run_env_map)

    def _ensure_node_win(self):
        """NodeListWindow가 없으면 생성."""
        if self._node_win is None:
            self._node_win = NodeListWindow(self)
            self._node_win.sig_map_refresh.connect(self._refresh_map)

    # ── 히트맵 ──────────────────────────────────────────────
    def _start_worker(self, worker):
        # 실행 중 재진입은 _run_heatmap/_run_coverage에서 이미 차단됨
        t = QThread()
        
        self._thread = t
        self._worker = worker
        worker.moveToThread(t)
        t.started.connect(worker.run)
        worker.sig_log.connect(lambda msg: self.status.showMessage(msg))
        worker.sig_err.connect(self._on_error)
        worker.sig_err.connect(lambda m: print(f"[ERROR] {m}"))
        worker.sig_done.connect(t.quit)
        worker.sig_err.connect(t.quit)
        # 스레드 종료 시 객체 정리 (크래시 방지)
        t.finished.connect(worker.deleteLater)
        t.finished.connect(t.deleteLater)
        t.finished.connect(self._on_thread_finished)
        t.start()

    def _on_thread_finished(self):
        """스레드 종료 시 참조 정리."""
        self._thread = None
        self._worker = None

    def _run_heatmap(self, gws, settings):
            if self.spatial is None:
                self.status.showMessage("공간 데이터 로드 중..."); return
                
            # ── 이미 계산 중이면 무시 (크래시 방지) ──────────────
            if self._thread and self._thread.isRunning():
                self.status.showMessage(
                    "이전 히트맵 계산이 진행 중입니다. 잠시 기다려주세요.")
                return

            merged = dict(self._settings)
            merged.update(settings)
            if self._legend_levels:
                merged['color_levels'] = self._legend_levels
            if 'color_levels' in self._settings:
                merged['color_levels'] = self._settings['color_levels']

            # ── gw_color_map 생성 (버그 수정: set → list) ────────
            GW_HEX_COLORS = [
                '#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22',
                '#c0392b', '#2980b9', '#27ae60', '#8e44ad', '#17a589',
                '#e91e8c', '#5dade2', '#58d68d', '#f0e68c', '#2c3e50',
            ]
            active_gws = [g for g in gws if g.enabled]
            merged['gw_color_map'] = {
                g.callsign: GW_HEX_COLORS[i % len(GW_HEX_COLORS)]
                for i, g in enumerate(active_gws)
            }

            # ── 커버리지 결과와 Node 전달 (보간 히트맵용) ────────────
            nodes = self._node_win.get_nodes() if self._node_win else []
            # merged['_nodes']  = nodes
            # merged['_result'] = self._result   # None이면 직접 계산 방식 사용

            env = merged.get('env', 2)
            if env is None:
                env = 2

            fc  = merged.get('fc_mhz', 915.0)

            self.status.showMessage(
                f"GW 커버리지 계산 중: {', '.join(g.callsign for g in gws)}")
            self.map_w.show_loading(
                f"GW 커버리지 계산 중: {', '.join(g.callsign for g in gws)}")
            w = HeatmapWorker(self.spatial, gws, merged, env=env, fc=fc)
            w.sig_log.connect(self.map_w.update_loading_text)   # ← 진행률 연동
            w.sig_done.connect(self._on_heatmap_done)
            w.sig_err.connect(lambda m: self.map_w.hide_loading())  # ← 에러 시도 해제
            self._start_worker(w)

    def _on_heatmap_done(self, hms):
        self.map_w.hide_loading()
        self._heatmaps = hms
        # 합성 히트맵('COMBINED')인 경우, 실제 GW callsign 목록을 풀어서 사용
        # → Node 필터링에서 best_gw와 정확히 매칭되도록
        sel = []
        for h in hms:
            if h.get('type') == 'combined':
                sel.extend(h.get('gws', []))
            else:
                sel.append(h['callsign'])
        gws   = self._gw_win.get_gws()    if self._gw_win   else []
        nodes = self._node_win.get_nodes() if self._node_win else []
        tile = self._settings.get('map_tile', 'CartoDB Voyager')
        self.map_w.refresh(
            gws=gws, nodes=nodes,
            result=self._result,
            heatmaps=hms,
            selected_gws=sel,
            map_tile=tile)
        self.lbl.setText(f"히트맵: {', '.join(sel)}")
        self.status.showMessage(f"히트맵 완료: {', '.join(sel)}")

        # ── 히트맵에 사용된 GW로만 커버리지 분석 ────────────────
        # if nodes:
        #     if 'COMBINED' in sel:
        #         # hms에 저장된 gws 목록 사용
        #         hm_gw_callsigns = set()
        #         for hm in hms:
        #             if hm.get('type') == 'combined':
        #                 hm_gw_callsigns.update(hm.get('gws', []))
        #         cov_gws = [g for g in gws
        #                 if g.callsign in hm_gw_callsigns] \
        #                 if hm_gw_callsigns else \
        #                 [g for g in gws if g.enabled]
        #     else:
        #         hm_callsigns = set(sel)
        #         cov_gws = [g for g in gws if g.callsign in hm_callsigns]

        #     if cov_gws:
        #         self.status.showMessage(
        #             f"히트맵 완료 — 커버리지 분석 자동 시작 "
        #             f"({', '.join(g.callsign for g in cov_gws)})...")
        #         self._run_coverage(cov_gws)

    # ── 최적 배치 결과 ───────────────────────────────────────

    def _on_optimize_done(self, result, nodes):
        from core.coverage import GWEntry, CoverageResult, LinkResult

        opt_gws = []
        for i in range(result.num_gw):
            opt_gws.append(GWEntry(
                callsign=f"OPT-GW{i+1}",
                lon=float(result.gw_lon[i]),
                lat=float(result.gw_lat[i]),
                pt_dbm=14.0, gt_dbi=2.15,
                lt_db=0.0, hb_m=15.0, enabled=True))

        if self._gw_win is None:
            self._gw_win = GWListWindow(self)
            self._gw_win.sig_coverage_requested.connect(self._run_heatmap)
            self._gw_win.sig_coverage_clear.connect(self._clear_heatmap)
            self._gw_win.sig_coverage_analyze.connect(self._run_coverage)
            self._gw_win.sig_map_refresh.connect(self._refresh_map)

        existing = [g for g in self._gw_win._gws
                    if not g.callsign.startswith("OPT-")]
        self._gw_win._gws = existing + opt_gws
        self._gw_win._refresh_table(suppress_map=True)

        cov_result = CoverageResult(n_total=len(nodes))
        for ni in range(len(nodes)):
            gw_no   = int(result.node_gw[ni])
            cov     = gw_no > 0
            best_gw = f"OPT-GW{gw_no}" if cov else ""
            cov_result.nodes.append(LinkResult(
                covered=cov, best_gw=best_gw,
                best_pr=0.0, gw_prs={}))
            if cov:
                cov_result.n_covered += 1
                cov_result.gw_counts[best_gw] = \
                    cov_result.gw_counts.get(best_gw, 0) + 1

        self._result = cov_result
        all_gws = self._gw_win._gws
        self.map_w.refresh(
            gws=all_gws, nodes=nodes,
            result=cov_result,
            heatmaps=[], selected_gws=[],
            map_tile=self._settings.get('map_tile', 'CartoDB Voyager'))

        pct = cov_result.coverage_pct
        self.lbl.setText(
            f"최적 배치: GW {result.num_gw}개 | 커버리지 {pct:.1f}%")
        self.status.showMessage(
            f"GW 최적 배치 완료 — 커버리지 분석 자동 시작...")
        self._run_coverage(opt_gws)

    def _toggle_measure(self, checked):
        self._measuring   = checked
        self._measure_pts = []
        if checked:
            self.status.showMessage(
                "거리 측정 모드: 지도에서 클릭하세요. "
                "여러 점 연속 측정 가능 | 종료: 버튼 다시 클릭")
        else:
            self._refresh_map()
            self.status.showMessage("거리 측정 모드 종료")

    def _on_map_clicked(self, lon, lat):
        if self._measuring:
            self._measure_pts.append((lon, lat))
            n = len(self._measure_pts)

            if n == 1:
                self.status.showMessage(
                    f"P1: ({lat:.5f}, {lon:.5f}) — 다음 점을 클릭하세요.")
            else:
                from core.utils import haversine, bearing
                p1   = self._measure_pts[-2]
                p2   = self._measure_pts[-1]
                dist = haversine(p1[0], p1[1], p2[0], p2[1])
                brg  = bearing(p1[0], p1[1], p2[0], p2[1])
                total = sum(
                    haversine(self._measure_pts[i][0], self._measure_pts[i][1],
                              self._measure_pts[i+1][0], self._measure_pts[i+1][1])
                    for i in range(len(self._measure_pts)-1))
                self.status.showMessage(
                    f"P{n}: ({lat:.5f}, {lon:.5f}) | "
                    f"구간: {dist:.3f}km / {brg:.1f}° | "
                    f"누적: {total:.3f}km")
                self.lbl.setText(
                    f"📏 구간 {dist:.3f}km | 누적 {total:.3f}km | {brg:.1f}°")

            gws   = self._gw_win.get_gws()    if self._gw_win   else []
            nodes = self._node_win.get_nodes() if self._node_win else []
            tile  = self._settings.get('map_tile', 'CartoDB Voyager')
            sel   = [h['callsign'] for h in self._heatmaps] if self._heatmaps else []
            self.map_w.refresh(
                gws=gws, nodes=nodes,
                result=self._result,
                heatmaps=self._heatmaps,
                selected_gws=sel,
                map_tile=tile,
                measure_pts=self._measure_pts)
            return

        self.status.showMessage(f"지도 클릭: ({lat:.5f}, {lon:.5f})")

    # ── 공통 ────────────────────────────────────────────────

    def _save_result(self):
        import json
        from PyQt5.QtWidgets import QFileDialog

        class _Enc(json.JSONEncoder):
            def default(self, o):
                import numpy as np
                if isinstance(o, (np.integer,)):  return int(o)
                if isinstance(o, (np.floating,)):  return float(o)
                if isinstance(o, (np.bool_,)):     return bool(o)
                if isinstance(o, np.ndarray):      return o.tolist()
                return super().default(o)

        if self._result is None:
            self.status.showMessage("저장할 분석 결과가 없습니다.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "결과 저장", "coverage_result.json", "JSON (*.json)")
        if not path: return
        gws   = self._gw_win.get_gws()    if self._gw_win   else []
        nodes = self._node_win.get_nodes() if self._node_win else []
        data = {
            'gws'  : [{'callsign': g.callsign, 'lon': float(g.lon),
                        'lat': float(g.lat), 'pt_dbm': float(g.pt_dbm),
                        'gt_dbi': float(g.gt_dbi), 'lt_db': float(g.lt_db),
                        'hb_m': float(g.hb_m), 'enabled': bool(g.enabled)}
                       for g in gws],
            'nodes': [{'callsign': n.callsign, 'lon': float(n.lon),
                        'lat': float(n.lat), 'gr_dbi': float(n.gr_dbi),
                        'lr_db': float(n.lr_db), 'hm_m': float(n.hm_m),
                        'min_rx_dbm': float(n.min_rx_dbm)}
                       for n in nodes],
            'result': {
                'n_covered' : int(self._result.n_covered),
                'n_total'   : int(self._result.n_total),
                'gw_counts' : {str(k): int(v)
                               for k, v in self._result.gw_counts.items()},
                'nodes'     : [{'covered' : bool(nd.covered),
                                 'best_gw' : str(nd.best_gw),
                                 'best_pr' : float(nd.best_pr),
                                 'gw_prs'  : {str(k): float(v)
                                              for k, v in nd.gw_prs.items()},
                                 'macro_pr': float(getattr(nd,'macro_pr',-999.0)),
                                 'n_rx_gw' : int(getattr(nd,'n_rx_gw',0))}
                                for nd in self._result.nodes],
            },
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, cls=_Enc, ensure_ascii=False, indent=2)
        self.status.showMessage(f"결과 저장 완료: {path}")

    def _load_result(self):
        import json
        from PyQt5.QtWidgets import QFileDialog
        from core.coverage import GWEntry, NodeEntry, CoverageResult, LinkResult
        path, _ = QFileDialog.getOpenFileName(
            self, "결과 불러오기", "", "JSON (*.json)")
        if not path: return
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)

            if self._gw_win is None:
                self._gw_win = GWListWindow(self)
                self._gw_win.sig_coverage_requested.connect(self._run_heatmap)
                self._gw_win.sig_coverage_clear.connect(self._clear_heatmap)
                self._gw_win.sig_coverage_analyze.connect(self._run_coverage)
                self._gw_win.sig_map_refresh.connect(self._refresh_map)
            self._gw_win._gws = [GWEntry(**g) for g in data.get('gws', [])]
            self._gw_win._refresh_table(suppress_map=True)

            if self._node_win is None:
                self._node_win = NodeListWindow(self)
                self._node_win.sig_map_refresh.connect(self._refresh_map)
            self._node_win._nodes = [NodeEntry(**n) for n in data.get('nodes', [])]
            self._node_win._refresh_table(suppress_map=True)

            r_data = data.get('result', {})
            result = CoverageResult(
                n_covered = r_data.get('n_covered', 0),
                n_total   = r_data.get('n_total', 0),
                gw_counts = r_data.get('gw_counts', {}),
            )
            for nd in r_data.get('nodes', []):
                result.nodes.append(LinkResult(
                    covered = nd.get('covered', False),
                    best_gw = nd.get('best_gw', ''),
                    best_pr = nd.get('best_pr', -999.0),
                    gw_prs  = nd.get('gw_prs', {}),
                ))
            self._result = result

            gws   = self._gw_win.get_gws()
            nodes = self._node_win.get_nodes()
            tile  = self._settings.get('map_tile', 'CartoDB Voyager')
            self.map_w.refresh(gws=gws, nodes=nodes,
                               result=result, heatmaps=[],
                               selected_gws=[], map_tile=tile)
            self.result_panel.update_result(result, gws)
            self.lbl.setText(
                f"불러오기 완료: {result.n_covered}/{result.n_total} "
                f"({result.coverage_pct:.1f}%)")
            self.status.showMessage(f"결과 불러오기 완료: {path}")
        except Exception:
            import traceback
            print(traceback.format_exc(), flush=True)
            self.status.showMessage("결과 불러오기 실패 — 콘솔 확인")

    def _run_env_map(self):
        if self.spatial is None: return
        self.status.showMessage("환경 분류 지도 계산 중...")

        class EnvMapWorker(QObject):
            sig_done = pyqtSignal(dict)
            sig_log  = pyqtSignal(str)
            sig_err  = pyqtSignal(str)
            def __init__(self, spatial):
                super().__init__()
                self.spatial = spatial
            def run(self):
                try:
                    from core.coverage import CoverageEngine
                    eng = CoverageEngine(self.spatial)
                    hm  = eng.env_map(step=0.003, cb=self.sig_log.emit)
                    self.sig_done.emit(hm)
                except Exception:
                    import traceback
                    self.sig_err.emit(traceback.format_exc())

        w = EnvMapWorker(self.spatial)
        w.sig_done.connect(self._on_env_map_done)
        self._start_worker(w)

    def _on_env_map_done(self, hm):
        self._heatmaps = [hm]
        gws   = self._gw_win.get_gws()    if self._gw_win   else []
        nodes = self._node_win.get_nodes() if self._node_win else []
        tile  = self._settings.get('map_tile', 'CartoDB Voyager')
        self.map_w.refresh(gws=gws, nodes=nodes,
                           result=self._result,
                           heatmaps=[hm],
                           selected_gws=[], map_tile=tile)
        self.status.showMessage(
            "환경 분류 지도 완료 | "
            "🔴 Dense Urban  🟠 Urban  🟡 Suburban  🟢 Open")
        self.lbl.setText(
            "ENV: 🔴Dense Urban 🟠Urban 🟡Suburban 🟢Open")

    def _on_error(self, msg):
        print(f"[오류] {msg}")
        self.status.showMessage("오류 발생 — 콘솔 확인")

    def _load_spatial(self):
        # main.py의 _launch()에서 이미 로드해서 주입하므로
        # 여기서는 spatial이 없을 때만 로드 (fallback)
        if self.spatial is not None:
            return
        from core.dem_loader import SpatialData
        self.status.showMessage("공간 데이터 로드 중...")
        try:
            self.spatial = SpatialData(self._shp, self._dem)
            self.spatial.load()
            self.status.showMessage("준비")
            self.map_w.set_bounds(tuple(self.spatial.bounds))
        except Exception as e:
            self.status.showMessage(f"공간 데이터 로드 실패: {e}")
        self.map_w.refresh()