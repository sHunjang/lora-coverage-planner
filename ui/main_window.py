# ui/main_window.py
from __future__ import annotations
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QStatusBar, QLabel,
    QToolBar, QAction, QApplication, QSizePolicy,
)
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QThread, QSize
from PyQt5.QtGui import QCursor
from ui.map_widget       import MapWidget
from ui.gw_list_window   import GWListWindow
from ui.node_list_window import NodeListWindow
from ui.result_panel     import ResultPanel
from core.coverage       import CoverageEngine, GWEntry

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


class CoverageWorker(QObject):
    sig_done = pyqtSignal(object)   # ← 누락
    sig_err  = pyqtSignal(str)      # ← 누락

    def __init__(self, spatial, gws, nodes, settings=None):
        super().__init__()
        self.spatial  = spatial
        self.gws      = gws
        self.nodes    = nodes
        self.settings = settings or {}

    def run(self):
        try:
            from core.coverage import CoverageEngine
            env  = self.settings.get('env', 2) or 2
            fc   = self.settings.get('fc_mhz', 915.0)
            nsmp = self.settings.get('n_samples', 100)
            eng    = CoverageEngine(self.spatial, env=env,
                                    fc=fc, n_samples=nsmp)
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
        self.env, self.fc = env, fc

    def run(self):
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.ndimage import label as nd_label
            from core.coverage import CoverageEngine

            eng    = CoverageEngine(self.spatial, self.env, self.fc)
            min_rx = self.settings.get('min_rx', -126.6)
            step = float(self.settings.get('heatmap_step', 0.0015))

            color_levels = self.settings.get('color_levels', [
                {'pr': -90,  'color': '#FF2020'},
                {'pr': -100, 'color': '#FF8C00'},
                {'pr': -110, 'color': '#FFD700'},
                {'pr': -120, 'color': '#00C94A'},
                {'pr': -130, 'color': '#4f8ef7'},
            ])
            SF_SENS = {
                7: -123.0, 8: -126.0, 9: -129.0,
                10: -132.0, 11: -134.5, 12: -137.0
            }
            SF_COLORS = {
                7: '#FF2020', 8: '#FF8C00', 9: '#FFD700',
                10: '#00C94A', 11: '#4f8ef7', 12: '#9B59B6',
            }

            hms = []

            if len(self.gws) > 1:
                # ── 여러 GW → 합성 히트맵 (최대 Pr 기준) ────────────
                self.sig_log.emit(
                    f"{len(self.gws)}개 GW 합성 히트맵 계산 중...")
                pr_min = min((lv['pr'] for lv in color_levels), default=None)
                hm = eng.heatmap_combined(
                    self.gws, min_rx, step=step,
                    cb=self.sig_log.emit,
                    pr_min=pr_min)

                # 등고선 계산
                ps  = hm.get('ps')
                cm  = hm.get('cm')
                contours = []

                if ps is not None and cm is not None:
                    lmin    = hm.get('lon_min', 0)
                    latmin  = hm.get('lat_min', 0)
                    lon_ax  = np.linspace(lmin,
                                        lmin + step * ps.shape[1],
                                        ps.shape[1])
                    lat_ax  = np.linspace(latmin,
                                        latmin + step * ps.shape[0],
                                        ps.shape[0])
                    pr_m     = np.where(cm, ps, np.nan)
                    ps_in_cm = ps[cm]

                    if len(ps_in_cm) > 0:
                        pr_min_in = float(ps_in_cm.min())
                        pr_max_in = float(ps_in_cm.max())

                        for lv in color_levels:
                            pv = float(lv['pr'])
                            if pv < pr_min_in or pv > pr_max_in:
                                continue
                            try:
                                fig, ax = plt.subplots()
                                cs = ax.contour(lon_ax, lat_ax, pr_m, levels=[pv])
                                plt.close(fig)
                            except Exception:
                                plt.close('all')
                                continue
                            segs, lpts = [], []
                            for col_segs in cs.allsegs:
                                for seg in col_segs:
                                    if len(seg) < 4:
                                        continue
                                    d = np.diff(seg, axis=0)
                                    if float(np.sqrt(
                                            (d**2).sum(axis=1)).sum()) < step:
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

                        # SF별 등고선
                        sf_layers = []
                        for sf, sens in SF_SENS.items():
                            if not (ps[cm] >= sens).any():
                                continue
                            pr_sf = np.where(cm, ps, np.nan)
                            try:
                                fig, ax = plt.subplots()
                                cs_sf = ax.contour(
                                    lon_ax, lat_ax, pr_sf, levels=[sens])
                                plt.close(fig)
                            except Exception:
                                plt.close('all')
                                continue
                            segs_sf = []
                            for col_segs in cs_sf.allsegs:
                                for seg in col_segs:
                                    if len(seg) < 4:
                                        continue
                                    d = np.diff(seg, axis=0)
                                    if float(np.sqrt(
                                            (d**2).sum(axis=1)).sum()) < step:
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

            else:
                # ── 단일 GW → 개별 히트맵 ────────────────────────────
                for gw in self.gws:
                    self.sig_log.emit(f"{gw.callsign} 히트맵 계산 중...")
                    use_deygout = self.settings.get('heatmap_diff', False)
                    pr_min = min((lv['pr'] for lv in color_levels), default=None)
                    hm = eng.heatmap(gw, min_rx, step=step,
                                    cb=self.sig_log.emit,
                                    pr_min=pr_min)

                    ps = hm.get('ps')
                    cm = hm.get('cm')
                    contours = []

                    if ps is not None and cm is not None:
                        lmin   = hm.get('lon_min', 0)
                        latmin = hm.get('lat_min', 0)
                        lon_ax = np.linspace(lmin,
                                            lmin + step * ps.shape[1],
                                            ps.shape[1])
                        lat_ax = np.linspace(latmin,
                                            latmin + step * ps.shape[0],
                                            ps.shape[0])
                        pr_m     = np.where(cm, ps, np.nan)
                        ps_in_cm = ps[cm]

                        if len(ps_in_cm) > 0:
                            pr_min_in = float(ps_in_cm.min())
                            pr_max_in = float(ps_in_cm.max())

                            for lv in color_levels:
                                pv = float(lv['pr'])
                                if pv < pr_min_in or pv > pr_max_in:
                                    continue
                                try:
                                    fig, ax = plt.subplots()
                                    cs = ax.contour(
                                        lon_ax, lat_ax, pr_m, levels=[pv])
                                    plt.close(fig)
                                except Exception:
                                    plt.close('all')
                                    continue
                                segs, lpts = [], []
                                for col_segs in cs.allsegs:
                                    for seg in col_segs:
                                        if len(seg) < 4:
                                            continue
                                        d = np.diff(seg, axis=0)
                                        if float(np.sqrt(
                                                (d**2).sum(axis=1)).sum()) < step:
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

                            sf_layers = []
                            for sf, sens in SF_SENS.items():
                                if not (ps[cm] >= sens).any():
                                    continue
                                pr_sf = np.where(cm, ps, np.nan)
                                try:
                                    fig, ax = plt.subplots()
                                    cs_sf = ax.contour(
                                        lon_ax, lat_ax, pr_sf, levels=[sens])
                                    plt.close(fig)
                                except Exception:
                                    plt.close('all')
                                    continue
                                segs_sf = []
                                for col_segs in cs_sf.allsegs:
                                    for seg in col_segs:
                                        if len(seg) < 4:
                                            continue
                                        d = np.diff(seg, axis=0)
                                        if float(np.sqrt(
                                                (d**2).sum(axis=1)).sum()) < step:
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


class MainWindow(QMainWindow):
    def __init__(self, shp_path, dem_path):
        super().__init__()
        self.setWindowTitle("LoRa Coverage Planner")
        geo = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(int(geo.width()*0.03), int(geo.height()*0.03),
                         int(geo.width()*0.94), int(geo.height()*0.94))
        self.setStyleSheet(f"QMainWindow{{background:{DARK};}}")

        self.spatial     = None
        self._thread     = None
        self._cov_thread = None
        self._cov_worker = None
        self._legend_levels = None   # None이면 기본값 사용
        self._legend_win    = None
        self._result     = None
        self._heatmaps   = []
        self._shp        = shp_path
        self._dem        = dem_path
        self._gw_win     = None
        self._node_win   = None
        self._opt_win    = None

        # 설정 로드
        from ui.settings_window import load_settings
        self._settings = load_settings()

        self._build_ui()
        self._load_spatial()

    def _build_ui(self):
        tb = QToolBar()
        tb.setMovable(False)
        tb.setIconSize(QSize(20, 20))
        tb.setStyleSheet(TOOLBAR_STYLE)
        self.addToolBar(Qt.TopToolBarArea, tb)

        act_gw   = QAction("📡  GW 목록",      self)
        act_node = QAction("📶  단말 목록",     self)
        act_opt  = QAction("⚙   GW 최적 배치", self)
        act_legend = QAction("🎨  범례 설정", self)
        act_cfg  = QAction("🔧  설정",          self)
        act_dist = QAction("📏  거리 측정", self, checkable=True)
        act_save = QAction("💾  결과 저장",     self)
        act_load = QAction("📂  결과 불러오기", self)

        for a in [act_gw, act_node, act_opt, act_legend, act_cfg,
                  act_dist, act_save, act_load]:
            tb.addAction(a)

        act_gw.triggered.connect(self._open_gw_list)
        act_node.triggered.connect(self._open_node_list)
        act_opt.triggered.connect(self._open_optimize)
        act_legend.triggered.connect(self._open_legend)
        act_dist.triggered.connect(self._toggle_measure)
        self._measuring = False
        self._measure_pts = []
        act_cfg.triggered.connect(self._open_settings)
        act_save.triggered.connect(self._save_result)
        act_load.triggered.connect(self._load_result)

        # 중앙: 지도(좌) + 결과 패널(우) 분할
        from PyQt5.QtWidgets import QSplitter
        splitter = QSplitter(Qt.Horizontal)
        self.map_w      = MapWidget()
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
        # self.map_w.sig_map_right_clicked.connect(self._on_map_right_clicked)

    # ── 창 열기 ─────────────────────────────────────────────

    def _open_gw_list(self):
        if self._gw_win is None:
            self._gw_win = GWListWindow(self)
            self._gw_win.sig_coverage_requested.connect(self._run_heatmap)
            self._gw_win.sig_coverage_clear.connect(self._clear_heatmap)
            self._gw_win.sig_coverage_analyze.connect(self._run_coverage)
            self._gw_win.sig_map_refresh.connect(self._refresh_map)
            self._gw_win.sig_env_map_requested.connect(self._run_env_map)
        self._gw_win.show(); self._gw_win.raise_()

    def _open_node_list(self):
        if self._node_win is None:
            self._node_win = NodeListWindow(self)
            self._node_win.sig_map_refresh.connect(self._refresh_map)
        self._node_win.show(); self._node_win.raise_()

    def _open_settings(self):
        from ui.settings_window import SettingsWindow
        dlg = SettingsWindow(self)
        dlg.sig_settings_changed.connect(self._on_settings_changed)
        dlg.exec_()

    def _on_settings_changed(self, settings: dict):
        self._settings = settings
        # 지도 타일 변경 시 지도 새로고침
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
        # 매번 현재 레벨로 새로 생성
        levels = self._legend_levels or DEFAULT_LEVELS
        self._legend_win = LegendWindow(levels=levels, parent=self)
        self._legend_win.sig_levels_changed.connect(self._on_legend_changed)
        self._legend_win.show()
        self._legend_win.raise_()

    def _on_legend_changed(self, levels: list):
        self._legend_levels = levels
        self._settings['color_levels'] = levels
        self.status.showMessage(
            f"범례 업데이트 완료 — {len(levels)}개 레벨 | "
            f"히트맵을 다시 계산하면 반영됩니다.")

    # ── 커버리지 분석 ────────────────────────────────────────

    def _run_coverage(self, gws):
        if self.spatial is None:
            self.status.showMessage("공간 데이터 로드 중..."); return
        nodes = self._node_win.get_nodes() if self._node_win else []
        if not nodes:
            self.status.showMessage("단말기를 먼저 추가하세요."); return

        # 이전 스레드 종료 후 재시작
        if self._cov_thread and self._cov_thread.isRunning():
            self._cov_thread.quit()
            self._cov_thread.wait(2000)

        self.status.showMessage(
            f"커버리지 분석 중: GW {len(gws)}개 × Node {len(nodes)}개...")

        w = CoverageWorker(self.spatial, gws, nodes, settings=self._settings)
        t = QThread()
        w.moveToThread(t)
        t.started.connect(w.run)
        w.sig_done.connect(self._on_coverage_done)
        w.sig_err.connect(lambda m: print(f"[COV ERR] {m}"))
        self._cov_worker = w
        self._cov_thread = t
        t.start()

    def _on_coverage_done(self, result):
        self._result = result
        gws   = self._gw_win.get_gws()    if self._gw_win   else []
        nodes = self._node_win.get_nodes() if self._node_win else []
        sel   = [h['callsign'] for h in self._heatmaps] if self._heatmaps else []
        self.map_w.refresh(
            gws=gws, nodes=nodes,
            result=result,
            heatmaps=self._heatmaps,
            selected_gws=sel)
        self.result_panel.update_result(result, gws)
        pct = result.coverage_pct
        self.lbl.setText(
            f"커버리지: {result.n_covered}/{result.n_total} ({pct:.1f}%)")
        self.status.showMessage(f"커버리지 분석 완료: {pct:.1f}%")

    # ── 지도 갱신 ───────────────────────────────────────────
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
            measure_pts=pts)

    def _clear_heatmap(self):
        self._heatmaps = []
        self._result   = None
        gws   = self._gw_win.get_gws()    if self._gw_win   else []
        nodes = self._node_win.get_nodes() if self._node_win else []
        if callable(nodes): nodes = []
        if callable(gws):   gws   = []
        self.map_w.refresh(
            gws=gws, nodes=nodes,
            result=None, heatmaps=[], selected_gws=[])
        self.lbl.setText("─")
        self.status.showMessage("커버리지 초기화")

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
                # 드래그 완료 시 자동 커버리지 재계산
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

    # ── 우클릭 컨텍스트 메뉴 ─────────────────────────────────

    # def _on_map_right_clicked(self, lon: float, lat: float):
    #     from PyQt5.QtWidgets import QMenu
    #     from core.coverage import GWEntry, NodeEntry

    #     menu = QMenu(self)
    #     menu.setStyleSheet("""
    #         QMenu {
    #             background:#1e2130; color:#e0e4ef;
    #             border:1px solid #2a2f3b; border-radius:6px;
    #             padding:4px;
    #         }
    #         QMenu::item { padding:6px 20px; border-radius:4px; }
    #         QMenu::item:selected { background:#253a5a; color:#7ab8e8; }
    #         QMenu::separator { height:1px; background:#2a2f3b; margin:4px 8px; }
    #     """)

    #     lbl = menu.addAction(f"📍 ({lat:.5f}, {lon:.5f})")
    #     lbl.setEnabled(False)
    #     menu.addSeparator()
    #     act_gw   = menu.addAction("📡  이 위치에 GW 추가")
    #     act_node = menu.addAction("📶  이 위치에 단말기 추가")

    #     action = menu.exec_(QCursor.pos())

    #     if action == act_gw:
    #         if self._gw_win is None:
    #             self._gw_win = GWListWindow(self)
    #             self._gw_win.sig_coverage_requested.connect(self._run_heatmap)
    #             self._gw_win.sig_coverage_clear.connect(self._clear_heatmap)
    #             self._gw_win.sig_coverage_analyze.connect(self._run_coverage)
    #             self._gw_win.sig_map_refresh.connect(self._refresh_map)
    #         n  = len(self._gw_win._gws) + 1
    #         gw = GWEntry(callsign=f"GW{n}", lon=lon, lat=lat)
    #         self._gw_win._gws.append(gw)
    #         self._gw_win._refresh_table(suppress_map=True)
    #         self._refresh_map()
    #         self.status.showMessage(f"GW{n} 추가 → ({lat:.5f}, {lon:.5f})")

    #     elif action == act_node:
    #         if self._node_win is None:
    #             self._node_win = NodeListWindow(self)
    #             self._node_win.sig_map_refresh.connect(self._refresh_map)
    #         n  = len(self._node_win._nodes) + 1
    #         nd = NodeEntry(callsign=f"Node{n}", lon=lon, lat=lat)
    #         self._node_win._nodes.append(nd)
    #         self._node_win._refresh_table(suppress_map=True)
    #         self._refresh_map()
    #         self.status.showMessage(f"Node{n} 추가 → ({lat:.5f}, {lon:.5f})")

    # ── 히트맵 ──────────────────────────────────────────────

    def _start_worker(self, worker):
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(2000)
        t = QThread()
        self._thread = t
        worker.moveToThread(t)
        t.started.connect(worker.run)
        worker.sig_log.connect(lambda msg: self.status.showMessage(msg))
        worker.sig_err.connect(self._on_error)
        worker.sig_err.connect(lambda m: print(f"[ERROR] {m}"))
        self._worker = worker
        t.start()

    def _run_heatmap(self, gws, settings):
        if self.spatial is None:
            self.status.showMessage("공간 데이터 로드 중..."); return

        # 현재 범례 레벨을 settings에 병합
        merged = dict(settings)
        if self._legend_levels:
            merged['color_levels'] = self._legend_levels

        self.status.showMessage(
            f"히트맵 계산 중: {', '.join(g.callsign for g in gws)}")
        w = HeatmapWorker(self.spatial, gws, merged)
        w.sig_done.connect(self._on_heatmap_done)
        self._start_worker(w)

    def _on_heatmap_done(self, hms):
        self._heatmaps = hms
        sel   = [h['callsign'] for h in hms]
        gws   = self._gw_win.get_gws()    if self._gw_win   else []
        nodes = self._node_win.get_nodes() if self._node_win else []
        self.map_w.refresh(
            gws=gws, nodes=nodes,
            result=self._result,
            heatmaps=hms,
            selected_gws=sel)
        self.lbl.setText(f"히트맵: {', '.join(sel)}")
        self.status.showMessage(f"히트맵 완료: {', '.join(sel)}")

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
            heatmaps=[], selected_gws=[])

        pct = cov_result.coverage_pct
        self.lbl.setText(
            f"최적 배치: GW {result.num_gw}개 | 커버리지 {pct:.1f}%")
        self.status.showMessage(
            f"GW 최적 배치 완료 — GW {result.num_gw}개 | {pct:.1f}% 커버")

        # 배치 완료 후 커버리지 자동 실행
        self.status.showMessage(
            f"GW 최적 배치 완료 — 커버리지 분석 자동 시작...")
        self._run_coverage(opt_gws)

    def _toggle_measure(self, checked):
        self._measuring = checked
        self._measure_pts = []
        if checked:
            self.status.showMessage(
                "거리 측정 모드: 지도에서 클릭하세요. "
                "여러 점 연속 측정 가능 | 종료: 버튼 다시 클릭")
        else:
            # 측정 종료 시 선 제거
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
                # 마지막 두 점 간 거리/방위각 계산
                from core.utils import haversine, bearing
                p1 = self._measure_pts[-2]
                p2 = self._measure_pts[-1]
                dist = haversine(p1[0], p1[1], p2[0], p2[1])
                brg  = bearing(p1[0], p1[1], p2[0], p2[1])

                # 전체 누적 거리
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

            # 지도에 측정선 즉시 반영
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
        if self._gw_win and self._gw_win.isVisible():
            self._gw_win.set_coord(lon, lat)
        if self._node_win and self._node_win.isVisible():
            self._node_win.set_coord(lon, lat)

    # ── 공통 ────────────────────────────────────────────────

    def _save_result(self):
        """현재 커버리지 분석 결과와 GW/Node를 JSON으로 저장."""
        import json
        from PyQt5.QtWidgets import QFileDialog

        # numpy/bool 타입을 Python 기본 타입으로 변환하는 인코더
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
        """저장된 JSON에서 GW/Node/결과 불러오기."""
        import json
        from PyQt5.QtWidgets import QFileDialog
        from core.coverage import GWEntry, NodeEntry, CoverageResult, LinkResult
        path, _ = QFileDialog.getOpenFileName(
            self, "결과 불러오기", "", "JSON (*.json)")
        if not path: return
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)

            # GW 복원
            if self._gw_win is None:
                self._gw_win = GWListWindow(self)
                self._gw_win.sig_coverage_requested.connect(self._run_heatmap)
                self._gw_win.sig_coverage_clear.connect(self._clear_heatmap)
                self._gw_win.sig_coverage_analyze.connect(self._run_coverage)
                self._gw_win.sig_map_refresh.connect(self._refresh_map)
            self._gw_win._gws = [GWEntry(**g) for g in data.get('gws', [])]
            self._gw_win._refresh_table(suppress_map=True)

            # Node 복원
            if self._node_win is None:
                self._node_win = NodeListWindow(self)
                self._node_win.sig_map_refresh.connect(self._refresh_map)
            self._node_win._nodes = [NodeEntry(**n) for n in data.get('nodes', [])]
            self._node_win._refresh_table(suppress_map=True)

            # 결과 복원
            r_data = data.get('result', {})
            result = CoverageResult(
                n_covered  = r_data.get('n_covered', 0),
                n_total    = r_data.get('n_total', 0),
                gw_counts  = r_data.get('gw_counts', {}),
            )
            for nd in r_data.get('nodes', []):
                result.nodes.append(LinkResult(
                    covered  = nd.get('covered', False),
                    best_gw  = nd.get('best_gw', ''),
                    best_pr  = nd.get('best_pr', -999.0),
                    gw_prs   = nd.get('gw_prs', {}),
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
        from core.dem_loader import SpatialData
        self.status.showMessage("공간 데이터 로드 중...")
        try:
            self.spatial = SpatialData(self._shp, self._dem)
            self.spatial.load()
            self.status.showMessage("준비")
        except Exception as e:
            self.status.showMessage(f"공간 데이터 로드 실패: {e}")
        self.map_w.refresh()