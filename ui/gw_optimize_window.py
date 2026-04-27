# ui/gw_optimize_window.py
from __future__ import annotations
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QDoubleSpinBox, QSpinBox, QCheckBox,
    QGroupBox, QFormLayout, QPlainTextEdit,
    QProgressBar, QMessageBox,
)
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QThread

from ui.dialogs import DARK, PANEL, TEXT, MUTED, BORDER, STYLE_DLG

STYLE = STYLE_DLG + f"""
QPlainTextEdit {{
    background:#0e1015; color:#5a8a6a;
    border:1px solid {BORDER}; border-radius:4px;
    font-family:Consolas,monospace; font-size:10px;
}}
QProgressBar {{
    background:{PANEL}; border:1px solid {BORDER};
    border-radius:4px; height:16px; text-align:center;
    color:{TEXT}; font-size:10px;
}}
QProgressBar::chunk {{ background:#1d6a1d; border-radius:3px; }}
"""

BTN = ("QPushButton{background:#1c2a3a;color:#7ab8e8;"
       "border:1px solid #2a4a6a;border-radius:5px;"
       "padding:6px 16px;font-size:12px;}"
       "QPushButton:hover{background:#254d78;}"
       "QPushButton:disabled{color:#3a5a6a;border-color:#1a2a3a;}")
BTN_RUN = ("QPushButton{background:#1d4a1d;color:#7ae87a;"
           "border:1px solid #2a6a2a;border-radius:5px;"
           "padding:8px 20px;font-size:13px;font-weight:bold;}"
           "QPushButton:hover{background:#256a25;}"
           "QPushButton:disabled{color:#3a5a3a;border-color:#1a3a1a;}")


class LinkMatrixWorker(QObject):
    sig_log   = pyqtSignal(str)
    sig_done  = pyqtSignal(object)
    sig_error = pyqtSignal(str)

    def __init__(self, spatial, nodes, params):
        super().__init__()
        self.spatial = spatial
        self.nodes   = nodes
        self.params  = params

    def run(self):
        try:
            from core.propagation import PathLossModel
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import numpy as np

            p  = self.params
            N  = len(self.nodes)

            st_xy = [self.spatial.lonlat_to_xy(n.lon, n.lat)
                     for n in self.nodes]
            st_x  = np.array([float(xy[0]) for xy in st_xy])
            st_y  = np.array([float(xy[1]) for xy in st_xy])

            model = PathLossModel(
                self.spatial,
                h_station = p["hm"],
                env       = p["env"],
                fc        = p["fc"],
                n_samples = 100,
            )

            matrix    = np.zeros((N, N), dtype=np.float32)
            pl_limit  = p["pl_limit"]
            total     = N * (N - 1) // 2
            completed = [0]

            self.sig_log.emit(
                f"링크 행렬 계산 중... (Node {N}개, {total}쌍)")

            def _calc_row(i):
                row = []
                for j in range(i + 1, N):
                    pl  = model.path_loss(
                        float(st_x[i]), float(st_y[i]),
                        float(st_x[j]), float(st_y[j]))
                    val = float(pl) if pl <= pl_limit else 0.0
                    row.append((i, j, val))
                return i, row

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = {pool.submit(_calc_row, i): i
                           for i in range(N - 1)}
                for fut in as_completed(futures):
                    i, row = fut.result()
                    for ri, rj, val in row:
                        matrix[ri, rj] = val
                        matrix[rj, ri] = val
                    completed[0] += 1
                    if completed[0] % max(1, (N - 1) // 10) == 0:
                        pct = completed[0] / (N - 1) * 100
                        self.sig_log.emit(
                            f"  {completed[0]}/{N-1}행 완료 ({pct:.0f}%)")

            n_linked = int((matrix > 0).sum() // 2)
            self.sig_log.emit(f"완료: 유효 링크 {n_linked}개")
            self.sig_done.emit(matrix)

        except Exception:
            import traceback
            self.sig_error.emit(traceback.format_exc())


class GWOptimizeWorker(QObject):
    sig_log   = pyqtSignal(str)
    sig_done  = pyqtSignal(object)
    sig_error = pyqtSignal(str)

    def __init__(self, spatial, nodes, pl_matrix, params):
        super().__init__()
        self.spatial   = spatial
        self.nodes     = nodes
        self.pl_matrix = pl_matrix
        self.params    = params

    def run(self):
        try:
            import pandas as pd
            from core.gw_optimizer import GWOptimizer

            # ── Node 좌표 → 3857 변환 (벡터화) ──────────────
            lons = np.array([n.lon for n in self.nodes])
            lats = np.array([n.lat for n in self.nodes])
            st_x, st_y = self.spatial.lonlat_to_xy(lons, lats)
            st_x = np.array(st_x, dtype=np.float64)
            st_y = np.array(st_y, dtype=np.float64)

            # ── 실제 지형 고도 계산 (벡터화) ─────────────────
            elevations = self.spatial.get_elevation_batch(st_x, st_y)
            valid_cnt  = int(np.sum(elevations > 0))
            self.sig_log.emit(
                f"Node 고도 계산 완료: {valid_cnt}/{len(self.nodes)}개 유효 "
                f"| 범위 {elevations.min():.0f}~{elevations.max():.0f}m")

            df = pd.DataFrame({
                "longitude"  : lons,
                "latitude"   : lats,
                "elevation_m": elevations,
            })

            p   = self.params
            opt = GWOptimizer(
                pl_matrix           = self.pl_matrix,
                stations            = df,
                spatial             = self.spatial,
                hb_gw               = p["hb_gw"],
                pl_limit            = p["pl_limit"],
                env                 = p["env"],
                fc                  = p["fc"],
                n_samples           = 100,
                seed                = p.get("seed", 42),
                min_cover           = p.get("min_cover", 3),
                max_stations_per_gw = p.get("max_cover", 0),
                use_traffic_weight  = p.get("use_traffic", True),
                optimize_hb         = p.get("opt_hb", False),
            )
            result = opt.run(progress_cb=self.sig_log.emit)

            # ── 미커버 Node 처리 ────────────────────────────────
            # truly_isolated: 물리적으로 어떤 GW로도 커버 불가 Node
            # → GW를 추가해도 커버 불가이므로 제외
            isolated   = getattr(result, 'truly_isolated', set())
            uncovered  = [i for i, gw_no in enumerate(result.node_gw)
                          if gw_no == 0 and i not in isolated]
            n_isolated = len([i for i, gw_no in enumerate(result.node_gw)
                              if gw_no == 0 and i in isolated])

            if n_isolated > 0:
                self.sig_log.emit(
                    f"물리적 커버 불가 Node {n_isolated}개 → GW 추가 불필요 (지형 차단)")

            if uncovered:
                self.sig_log.emit(
                    f"미커버 Node {len(uncovered)}개 → 바로 옆에 GW 추가 중...")

                gw_lons = list(result.gw_lon)
                gw_lats = list(result.gw_lat)
                node_gw = list(result.node_gw)

                for ni in uncovered:
                    nd = self.nodes[ni]
                    offset = 0.0001  # ~10m
                    gw_lons.append(nd.lon + offset)
                    gw_lats.append(nd.lat + offset)
                    gw_idx = len(gw_lons)
                    node_gw[ni] = gw_idx
                    self.sig_log.emit(
                        f"  Node{ni+1} → GW{gw_idx} 추가 "
                        f"({nd.lat:.5f}, {nd.lon:.5f})")

                result.gw_lon          = np.array(gw_lons)
                result.gw_lat          = np.array(gw_lats)
                result.node_gw         = np.array(node_gw)
                result.num_gw          = len(gw_lons)
                result.coverage        = float(
                    np.sum(np.array(node_gw) > 0) / len(node_gw))
                result.gw_cover_counts = np.bincount(
                    [g for g in node_gw if g > 0],
                    minlength=result.num_gw + 1)[1:]

                self.sig_log.emit(
                    f"미커버 해결 완료: GW {result.num_gw}개")
            else:
                self.sig_log.emit("모든 커버 가능 Node 배정 완료")

            self.sig_done.emit(result)

        except Exception:
            import traceback
            self.sig_error.emit(traceback.format_exc())


class GWOptimizeWindow(QDialog):
    sig_result_ready = pyqtSignal(object, list)

    def __init__(self, spatial, nodes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GW 최적 배치")
        self.setStyleSheet(STYLE)
        self.resize(540, 680)
        self.setWindowFlag(Qt.Window)

        self.spatial    = spatial
        self.nodes      = nodes
        self._thread    = None
        self._pl_matrix = None
        self._pl_limit  = 0.0   # _update_pl()에서 설정됨, AttributeError 방지

        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        self.lbl_info = QLabel(
            f"Node {len(self.nodes)}개 기준으로 최적 GW 위치를 탐색합니다.")
        self.lbl_info.setStyleSheet(f"color:{MUTED};font-size:12px;")
        lay.addWidget(self.lbl_info)

        pg = QGroupBox("전파 파라미터")
        pg.setStyleSheet(
            f"QGroupBox{{color:{MUTED};border:1px solid {BORDER};"
            f"border-radius:6px;margin-top:6px;padding-top:8px;}}"
            f"QGroupBox::title{{subcontrol-origin:margin;left:8px;}}")
        fl = QFormLayout(pg); fl.setSpacing(8)

        def _sp(lo, hi, val, dec=1, suf=""):
            s = QDoubleSpinBox()
            s.setRange(lo, hi); s.setValue(val)
            s.setDecimals(dec); s.setSuffix(suf)
            return s

        self.sp_hm     = _sp(0.1, 50,   1.5, 1, " m")
        self.sp_hb     = _sp(1,   200,  15.0,1, " m")
        self.sp_pt     = _sp(-30, 50,   14.0,1, " dBm")
        self.sp_gt     = _sp(0,   30,   3.0,  2, " dBi")
        self.sp_lt     = _sp(0,   20,   0.0,  2, " dB")
        self.sp_p_edge = _sp(0.5, 1.0,  0.9, 2)
        self.sp_fc     = _sp(400, 2000, 915., 1, " MHz")

        fl.addRow("Node 안테나 높이 hm", self.sp_hm)
        fl.addRow("GW 안테나 높이 hb",   self.sp_hb)
        fl.addRow("GW 송신 출력 Pt",     self.sp_pt)
        fl.addRow("GW 안테나 이득 Gt",   self.sp_gt)
        fl.addRow("GW 케이블 손실 Lt",   self.sp_lt)
        fl.addRow("P_edge",              self.sp_p_edge)
        fl.addRow("주파수",              self.sp_fc)

        self.lbl_pl = QLabel()
        self.lbl_pl.setStyleSheet(f"color:{MUTED};font-size:11px;")
        fl.addRow("", self.lbl_pl)
        self._update_pl()
        for sp in [self.sp_p_edge, self.sp_pt, self.sp_gt, self.sp_lt]:
            sp.valueChanged.connect(self._update_pl)
        lay.addWidget(pg)

        gg = QGroupBox("GW 배치 파라미터")
        gg.setStyleSheet(pg.styleSheet())
        gl = QFormLayout(gg); gl.setSpacing(8)

        self.sp_min_cov = QSpinBox()
        self.sp_min_cov.setRange(1, 20); self.sp_min_cov.setValue(3)
        self.sp_max_cov = QSpinBox()
        self.sp_max_cov.setRange(0, 500); self.sp_max_cov.setValue(0)
        self.sp_max_cov.setToolTip("0 = 무제한")
        self.sp_seed = QSpinBox()
        self.sp_seed.setRange(0, 9999); self.sp_seed.setValue(42)

        self.chk_traffic = QCheckBox("트래픽 가중치 사용")
        self.chk_traffic.setChecked(True)
        self.chk_traffic.setStyleSheet(f"color:{TEXT};")
        self.chk_opt_hb = QCheckBox("안테나 높이 자동 최적화 (느림)")
        self.chk_opt_hb.setStyleSheet(f"color:{TEXT};")

        gl.addRow("GW 최소 커버 수", self.sp_min_cov)
        gl.addRow("GW 최대 커버 수", self.sp_max_cov)
        gl.addRow("랜덤 시드",       self.sp_seed)
        gl.addRow("",               self.chk_traffic)
        gl.addRow("",               self.chk_opt_hb)
        lay.addWidget(gg)

        btn_row = QHBoxLayout()
        self.btn_step1 = QPushButton("1단계: 링크 행렬 계산")
        self.btn_step1.setStyleSheet(BTN)
        self.btn_step2 = QPushButton("2단계: GW 최적 배치")
        self.btn_step2.setStyleSheet(BTN_RUN)
        self.btn_step2.setEnabled(False)
        btn_row.addWidget(self.btn_step1)
        btn_row.addWidget(self.btn_step2)
        lay.addLayout(btn_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        lay.addWidget(self.progress)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setFixedHeight(200)
        lay.addWidget(self.log)

        self.lbl_result = QLabel("─")
        self.lbl_result.setStyleSheet(
            f"color:{MUTED};font-size:12px;padding:4px;")
        lay.addWidget(self.lbl_result)

        self.btn_step1.clicked.connect(self._run_step1)
        self.btn_step2.clicked.connect(self._run_step2)

    def set_nodes(self, nodes):
        self.nodes = nodes
        self.lbl_info.setText(
            f"Node {len(self.nodes)}개 기준으로 최적 GW 위치를 탐색합니다.")
        self._pl_matrix = None
        self.btn_step2.setEnabled(False)
        self.log.clear()
        self.lbl_result.setText("─")

    def _update_pl(self):
        p    = self.sp_p_edge.value()
        pt   = self.sp_pt.value()
        gt   = self.sp_gt.value()
        lt   = self.sp_lt.value()
        eirp = pt + gt - lt
        mg   = 53.383*p**3 - 80.075*p**2 + 57.512*p - 15.41
        pl   = eirp - (-137.0 + mg)
        try:
            d = 10 ** ((pl - 118.4) / 24.4)
        except Exception:
            d = 0
        self.lbl_pl.setText(
            f"EIRP = {eirp:.1f} dBm  |  PL_limit = {pl:.2f} dB  (최대 ~{d:.1f}km)")
        self._pl_limit = pl

    def _params(self):
        return dict(
            hm          = self.sp_hm.value(),
            hb_gw       = self.sp_hb.value(),
            pt          = self.sp_pt.value(),
            gt          = self.sp_gt.value(),
            lt          = self.sp_lt.value(),
            p_edge      = self.sp_p_edge.value(),
            fc          = self.sp_fc.value(),
            env         = 2,
            pl_limit    = self._pl_limit,
            min_cover   = self.sp_min_cov.value(),
            max_cover   = self.sp_max_cov.value(),
            seed        = self.sp_seed.value(),
            use_traffic = self.chk_traffic.isChecked(),
            opt_hb      = self.chk_opt_hb.isChecked(),
        )

    def _run_step1(self):
        if not self.nodes:
            QMessageBox.warning(self, "경고", "Node를 먼저 추가하세요.")
            return
        self._set_busy(True)
        self._log("=" * 40)
        self._log(f"[1단계] 링크 행렬 계산 시작 (Node {len(self.nodes)}개)")
        w = LinkMatrixWorker(self.spatial, self.nodes, self._params())
        w.sig_log.connect(self._log)
        w.sig_done.connect(self._on_step1_done)
        w.sig_error.connect(self._on_error)
        self._start(w)

    def _on_step1_done(self, matrix):
        self._pl_matrix = matrix
        self._set_busy(False)
        self.btn_step2.setEnabled(True)
        n_links = int((matrix > 0).sum() // 2)
        self._log(f"[1단계 완료] 유효 링크: {n_links}개")
        self.lbl_result.setText(
            f"1단계 완료 — 유효 링크 {n_links}개 | 2단계를 실행하세요.")
        self.lbl_result.setStyleSheet(
            "color:#FFD700;font-size:12px;padding:4px;")

    def _run_step2(self):
        if self._pl_matrix is None:
            QMessageBox.warning(self, "경고", "1단계를 먼저 실행하세요.")
            return
        self._set_busy(True)
        self._log("=" * 40)
        self._log("[2단계] GW 최적 배치 시작")
        w = GWOptimizeWorker(
            self.spatial, self.nodes,
            self._pl_matrix, self._params())
        w.sig_log.connect(self._log)
        w.sig_done.connect(self._on_step2_done)
        w.sig_error.connect(self._on_error)
        self._start(w)

    def _on_step2_done(self, result):
        self._set_busy(False)
        n_gw = result.num_gw
        cov  = result.coverage * 100
        avg  = float(result.gw_cover_counts.mean())                if len(result.gw_cover_counts) else 0
        self._log(
            f"[2단계 완료] GW {n_gw}개 | 커버리지 {cov:.1f}% "
            f"| 평균 {avg:.1f}개/GW")
        col = "#00C94A" if cov >= 90 else "#FFD700" if cov >= 70 else "#FF6B00"
        self.lbl_result.setText(
            f"GW {n_gw}개 | 커버리지 {cov:.1f}% | 평균 {avg:.1f}개/GW")
        self.lbl_result.setStyleSheet(
            f"color:{col};font-size:12px;padding:4px;font-weight:bold;")
        self.sig_result_ready.emit(result, self.nodes)

    def _start(self, worker):
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(3000)
        t = QThread(self)
        self._thread = t
        self._worker = worker
        worker.moveToThread(t)
        t.started.connect(worker.run)
        worker.sig_done.connect(t.quit)
        worker.sig_error.connect(t.quit)
        t.start()

    def _set_busy(self, busy):
        self.btn_step1.setEnabled(not busy)
        self.btn_step2.setEnabled(not busy and self._pl_matrix is not None)
        self.progress.setVisible(busy)

    def _log(self, msg):
        self.log.appendPlainText(msg)
        self.log.verticalScrollBar().setValue(
            self.log.verticalScrollBar().maximum())

    def _on_error(self, msg):
        self._set_busy(False)
        self._log(f"[오류] {msg}")
        print(f"[ERROR] {msg}", flush=True)
