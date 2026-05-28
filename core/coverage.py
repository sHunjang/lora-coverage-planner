# core/coverage.py
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


@dataclass
class GWEntry:
    """게이트웨이(GW) 파라미터 컨테이너."""
    callsign : str   = "GW1"
    lon      : float = 127.10
    lat      : float = 37.40
    pt_dbm   : float = 14.0
    gt_dbi   : float = 2.15
    lt_db    : float = 0.0
    hb_m     : float = 15.0
    enabled  : bool  = True


@dataclass
class NodeEntry:
    """단말(Node) 파라미터 컨테이너."""
    callsign      : str   = "Node1"
    lon           : float = 127.10
    lat           : float = 37.40
    gr_dbi        : float = 2.15
    lr_db         : float = 0.0
    hm_m          : float = 1.5
    min_rx_dbm    : float = -126.6
    indoor_loss_db: float = 0.0


@dataclass
class LinkResult:
    """단일 Node의 링크 분석 결과."""
    covered   : bool  = False
    best_gw   : str   = ""
    best_pr   : float = -999.0
    gw_prs    : dict  = field(default_factory=dict)
    macro_pr  : float = -999.0
    n_rx_gw   : int   = 0
    # SNR 관련 필드 추가
    best_snr  : float = -999.0   # 최고 SNR (dB)
    snr_margin: float = -999.0   # SNR 마진 = SNR - SF별 임계값 (dB)
    link_ok   : bool  = False    # SNR 마진 기준 통신 성공 여부
    # 트래픽 용량 분석 필드
    adr_sf    : int   = 12      # ADR 결정 SF (7~12)
    toa_ms    : float = 0.0     # 이 Node의 ToA (ms)


@dataclass
class CoverageResult:
    """전체 커버리지 분석 결과."""
    nodes                : list  = field(default_factory=list)
    gw_counts            : dict  = field(default_factory=dict)
    n_covered            : int   = 0
    n_total              : int   = 0
    macro_diversity_gain : float = 0.0
    avg_n_rx_gw          : float = 0.0
    adr_sf_distribution  : dict  = field(default_factory=dict)
    avg_toa_ms           : float = 0.0
    # 통신 성공율 관련 필드 추가
    cell_success_rate    : float = 0.0   # 셀 전체 통신 성공율 (%)
    edge_success_rate    : float = 0.0   # 셀 경계 통신 성공율 (%)
    avg_snr              : float = 0.0   # 평균 SNR (dB)
    avg_snr_margin       : float = 0.0   # 평균 SNR 마진 (dB)
    # 트래픽 용량 분석 필드
    gw_traffic           : dict  = field(default_factory=dict)
    # {callsign: {'load_pct': float, 'pdr': float, 'n_nodes': int,
    #             'total_toa_ms': float, 'overloaded': bool}}
    avg_pdr              : float = 0.0   # 전체 평균 PDR (%)
    n_overloaded_gw      : int   = 0     # 과부하 GW 수
    avg_load_pct         : float = 0.0   # 평균 트래픽 부하 (%)

    @property
    def coverage_pct(self):
        return self.n_covered / self.n_total * 100 if self.n_total else 0


# 히트맵 격자 간격 최솟값
STEP_MIN_HM = 0.0005

# LoRa SF별 SNR 임계값 (dB) — 표준 스펙
SF_SNR_THRESH = {
    7: -7.5, 8: -10.0, 9: -12.5,
    10: -15.0, 11: -17.5, 12: -20.0,
}

# 셀 경계 기준 SNR 마진 임계값 (dB)
# 이 값 미만이면 셀 경계(edge) Node로 분류
EDGE_MARGIN_THRESH = 3.0


class CoverageEngine:
    """커버리지 계산 엔진."""

    def __init__(self, spatial, env=2, fc=915.0, n_samples=100, settings=None):
        self.spatial   = spatial
        self.env       = env
        self.fc        = fc
        self.n_samples = n_samples
        self.settings  = settings or {}

    def _model(self, hb, hm):
        from core.propagation import PathLossModel
        prop_model = self.settings.get('prop_model', 'smartcity')
        return PathLossModel(
            self.spatial,
            h_station  = hm,
            hb_gw      = hb,
            env        = self.env,
            fc         = self.fc,
            n_samples  = self.n_samples,
            prop_model = prop_model,
        )

    def _calc_snr(self, pr_dbm: float, sf: int) -> tuple[float, float]:
        """
        수신전력과 SF로 SNR 및 SNR 마진 계산.

        SNR = Pr - 열잡음 - 잡음지수
        SNR 마진 = SNR - SF별 임계값

        Args:
            pr_dbm: 수신전력 (dBm)
            sf    : Spreading Factor (7~12)
        Returns:
            (snr_db, snr_margin_db)
        """
        bw_khz  = self.settings.get('bandwidth_khz',  125.0)
        nf_db   = self.settings.get('noise_figure_db', 6.0)

        # 열잡음 = -174 + 10*log10(BW_Hz) dBm
        thermal_noise_dbm = -174.0 + 10.0 * np.log10(bw_khz * 1000.0)

        # SNR = Pr - (열잡음 + NF)
        snr_db = pr_dbm - (thermal_noise_dbm + nf_db)

        # SNR 마진 = SNR - SF별 임계값
        sf_thresh   = SF_SNR_THRESH.get(sf, -20.0)
        snr_margin  = snr_db - sf_thresh

        return round(snr_db, 2), round(snr_margin, 2)

    def run(self, gws, nodes, cb=None):
            """
            Node별 수신전력/SNR을 계산하여 커버리지 및 통신 성공율을 분석합니다.

            병렬화 전략:
            - Node 단위로 ThreadPoolExecutor 병렬 실행
            - 각 Node의 GW별 수신전력 계산은 독립적이므로 race condition 없음
            - PathLossModel을 GW별로 미리 생성하여 스레드 간 공유
                (모델 객체 자체는 읽기 전용이므로 thread-safe)
            - 결과는 Node 인덱스 기준으로 수집 후 순서대로 조립
            """
            def _log(m):
                if cb: cb(m)

            SF_SENS = {
                7: -123.0, 8: -126.0, 9: -129.0,
                10: -132.0, 11: -134.5, 12: -137.0,
            }
            SF_TOA = {
                7: 61.7, 8: 123.4, 9: 246.8,
                10: 493.5, 11: 987.1, 12: 1974.1,
            }

            active = [g for g in gws if g.enabled]
            result = CoverageResult(n_total=len(nodes))
            for g in active:
                result.gw_counts[g.callsign] = 0

            if not active or not nodes:
                return result

            # ── 좌표 사전 변환 ───────────────────────────────────
            # 메인 스레드에서 한 번만 변환하여 워커에 전달
            gw_xy = {g.callsign: self.spatial.lonlat_to_xy(g.lon, g.lat)
                    for g in active}
            nd_xy = [self.spatial.lonlat_to_xy(n.lon, n.lat) for n in nodes]

            # GW별 좌표를 float 튜플로 변환 (스레드 안전)
            gw_coords = {
                g.callsign: (
                    float(gw_xy[g.callsign][0]),
                    float(gw_xy[g.callsign][1]),
                )
                for g in active
            }

            # ── GW별 PathLossModel 사전 생성 ─────────────────────
            # 스레드마다 모델을 새로 생성하면 오버헤드가 크므로
            # GW별로 미리 생성하여 공유 (읽기 전용 사용이므로 thread-safe)
            # Node hm_m이 다를 수 있으므로 (hb_gw, hm_m) 조합별 캐싱
            _model_cache: dict = {}

            def _get_model(hb: float, hm: float):
                key = (round(hb, 3), round(hm, 3))
                if key not in _model_cache:
                    _model_cache[key] = self._model(hb, hm)
                return _model_cache[key]

            _log(f"분석 시작: GW {len(active)}개 × Node {len(nodes)}개 "
                f"(병렬 처리)")

            n_workers = min(os.cpu_count() or 4, 16)

            # ── Node별 계산 함수 (워커) ──────────────────────────
            def _calc_node(ni: int):
                """
                단일 Node에 대해 모든 활성 GW의 수신전력을 계산합니다.
                반환: (ni, LinkResult, adr_sf, cov)
                """
                nd  = nodes[ni]
                nx  = float(nd_xy[ni][0])
                ny  = float(nd_xy[ni][1])
                indoor = getattr(nd, 'indoor_loss_db', 0.0)

                gw_prs  = {}
                best_pr = -999.0
                best_gw = ""

                for gw in active:
                    gx, gy = gw_coords[gw.callsign]
                    model  = _get_model(gw.hb_m, nd.hm_m)
                    pl     = model.path_loss(gx, gy, nx, ny)
                    pr     = (gw.pt_dbm + gw.gt_dbi - gw.lt_db
                            - pl + nd.gr_dbi - nd.lr_db - indoor)
                    gw_prs[gw.callsign] = round(float(pr), 1)
                    if pr > best_pr:
                        best_pr, best_gw = pr, gw.callsign

                # 수신 가능 GW 목록
                rx_gws = [cs for cs, pr in gw_prs.items()
                        if pr >= nd.min_rx_dbm]
                n_rx   = len(rx_gws)

                # 매크로 다이버시티
                if n_rx >= 2:
                    linear_sum = sum(10 ** (gw_prs[cs] / 10) for cs in rx_gws)
                    macro_pr   = 10 * np.log10(linear_sum)
                    gain       = macro_pr - best_pr
                else:
                    macro_pr = best_pr
                    gain     = 0.0

                # 커버 여부
                cov = best_pr >= nd.min_rx_dbm

                # ADR SF 결정
                adr_sf = 12
                if cov:
                    for sf in sorted(SF_SENS.keys()):
                        if best_pr >= SF_SENS[sf]:
                            adr_sf = sf
                            break

                # SNR 및 마진 계산
                best_snr, snr_margin = -999.0, -999.0
                link_ok = False
                if best_pr > -999:
                    bw_khz = self.settings.get('bandwidth_khz', 125.0)
                    nf_db  = self.settings.get('noise_figure_db', 6.0)
                    thermal = -174.0 + 10.0 * np.log10(bw_khz * 1000.0)
                    best_snr   = round(best_pr - (thermal + nf_db), 2)
                    sf_thresh  = SF_SNR_THRESH.get(adr_sf, -20.0)
                    snr_margin = round(best_snr - sf_thresh, 2)
                    link_ok    = snr_margin > 0.0

                link = LinkResult(
                    covered    = cov,
                    best_gw    = best_gw,
                    best_pr    = round(best_pr, 1),
                    gw_prs     = gw_prs,
                    macro_pr   = round(macro_pr, 1),
                    n_rx_gw    = n_rx,
                    best_snr   = best_snr,
                    snr_margin = snr_margin,
                    link_ok    = link_ok,
                )
                return ni, link, gain, adr_sf, cov

            # ── 병렬 실행 ────────────────────────────────────────
            node_results = [None] * len(nodes)   # 순서 보장을 위한 배열
            completed    = [0]

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_calc_node, ni): ni
                        for ni in range(len(nodes))}

                for fut in as_completed(futures):
                    ni, link, gain, adr_sf, cov = fut.result()
                    node_results[ni] = (link, gain, adr_sf, cov)
                    completed[0] += 1

                    # 진행률 콜백 (10% 단위)
                    n_done = completed[0]
                    n_tot  = len(nodes)
                    if n_done % max(1, n_tot // 10) == 0 or n_done == n_tot:
                        _log(f"  {n_done}/{n_tot} ({n_done/n_tot*100:.0f}%)")
            
            # ── 결과 조립 (순서대로) ─────────────────────────────
            macro_gains     = []
            n_rx_gw_list    = []
            adr_sf_dist     = {sf: 0 for sf in range(7, 13)}
            toa_list        = []
            snr_list        = []
            snr_margin_list = []
            n_cell_total = n_cell_ok = 0
            n_edge_total = n_edge_ok = 0

            # 트래픽 용량 분석용 — GW별 ToA 누적
            # {callsign: 총 ToA (ms/시간)}
            gw_toa_accum = {g.callsign: 0.0 for g in active}

            # 전송 주기 설정 (초) — settings에서 가져오거나 기본값 사용
            # Node가 평균적으로 몇 초마다 한 번 패킷을 전송하는지
            tx_interval_s = float(
                self.settings.get('tx_interval_s', 300.0))  # 기본 5분

            for ni, (link, gain, adr_sf, cov) in enumerate(node_results):
                # LinkResult에 adr_sf, toa_ms 저장
                toa_ms = SF_TOA.get(adr_sf, 0.0) if cov else 0.0
                link.adr_sf = adr_sf
                link.toa_ms = toa_ms

                result.nodes.append(link)

                macro_gains.append(gain)
                n_rx_gw_list.append(link.n_rx_gw)

                if cov:
                    adr_sf_dist[adr_sf] = adr_sf_dist.get(adr_sf, 0) + 1
                    toa_list.append(toa_ms)
                    snr_list.append(link.best_snr)
                    snr_margin_list.append(link.snr_margin)

                    result.n_covered += 1
                    result.gw_counts[link.best_gw] = (
                        result.gw_counts.get(link.best_gw, 0) + 1)

                    # 통신 성공율 집계
                    n_cell_total += 1
                    if link.link_ok:
                        n_cell_ok += 1
                    if link.snr_margin < EDGE_MARGIN_THRESH:
                        n_edge_total += 1
                        if link.link_ok:
                            n_edge_ok += 1

                    # 트래픽 용량: GW별 ToA 누적
                    # Node 1개당 시간당 ToA = ToA_ms * (3600 / tx_interval_s)
                    toa_per_hour_ms = toa_ms * (3600.0 / tx_interval_s)
                    if link.best_gw in gw_toa_accum:
                        gw_toa_accum[link.best_gw] += toa_per_hour_ms

            # ── 통계 집계 ────────────────────────────────────────
            result.macro_diversity_gain = (float(np.mean(macro_gains))
                                           if macro_gains else 0.0)
            result.avg_n_rx_gw          = (float(np.mean(n_rx_gw_list))
                                           if n_rx_gw_list else 0.0)
            result.adr_sf_distribution  = adr_sf_dist
            result.avg_toa_ms           = (float(np.mean(toa_list))
                                           if toa_list else 0.0)
            result.avg_snr              = (float(np.mean(snr_list))
                                           if snr_list else 0.0)
            result.avg_snr_margin       = (float(np.mean(snr_margin_list))
                                           if snr_margin_list else 0.0)
            result.cell_success_rate    = (n_cell_ok / n_cell_total * 100
                                           if n_cell_total > 0 else 0.0)
            result.edge_success_rate    = (n_edge_ok / n_edge_total * 100
                                           if n_edge_total > 0 else 0.0)

            # ── 트래픽 용량 분석 ─────────────────────────────────
            # LoRa GW 채널 용량: 1시간(3,600,000ms) × 채널 수(8) × duty cycle(1%)
            # 실제 사용 가능한 시간 = 3,600,000ms × 8ch × 0.01 = 288,000ms/h
            N_CHANNELS   = 8      # LoRa GW 동시 수신 채널 수
            DUTY_CYCLE   = 0.01   # LoRa 듀티 사이클 (1%)
            capacity_ms  = 3_600_000.0 * N_CHANNELS * DUTY_CYCLE  # 288,000 ms/h

            # 과부하 기준: 채널 용량의 80% 이상
            OVERLOAD_THRESH = 80.0

            gw_traffic   = {}
            pdr_list     = []
            load_pct_list= []

            for gw in active:
                cs       = gw.callsign
                toa_used = gw_toa_accum.get(cs, 0.0)  # ms/h
                n_nodes  = result.gw_counts.get(cs, 0)

                # 트래픽 부하 (%)
                load_pct = min(toa_used / capacity_ms * 100, 999.9)

                # ALOHA 기반 PDR 계산
                # G = 트래픽 강도 (normalized load)
                # PDR = e^(-2G) — Pure ALOHA
                # G가 0이면 PDR=100%, G가 클수록 PDR 감소
                G   = toa_used / (3_600_000.0 * N_CHANNELS)
                pdr = float(np.exp(-2.0 * G)) * 100.0  # %
                pdr = max(0.0, min(100.0, pdr))

                overloaded = load_pct >= OVERLOAD_THRESH

                gw_traffic[cs] = {
                    'load_pct'    : round(load_pct, 1),
                    'pdr'         : round(pdr, 1),
                    'n_nodes'     : n_nodes,
                    'total_toa_ms': round(toa_used, 1),
                    'overloaded'  : overloaded,
                }
                pdr_list.append(pdr)
                load_pct_list.append(load_pct)

            result.gw_traffic      = gw_traffic
            result.avg_pdr         = float(np.mean(pdr_list)) if pdr_list else 100.0
            result.n_overloaded_gw = sum(
                1 for v in gw_traffic.values() if v['overloaded'])
            result.avg_load_pct    = float(np.mean(load_pct_list)) if load_pct_list else 0.0

            _log(f"완료: {result.n_covered}/{result.n_total}개 "
                 f"({result.coverage_pct:.1f}%) | "
                 f"셀 성공율 {result.cell_success_rate:.1f}% | "
                 f"경계 성공율 {result.edge_success_rate:.1f}% | "
                 f"평균 PDR {result.avg_pdr:.1f}% | "
                 f"평균 부하 {result.avg_load_pct:.1f}%")
            return result

    def heatmap(self, gw, min_rx, step=0.0015, cb=None,
                use_deygout=False, radius_km=25.0,
                pr_min=None, pr_max=None):
        """단일 GW 격자 히트맵 생성."""
        import base64, io
        from pyproj import Transformer
        from scipy.ndimage import gaussian_filter

        actual_step = max(step, STEP_MIN_HM)
        upscale     = 8 if actual_step >= STEP_MIN_HM else 4
        if actual_step != step and cb:
            cb(f"step {step:.5f}° → {actual_step:.5f}° 자동 조정")

        b = self.spatial.bounds
        lmin, latmin, lmax, latmax = b[0], b[1], b[2], b[3]

        lons = np.arange(lmin, lmax, actual_step)
        lats = np.arange(latmin, latmax, actual_step)
        lon2d, lat2d = np.meshgrid(lons, lats)
        fl = lon2d.ravel(); fa = lat2d.ravel()

        try:
            from shapely import points as _sp, contains as _sc
            mask = _sc(self.spatial.polygon_4326,
                       _sp(np.stack([fl, fa], axis=1)))
        except Exception:
            from shapely.geometry import Point
            mask = np.array([self.spatial.polygon_4326.contains(
                Point(lo, la)) for lo, la in zip(fl, fa)])

        tr = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
        px, py         = tr.transform(fl, fa)
        gx_arr, gy_arr = tr.transform(gw.lon, gw.lat)
        gx, gy         = float(gx_arr), float(gy_arr)

        nd_hm = float(self.settings.get('nd_hm_m', 1.5))
        from core.propagation import PathLossModel
        model = PathLossModel(
            self.spatial,
            h_station  = nd_hm,
            hb_gw      = float(gw.hb_m),
            env        = self.env,
            fc         = self.fc,
            n_samples  = self.n_samples,
            diff_order = 2,
            prop_model = self.settings.get('prop_model', 'smartcity'),
        )

        eirp      = float(gw.pt_dbm + gw.gt_dbi - gw.lt_db)
        idx       = np.where(mask)[0]
        pf        = np.full(len(px), float(min_rx) - 50.0)
        px_idx    = px.astype(np.float64)[idx]
        py_idx    = py.astype(np.float64)[idx]
        n_workers = min(os.cpu_count() or 4, 16)

        if cb: cb(f"히트맵 계산 중... ({len(idx):,}개 격자점)")

        def _calc(k):
            pl = model.path_loss(gx, gy,
                                 float(px_idx[k]),
                                 float(py_idx[k]))
            return k, eirp - pl

        results   = np.full(len(idx), float(min_rx) - 50.0)
        completed = [0]
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_calc, k): k for k in range(len(idx))}
            for fut in as_completed(futures):
                k, pr = fut.result()
                results[k] = pr
                completed[0] += 1
                if cb and completed[0] % max(1, len(idx) // 20) == 0:
                    cb(f"  {completed[0]:,}/{len(idx):,} "
                       f"({completed[0]/len(idx)*100:.0f}%)")

        pf[idx] = results

        pg            = pf.reshape(lon2d.shape)
        boundary_mask = mask.reshape(lon2d.shape)
        pg_masked     = np.where(boundary_mask, pg, np.nan)
        pg_filled     = np.where(np.isnan(pg_masked),
                                 float(min_rx) - 50.0, pg_masked)

        ps = gaussian_filter(pg_filled.astype(float), sigma=0)
        ps = np.where(boundary_mask, ps, float(min_rx) - 50.0)
        cm = (ps >= min_rx) & boundary_mask

        color_levels = self.settings.get('color_levels')
        url = self._render_heatmap_image(
            ps, cm, boundary_mask, min_rx, color_levels, upscale=upscale)

        if cb: cb("히트맵 완료")
        return {
            'url'     : url,
            'bounds'  : [[float(latmin), float(lmin)],
                         [float(latmax), float(lmax)]],
            'callsign': gw.callsign,
            'min_rx'  : min_rx,
            'ps'      : ps,
            'cm'      : cm,
            'lon_min' : float(lmin),
            'lat_min' : float(latmin),
            'step'    : actual_step,
        }

    def heatmap_combined(self, gws, min_rx, step=0.0015,
                         cb=None, radius_km=25.0,
                         pr_min=None, pr_max=None):
        """다중 GW 합성 격자 히트맵 생성."""
        import base64, io
        from pyproj import Transformer
        from scipy.ndimage import gaussian_filter

        actual_step = max(step, STEP_MIN_HM)
        upscale     = 8 if actual_step >= STEP_MIN_HM else 4
        if actual_step != step and cb:
            cb(f"step {step:.5f}° → {actual_step:.5f}° 자동 조정")

        b = self.spatial.bounds
        lmin, latmin, lmax, latmax = b[0], b[1], b[2], b[3]

        lons = np.arange(lmin, lmax, actual_step)
        lats = np.arange(latmin, latmax, actual_step)
        lon2d, lat2d = np.meshgrid(lons, lats)
        fl = lon2d.ravel(); fa = lat2d.ravel()

        try:
            from shapely import points as _sp, contains as _sc
            mask = _sc(self.spatial.polygon_4326,
                       _sp(np.stack([fl, fa], axis=1)))
        except Exception:
            from shapely.geometry import Point
            mask = np.array([self.spatial.polygon_4326.contains(
                Point(lo, la)) for lo, la in zip(fl, fa)])

        tr = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
        px, py = tr.transform(fl, fa)
        px_f   = px.astype(np.float64)
        py_f   = py.astype(np.float64)

        pr_max_grid = np.full(len(px), float(min_rx) - 50.0)
        gw_idx_grid = np.full(len(px), -1, dtype=np.int32)

        idx    = np.where(mask)[0]
        px_idx = px_f[idx]
        py_idx = py_f[idx]

        nd_hm     = float(self.settings.get('nd_hm_m', 1.5))
        n_workers = min(os.cpu_count() or 4, 16)
        from core.propagation import PathLossModel

        for gi, gw in enumerate(gws):
            if cb: cb(f"히트맵 계산 중... GW {gi+1}/{len(gws)}: {gw.callsign}")

            gx_arr, gy_arr = tr.transform(gw.lon, gw.lat)
            gx, gy = float(gx_arr), float(gy_arr)
            eirp   = float(gw.pt_dbm + gw.gt_dbi - gw.lt_db)

            model = PathLossModel(
                self.spatial,
                h_station  = nd_hm,
                hb_gw      = float(gw.hb_m),
                env        = self.env,
                fc         = self.fc,
                n_samples  = self.n_samples,
                diff_order = 2,
                prop_model = self.settings.get('prop_model', 'smartcity'),
            )

            def _calc_combined(k, _gx=gx, _gy=gy, _eirp=eirp, _model=model):
                pl = _model.path_loss(_gx, _gy,
                                      float(px_idx[k]),
                                      float(py_idx[k]))
                return k, _eirp - pl

            results_c = np.full(len(idx), float(min_rx) - 50.0)
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_calc_combined, k): k
                           for k in range(len(idx))}
                for fut in as_completed(futures):
                    k, pr = fut.result()
                    results_c[k] = pr

            for k, i in enumerate(idx):
                pr = results_c[k]
                if pr > pr_max_grid[i]:
                    pr_max_grid[i] = pr
                    gw_idx_grid[i] = gi

        pg            = pr_max_grid.reshape(lon2d.shape)
        boundary_mask = mask.reshape(lon2d.shape)
        pg_masked     = np.where(boundary_mask, pg, np.nan)
        pg_filled     = np.where(np.isnan(pg_masked),
                                 float(min_rx) - 50.0, pg_masked)

        ps = gaussian_filter(pg_filled.astype(float), sigma=0)
        ps = np.where(boundary_mask, ps, float(min_rx) - 50.0)

        cm        = (ps >= min_rx) & boundary_mask
        gw_idx_2d = gw_idx_grid.reshape(lon2d.shape)

        color_levels = self.settings.get('color_levels')
        gw_color_map = self.settings.get('gw_color_map', {})

        url = self._render_heatmap_gw_colors(
            ps, cm, boundary_mask, min_rx,
            gw_idx_2d, gws, gw_color_map, color_levels,
            upscale=upscale)

        if cb: cb(f"합성 히트맵 완료 ({len(gws)}개 GW)")
        return {
            'url'     : url,
            'bounds'  : [[float(latmin), float(lmin)],
                         [float(latmax), float(lmax)]],
            'callsign': 'COMBINED',
            'type'    : 'combined',
            'gws'     : [g.callsign for g in gws],
            'min_rx'  : min_rx,
            'ps'      : ps,
            'cm'      : cm,
            'gw_idx'  : gw_idx_2d,
            'lon_min' : float(lmin),
            'lat_min' : float(latmin),
            'step'    : actual_step,
        }

    def env_map(self, step=0.003, cb=None):
        """DSM 기반 전파 환경 분류 지도."""
        import base64, io
        from PIL import Image
        from pyproj import Transformer

        b = self.spatial.bounds
        lmin, latmin, lmax, latmax = b
        lons = np.arange(lmin, lmax, step)
        lats = np.arange(latmin, latmax, step)
        lon2d, lat2d = np.meshgrid(lons, lats)
        fl = lon2d.ravel(); fa = lat2d.ravel()

        try:
            from shapely import points as _sp, contains as _sc
            mask = _sc(self.spatial.polygon_4326,
                       _sp(np.stack([fl, fa], axis=1)))
        except Exception:
            from shapely.geometry import Point
            mask = np.array([self.spatial.polygon_4326.contains(
                Point(lo, la)) for lo, la in zip(fl, fa)])

        tr = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
        px, py   = tr.transform(fl, fa)
        idx      = np.where(mask)[0]
        env_grid = np.zeros(len(px), dtype=np.uint8)

        if cb: cb(f"환경 분류 계산 중... ({len(idx)}개 포인트)")
        for i in idx:
            env_grid[i] = self.spatial.get_env_code(
                float(px[i]), float(py[i]))

        eg            = env_grid.reshape(lon2d.shape)
        boundary_mask = mask.reshape(lon2d.shape)

        ENV_COLORS = {
            1: (220, 50,  50,  160),
            2: (230, 140, 30,  140),
            3: (220, 210, 40,  120),
            4: (50,  180, 80,  100),
        }
        rgba = np.zeros((*lon2d.shape, 4), dtype=np.uint8)
        for code, color in ENV_COLORS.items():
            rgba[(eg == code) & boundary_mask] = color

        img = Image.fromarray(rgba[::-1, :, :], 'RGBA')
        w, h = img.size
        img  = img.resize((w*2, h*2), Image.NEAREST)
        buf  = io.BytesIO()
        img.save(buf, 'PNG', optimize=True)
        url  = f"data:image/png;base64,{__import__('base64').b64encode(buf.getvalue()).decode()}"

        if cb: cb("환경 분류 시각화 완료")
        return {
            'url'     : url,
            'bounds'  : [[float(latmin), float(lmin)],
                         [float(latmax), float(lmax)]],
            'callsign': 'ENV_MAP',
            'type'    : 'env_map',
        }

    @staticmethod
    def _render_heatmap_image(ps, cm, boundary_mask, min_rx,
                              color_levels=None, upscale=8):
        """단일 GW 히트맵 이미지 렌더링. 범례 외 커버 영역도 표시."""
        import io, base64
        from PIL import Image
        import numpy as np

        rows, cols = ps.shape
        rgba = np.zeros((rows, cols, 4), dtype=np.uint8)

        if color_levels:
            levels    = sorted(color_levels, key=lambda x: -x['pr'])
            pr_min    = float(levels[-1]['pr'])
            pr_max_lv = float(levels[0]['pr'])

            cm_display = cm & boundary_mask

            r_arr    = np.zeros((rows, cols), dtype=np.uint8)
            g_arr    = np.zeros((rows, cols), dtype=np.uint8)
            b_arr    = np.zeros((rows, cols), dtype=np.uint8)
            assigned = np.zeros((rows, cols), dtype=bool)

            for lv in reversed(levels):
                hx = lv['color'].lstrip('#')
                rv, gv, bv = (int(hx[0:2], 16),
                              int(hx[2:4], 16),
                              int(hx[4:6], 16))
                lv_mask = cm_display & (ps >= lv['pr'])
                r_arr[lv_mask] = rv
                g_arr[lv_mask] = gv
                b_arr[lv_mask] = bv
                assigned[lv_mask] = True

            # 범례 최솟값 미만 커버 → 마지막 색상으로 표시
            last_lv = levels[-1]
            hx      = last_lv['color'].lstrip('#')
            rv, gv, bv = int(hx[0:2],16), int(hx[2:4],16), int(hx[4:6],16)
            below_mask = cm_display & ~assigned
            r_arr[below_mask] = rv
            g_arr[below_mask] = gv
            b_arr[below_mask] = bv
            assigned[below_mask] = True

            ps_clipped = np.clip(ps, pr_min - 30.0, pr_max_lv)
            denom      = max(pr_max_lv - (pr_min - 30.0), 1.0)
            alpha = np.where(
                assigned,
                (0.40 + 0.35 * np.clip(
                    (ps_clipped - (pr_min - 30.0)) / denom, 0, 1)) * 255,
                0
            ).astype(np.uint8)

            rgba[..., 0] = r_arr
            rgba[..., 1] = g_arr
            rgba[..., 2] = b_arr
            rgba[..., 3] = alpha

        else:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.colors as mc

            pr_min        = float(min_rx)
            pr_max_actual = float(np.nanmax(np.where(boundary_mask, ps, np.nan)))
            vmin, vmax    = pr_min, max(pr_max_actual, pr_min + 1.0)
            cm_display    = cm & boundary_mask

            cmap   = plt.colormaps['jet']
            norm   = mc.Normalize(vmin=vmin, vmax=vmax, clip=True)
            rgba_f = cmap(norm(ps)).astype(float)
            pn     = np.clip((ps - vmin) / (vmax - vmin), 0, 1)
            rgba_f[..., 3] = np.where(cm_display, 0.45 + 0.35 * pn, 0.0)
            rgba = (rgba_f * 255).astype(np.uint8)

        img = Image.fromarray(rgba[::-1, :, :], 'RGBA')
        w, h_img = img.size
        img = img.resize((w * upscale, h_img * upscale), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, 'PNG', optimize=True)
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    @staticmethod
    def _render_heatmap_gw_colors(ps, cm, boundary_mask, min_rx,
                                   gw_idx_2d, gws, gw_color_map,
                                   color_levels=None, upscale=8):
        """GW별 고유 색상으로 합성 히트맵 렌더링."""
        import io, base64
        import numpy as np
        from PIL import Image

        GW_HEX_COLORS = [
            '#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22',
            '#c0392b', '#2980b9', '#27ae60', '#8e44ad', '#17a589',
            '#e91e8c', '#5dade2', '#58d68d', '#f0e68c', '#2c3e50',
        ]

        pr_min = float(color_levels[-1]['pr']) \
                 if color_levels else float(min_rx)

        rows, cols = ps.shape
        rgba       = np.zeros((rows, cols, 4), dtype=np.uint8)
        cm_display = cm & boundary_mask

        for gi, gw in enumerate(gws):
            gw_mask = cm_display & (gw_idx_2d == gi)
            if not gw_mask.any():
                continue

            cs    = gw.callsign
            hex_c = gw_color_map.get(cs, GW_HEX_COLORS[gi % len(GW_HEX_COLORS)])
            hx    = hex_c.lstrip('#')
            rv    = int(hx[0:2], 16)
            gv    = int(hx[2:4], 16)
            bv    = int(hx[4:6], 16)

            ps_in_mask = ps[gw_mask]
            pr_range   = max(float(ps_in_mask.max()) - pr_min, 1.0)
            alpha_full = (0.45 + 0.40 * np.clip(
                (ps - pr_min) / pr_range, 0, 1)) * 255
            alpha_full = alpha_full.astype(np.uint8)

            rgba[gw_mask, 0] = rv
            rgba[gw_mask, 1] = gv
            rgba[gw_mask, 2] = bv
            rgba[gw_mask, 3] = alpha_full[gw_mask]

        img = Image.fromarray(rgba[::-1, :, :], 'RGBA')
        w, h_img = img.size
        img = img.resize((w * upscale, h_img * upscale), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, 'PNG', optimize=True)
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"