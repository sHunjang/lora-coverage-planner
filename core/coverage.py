# core/coverage.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from concurrent.futures import ThreadPoolExecutor


@dataclass
class GWEntry:
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
    covered   : bool  = False
    best_gw   : str   = ""
    best_pr   : float = -999.0
    gw_prs    : dict  = field(default_factory=dict)
    macro_pr  : float = -999.0   # MRC 합성 수신 전력
    n_rx_gw   : int   = 0        # 수신 가능 GW 수


@dataclass
class CoverageResult:
    nodes                : list  = field(default_factory=list)
    gw_counts            : dict  = field(default_factory=dict)
    n_covered            : int   = 0
    n_total              : int   = 0
    macro_diversity_gain : float = 0.0
    avg_n_rx_gw          : float = 0.0
    adr_sf_distribution  : dict  = field(default_factory=dict)
    avg_toa_ms           : float = 0.0

    @property
    def coverage_pct(self):
        return self.n_covered / self.n_total * 100 if self.n_total else 0


class CoverageEngine:
    def __init__(self, spatial, env=2, fc=915.0, n_samples=100, settings=None):
        self.spatial   = spatial
        self.env       = env
        self.fc        = fc
        self.n_samples = n_samples
        self.settings  = settings or {}

    def _model(self, hb, hm):
        from core.propagation import PathLossModel
        return PathLossModel(self.spatial, h_station=hm,
                             hb_gw=hb, env=self.env,
                             fc=self.fc, n_samples=self.n_samples)

    def run(self, gws, nodes, cb=None):
        def _log(m):
            if cb: cb(m)

        SF_SENS = {
            7: -123.0, 8: -126.0, 9: -129.0,
            10: -132.0, 11: -134.5, 12: -137.0,
        }
        # SF별 ToA (ms), SF7~SF12, BW=125kHz, CR=4/5, payload=20byte 기준
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

        gw_xy = {g.callsign: self.spatial.lonlat_to_xy(g.lon, g.lat)
                for g in active}
        nd_xy = [self.spatial.lonlat_to_xy(n.lon, n.lat) for n in nodes]

        _log(f"분석 시작: GW {len(active)}개 × Node {len(nodes)}개")

        macro_gains = []
        n_rx_gw_list = []
        adr_sf_dist = {sf: 0 for sf in range(7, 13)}
        toa_list = []

        for ni, nd in enumerate(nodes):
            nx, ny  = float(nd_xy[ni][0]), float(nd_xy[ni][1])
            gw_prs  = {}
            best_pr = -999.0
            best_gw = ""

            for gw in active:
                gx, gy = float(gw_xy[gw.callsign][0]), float(gw_xy[gw.callsign][1])
                model  = self._model(gw.hb_m, nd.hm_m)
                pl     = model.path_loss(gx, gy, nx, ny)
                # 실내 투과 손실 적용
                indoor = getattr(nd, 'indoor_loss_db', 0.0)
                pr     = gw.pt_dbm + gw.gt_dbi - gw.lt_db - pl + nd.gr_dbi - nd.lr_db - indoor
                gw_prs[gw.callsign] = round(float(pr), 1)
                if pr > best_pr:
                    best_pr, best_gw = pr, gw.callsign

            # 수신 가능 GW 목록 (min_rx 이상)
            rx_gws = [cs for cs, pr in gw_prs.items() if pr >= nd.min_rx_dbm]
            n_rx   = len(rx_gws)

            # MRC (Maximum Ratio Combining) — 선형 합산 후 dBm 변환
            if n_rx >= 2:
                linear_sum = sum(10 ** (gw_prs[cs] / 10) for cs in rx_gws)
                macro_pr   = 10 * np.log10(linear_sum)
                gain       = macro_pr - best_pr
            else:
                macro_pr = best_pr
                gain     = 0.0

            macro_gains.append(gain)
            n_rx_gw_list.append(n_rx)

            cov = best_pr >= nd.min_rx_dbm

            # ADR: 커버된 Node의 최적 SF 결정
            if cov:
                adr_sf = 12  # 기본값 (가장 낮은 데이터레이트)
                for sf in sorted(SF_SENS.keys()):
                    if best_pr >= SF_SENS[sf]:
                        adr_sf = sf
                        break
                adr_sf_dist[adr_sf] = adr_sf_dist.get(adr_sf, 0) + 1
                toa_list.append(SF_TOA[adr_sf])

            result.nodes.append(LinkResult(
                covered  = cov,
                best_gw  = best_gw,
                best_pr  = round(best_pr, 1),
                gw_prs   = gw_prs,
                macro_pr = round(macro_pr, 1),
                n_rx_gw  = n_rx,
            ))
            if cov:
                result.n_covered += 1
                result.gw_counts[best_gw] = result.gw_counts.get(best_gw, 0) + 1

            if (ni + 1) % max(1, len(nodes) // 10) == 0:
                _log(f"  {ni+1}/{len(nodes)} ({(ni+1)/len(nodes)*100:.0f}%)")

        # 통계 집계
        result.macro_diversity_gain = float(np.mean(macro_gains)) if macro_gains else 0.0
        result.avg_n_rx_gw          = float(np.mean(n_rx_gw_list)) if n_rx_gw_list else 0.0
        result.adr_sf_distribution  = adr_sf_dist
        result.avg_toa_ms           = float(np.mean(toa_list)) if toa_list else 0.0

        _log(f"완료: {result.n_covered}/{result.n_total}개 ({result.coverage_pct:.1f}%)")
        return result

    def heatmap(self, gw, min_rx, step=0.0015, cb=None,
            use_deygout=False, radius_km=12.0,
            pr_min=None, pr_max=None):
        """
        PDF 명세 기반 히트맵 계산.
        - LOS: Song's Model (변곡점 거리 내 장애물 시 +20dB)
        - NLOS: max(Song's, PL_FS + Deygout)
        GW 중심 radius_km 반경 + 성남시 경계 교집합으로 계산 범위 제한.
        pr_min: 범례 최솟값 미만 픽셀 투명 처리
        """
        import base64, io
        from pyproj import Transformer
        from scipy.ndimage import gaussian_filter, label as nd_label

        b = self.spatial.bounds

        deg_lat = radius_km / 111.0
        deg_lon = radius_km / (111.0 * np.cos(np.radians(gw.lat)))

        lmin   = max(b[0], gw.lon - deg_lon)
        latmin = max(b[1], gw.lat - deg_lat)
        lmax   = min(b[2], gw.lon + deg_lon)
        latmax = min(b[3], gw.lat + deg_lat)

        lons = np.arange(lmin, lmax, step)
        lats = np.arange(latmin, latmax, step)
        lon2d, lat2d = np.meshgrid(lons, lats)
        fl = lon2d.ravel(); fa = lat2d.ravel()

        try:
            from shapely import points as _sp, contains as _sc
            mask_poly = _sc(self.spatial.polygon_4326,
                            _sp(np.stack([fl, fa], axis=1)))
        except Exception:
            from shapely.geometry import Point
            mask_poly = np.array([self.spatial.polygon_4326.contains(
                Point(lo, la)) for lo, la in zip(fl, fa)])

        dist_deg = np.sqrt(((fl - gw.lon) / deg_lon)**2 +
                        ((fa - gw.lat) / deg_lat)**2) * radius_km
        mask_circle = dist_deg <= radius_km
        mask = mask_poly & mask_circle

        tr = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
        px, py     = tr.transform(fl, fa)
        gx_arr, gy_arr = tr.transform(gw.lon, gw.lat)
        gx, gy     = float(gx_arr), float(gy_arr)

        # ── PathLossModel 사용 (propagation.py와 동일한 로직) ──
        from core.propagation import PathLossModel
        model = PathLossModel(
            self.spatial,
            h_station  = 1.5,
            hb_gw      = float(gw.hb_m),
            env        = self.env,
            fc         = self.fc,
            n_samples  = min(50, self.n_samples),
            diff_order = 2,
        )
        eirp = float(gw.pt_dbm + gw.gt_dbi - gw.lt_db)

        idx    = np.where(mask)[0]
        pf     = np.full(len(px), float(min_rx) - 50.0)
        px_f   = px.astype(np.float64)
        py_f   = py.astype(np.float64)

        if cb: cb(f"히트맵 계산 중... ({len(idx)}개 격자점)")

        for k, i in enumerate(idx):
            pl = model.path_loss(gx, gy, float(px_f[i]), float(py_f[i]))
            pf[i] = eirp - pl

            if cb and (k + 1) % max(1, len(idx) // 10) == 0:
                cb(f"  {k+1}/{len(idx)} ({(k+1)/len(idx)*100:.0f}%)")

        pg            = pf.reshape(lon2d.shape)
        boundary_mask = mask.reshape(lon2d.shape)
        pg_masked     = np.where(boundary_mask, pg, np.nan)

        pg_filled = np.where(np.isnan(pg_masked), float(min_rx) - 50.0, pg_masked)
        ps        = gaussian_filter(pg_filled.astype(float), sigma=0.5)
        ps        = np.where(boundary_mask, ps, float(min_rx) - 50.0)

        cov_raw    = ps >= min_rx
        labeled, n = nd_label(cov_raw)

        if n > 0:
            gw_col_g = int(np.clip((gw.lon - lmin) / step, 0, lon2d.shape[1] - 1))
            gw_row_g = int(np.clip((gw.lat - latmin) / step, 0, lon2d.shape[0] - 1))
            if labeled[gw_row_g, gw_col_g] > 0:
                main_label = labeled[gw_row_g, gw_col_g]
            else:
                r0 = max(0, gw_row_g - 10); r1 = min(lon2d.shape[0], gw_row_g + 10)
                c0 = max(0, gw_col_g - 10); c1 = min(lon2d.shape[1], gw_col_g + 10)
                region      = labeled[r0:r1, c0:c1]
                labels_near = region[region > 0]
                if len(labels_near) > 0:
                    main_label = int(np.bincount(labels_near).argmax())
                else:
                    sizes      = np.bincount(labeled.ravel())[1:]
                    main_label = int(np.argmax(sizes)) + 1
            cm = (labeled == main_label) & boundary_mask
        else:
            cm = cov_raw & boundary_mask

        color_levels = self.settings.get('color_levels') if hasattr(self, 'settings') else None
        url = self._render_heatmap_image(ps, cm, boundary_mask, min_rx, color_levels)

        if cb: cb("히트맵 완료")
        return {
            'url'     : url,
            'bounds'  : [[latmin, lmin], [latmax, lmax]],
            'callsign': gw.callsign,
            'min_rx'  : min_rx,
            'ps'      : ps,
            'cm'      : cm,
            'lon_min' : lmin,
            'lat_min' : latmin,
            'step'    : step,
        }

    def heatmap_combined(self, gws, min_rx, step=0.0015,
                        cb=None, radius_km=12.0,
                        pr_min=None, pr_max=None):
        """
        여러 GW의 히트맵을 합성 — 각 격자점마다 최대 Pr 선택.
        겹치는 구간은 신호가 가장 강한 GW 기준으로 표시.
        pr_min: 범례 최솟값 미만 픽셀 투명 처리
        """
        import base64, io
        from pyproj import Transformer
        from scipy.ndimage import gaussian_filter, label as nd_label

        b = self.spatial.bounds

        deg_lat  = radius_km / 111.0
        mean_lat = float(np.mean([g.lat for g in gws]))
        deg_lon  = radius_km / (111.0 * np.cos(np.radians(mean_lat)))

        lmin   = max(b[0], min(g.lon for g in gws) - deg_lon)
        latmin = max(b[1], min(g.lat for g in gws) - deg_lat)
        lmax   = min(b[2], max(g.lon for g in gws) + deg_lon)
        latmax = min(b[3], max(g.lat for g in gws) + deg_lat)

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
        px, py = tr.transform(fl, fa)
        px_f   = px.astype(np.float64)
        py_f   = py.astype(np.float64)

        pr_max_grid = np.full(len(px), float(min_rx) - 50.0)

        from core.propagation import PathLossModel

        for gi, gw in enumerate(gws):
            if cb: cb(f"히트맵 계산 중... GW {gi+1}/{len(gws)}: {gw.callsign}")

            gx_arr, gy_arr = tr.transform(gw.lon, gw.lat)
            gx, gy   = float(gx_arr), float(gy_arr)
            eirp     = float(gw.pt_dbm + gw.gt_dbi - gw.lt_db)

            # GW별 PathLossModel (hb_gw 반영)
            model = PathLossModel(
                self.spatial,
                h_station  = 1.5,
                hb_gw      = float(gw.hb_m),
                env        = self.env,
                fc         = self.fc,
                n_samples  = min(50, self.n_samples),
                diff_order = 2,
            )

            # GW 중심 반경 마스크
            deg_lon_gw = radius_km / (111.0 * np.cos(np.radians(gw.lat)))
            deg_lat_gw = radius_km / 111.0
            dist_deg   = np.sqrt(((fl - gw.lon) / deg_lon_gw)**2 +
                                ((fa - gw.lat) / deg_lat_gw)**2) * radius_km
            gw_mask    = mask & (dist_deg <= radius_km)
            idx        = np.where(gw_mask)[0]

            for k, i in enumerate(idx):
                pl = model.path_loss(gx, gy, float(px_f[i]), float(py_f[i]))
                pr = eirp - pl
                if pr > pr_max_grid[i]:
                    pr_max_grid[i] = pr

        pg            = pr_max_grid.reshape(lon2d.shape)
        boundary_mask = mask.reshape(lon2d.shape)
        pg_masked     = np.where(boundary_mask, pg, np.nan)

        pg_filled = np.where(np.isnan(pg_masked), float(min_rx) - 50.0, pg_masked)
        ps        = gaussian_filter(pg_filled.astype(float), sigma=0.5)
        ps        = np.where(boundary_mask, ps, float(min_rx) - 50.0)

        cm = (ps >= min_rx) & boundary_mask

        color_levels = self.settings.get('color_levels') if hasattr(self, 'settings') else None
        url = self._render_heatmap_image(ps, cm, boundary_mask, min_rx, color_levels)

        if cb: cb(f"합성 히트맵 완료 ({len(gws)}개 GW)")
        return {
            'url'     : url,
            'bounds'  : [[latmin, lmin], [latmax, lmax]],
            'callsign': 'COMBINED',
            'type'    : 'combined',
            'min_rx'  : min_rx,
            'ps'      : ps,
            'cm'      : cm,
            'lon_min' : lmin,
            'lat_min' : latmin,
            'step'    : step,
        }

    def env_map(self, step=0.003, cb=None):
        """DSM 기반 전파 환경 분류 지도 생성."""
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
            import shapely
            from shapely import points as _sp, contains as _sc
            mask = _sc(self.spatial.polygon_4326,
                       _sp(np.stack([fl, fa], axis=1)))
        except Exception:
            from shapely.geometry import Point
            mask = np.array([self.spatial.polygon_4326.contains(
                Point(lo, la)) for lo, la in zip(fl, fa)])

        tr = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
        px, py = tr.transform(fl, fa)

        idx = np.where(mask)[0]
        env_grid = np.zeros(len(px), dtype=np.uint8)

        if cb: cb(f"환경 분류 계산 중... ({len(idx)}개 포인트)")

        for i in idx:
            env_grid[i] = self.spatial.get_env_code(
                float(px[i]), float(py[i]))

        eg = env_grid.reshape(lon2d.shape)
        boundary_mask = mask.reshape(lon2d.shape)

        ENV_COLORS = {
            1: (220, 50,  50,  160),
            2: (230, 140, 30,  140),
            3: (220, 210, 40,  120),
            4: (50,  180, 80,  100),
        }
        rgba = np.zeros((*lon2d.shape, 4), dtype=np.uint8)
        for code, color in ENV_COLORS.items():
            where = (eg == code) & boundary_mask
            rgba[where] = color

        img = Image.fromarray(rgba[::-1, :, :], 'RGBA')
        w, h = img.size
        img  = img.resize((w*2, h*2), Image.NEAREST)
        buf  = io.BytesIO()
        img.save(buf, 'PNG', optimize=True)
        url  = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

        if cb: cb("환경 분류 시각화 완료")
        return {
            'url'     : url,
            'bounds'  : [[latmin, lmin], [latmax, lmax]],
            'callsign': 'ENV_MAP',
            'type'    : 'env_map',
        }

    @staticmethod
    def _render_heatmap_image(ps, cm, boundary_mask, min_rx, color_levels=None):
        """
        ps            : 수신 전력 2D 배열
        cm            : 커버리지 마스크 (bool 2D)
        boundary_mask : 성남시 경계 마스크 (bool 2D)
        min_rx        : 최소 수신 레벨 (dBm)
        color_levels  : [{'pr': float, 'color': '#RRGGBB'}, ...] 내림차순 정렬
        """
        import io, base64
        from PIL import Image
        import numpy as np

        rows, cols = ps.shape
        rgba = np.zeros((rows, cols, 4), dtype=np.uint8)

        if color_levels:
            # 내림차순 정렬 보장
            levels = sorted(color_levels, key=lambda x: -x['pr'])
            pr_min = float(levels[-1]['pr'])  # 가장 낮은 레벨

            # cm_display: 커버리지 마스크 & pr_min 이상인 픽셀만 표시
            cm_display = cm & boundary_mask & (ps >= pr_min)

            for row in range(rows):
                for col in range(cols):
                    if not cm_display[row, col]:
                        continue
                    pv = ps[row, col]
                    # 해당 픽셀의 색상 결정 (가장 높은 임계값부터 비교)
                    chosen_color = None
                    for lv in levels:
                        if pv >= lv['pr']:
                            chosen_color = lv['color']
                            break
                    if chosen_color is None:
                        continue
                    # hex → RGB
                    hx = chosen_color.lstrip('#')
                    r, g, b = int(hx[0:2],16), int(hx[2:4],16), int(hx[4:6],16)
                    # 알파: 신호 강도에 따라 0.5~0.85
                    pr_max_lv = float(levels[0]['pr'])
                    alpha = 0.5 + 0.35 * np.clip((pv - pr_min) / max(pr_max_lv - pr_min, 1), 0, 1)
                    rgba[row, col] = [r, g, b, int(alpha * 255)]
        else:
            # color_levels 없으면 기존 jet 방식 fallback
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.colors as mc
            pr_min = float(min_rx)
            pr_max_actual = float(np.nanmax(np.where(boundary_mask, ps, np.nan)))
            vmin, vmax = pr_min, max(pr_max_actual, pr_min + 1.0)
            cm_display = cm & boundary_mask & (ps >= vmin)
            cmap = plt.colormaps['jet']
            norm = mc.Normalize(vmin=vmin, vmax=vmax, clip=True)
            rgba_f = cmap(norm(ps)).astype(float)
            pn = np.clip((ps - vmin) / (vmax - vmin), 0, 1)
            rgba_f[..., 3] = np.where(cm_display, 0.45 + 0.35 * pn, 0.0)
            rgba = (rgba_f * 255).astype(np.uint8)

        img = Image.fromarray(rgba[::-1, :, :], 'RGBA')
        w, h_img = img.size
        img = img.resize((w * 2, h_img * 2), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, 'PNG', optimize=True)
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"