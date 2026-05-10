# core/coverage.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


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
    macro_pr  : float = -999.0
    n_rx_gw   : int   = 0

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
        prop_model = self.settings.get('prop_model', 'smartcity') \
                    if hasattr(self, 'settings') else 'smartcity'
        return PathLossModel(
            self.spatial,
            h_station  = hm,
            hb_gw      = hb,
            env        = self.env,
            fc         = self.fc,
            n_samples  = self.n_samples,
            prop_model = prop_model,
        )

    def run(self, gws, nodes, cb=None):
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

        gw_xy = {g.callsign: self.spatial.lonlat_to_xy(g.lon, g.lat)
                for g in active}
        nd_xy = [self.spatial.lonlat_to_xy(n.lon, n.lat) for n in nodes]

        _log(f"분석 시작: GW {len(active)}개 × Node {len(nodes)}개")

        macro_gains  = []
        n_rx_gw_list = []
        adr_sf_dist  = {sf: 0 for sf in range(7, 13)}
        toa_list     = []

        for ni, nd in enumerate(nodes):
            nx, ny  = float(nd_xy[ni][0]), float(nd_xy[ni][1])
            gw_prs  = {}
            best_pr = -999.0
            best_gw = ""

            for gw in active:
                gx, gy = float(gw_xy[gw.callsign][0]), float(gw_xy[gw.callsign][1])
                model  = self._model(gw.hb_m, nd.hm_m)
                pl     = model.path_loss(gx, gy, nx, ny)
                indoor = getattr(nd, 'indoor_loss_db', 0.0)
                pr     = (gw.pt_dbm + gw.gt_dbi - gw.lt_db
                          - pl + nd.gr_dbi - nd.lr_db - indoor)
                gw_prs[gw.callsign] = round(float(pr), 1)
                if pr > best_pr:
                    best_pr, best_gw = pr, gw.callsign

            rx_gws = [cs for cs, pr in gw_prs.items() if pr >= nd.min_rx_dbm]
            n_rx   = len(rx_gws)

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

            if cov:
                adr_sf = 12
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

        result.macro_diversity_gain = float(np.mean(macro_gains)) if macro_gains else 0.0
        result.avg_n_rx_gw          = float(np.mean(n_rx_gw_list)) if n_rx_gw_list else 0.0
        result.adr_sf_distribution  = adr_sf_dist
        result.avg_toa_ms           = float(np.mean(toa_list)) if toa_list else 0.0

        _log(f"완료: {result.n_covered}/{result.n_total}개 ({result.coverage_pct:.1f}%)")
        return result

    def heatmap(self, gw, min_rx, step=0.0015, cb=None,
                use_deygout=False, radius_km=12.0,
                pr_min=None, pr_max=None):
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

        dist_deg    = np.sqrt(((fl - gw.lon) / deg_lon)**2 +
                              ((fa - gw.lat) / deg_lat)**2) * radius_km
        mask_circle = dist_deg <= radius_km
        mask        = mask_poly & mask_circle

        tr = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
        px, py         = tr.transform(fl, fa)
        gx_arr, gy_arr = tr.transform(gw.lon, gw.lat)
        gx, gy         = float(gx_arr), float(gy_arr)

        # ── 히트맵 전용 경량 모델 ────────────────────────────
        N_SAMP_HM = max(30, min(50, self.n_samples // 2))  # ← 10~20 → 30~50
        from core.propagation import PathLossModel
        model = PathLossModel(
            self.spatial,
            h_station  = 1.5,
            hb_gw      = float(gw.hb_m),
            env        = self.env,
            fc         = self.fc,
            n_samples  = N_SAMP_HM,
            diff_order = 2,            # ← 1 → 2 복원
            prop_model = self.settings.get('prop_model', 'smartcity'),
        )
        eirp = float(gw.pt_dbm + gw.gt_dbi - gw.lt_db)
        idx  = np.where(mask)[0]
        pf   = np.full(len(px), float(min_rx) - 50.0)

        if cb: cb(f"히트맵 계산 중... ({len(idx):,}개 격자점)")

        px_idx  = px.astype(np.float64)[idx]
        py_idx  = py.astype(np.float64)[idx]
        n_workers = min(os.cpu_count() or 4, 16)

        def _calc(k):
            pl = model.path_loss(gx, gy,
                                 float(px_idx[k]),
                                 float(py_idx[k]))
            return k, eirp - pl

        completed = [0]
        results   = np.full(len(idx), float(min_rx) - 50.0)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_calc, k): k
                       for k in range(len(idx))}
            for fut in as_completed(futures):
                k, pr = fut.result()
                results[k] = pr
                completed[0] += 1
                if cb and completed[0] % max(1, len(idx) // 20) == 0:
                    pct = completed[0] / len(idx) * 100
                    cb(f"  {completed[0]:,}/{len(idx):,} ({pct:.0f}%)")

        pf[idx] = results

        pg            = pf.reshape(lon2d.shape)
        boundary_mask = mask.reshape(lon2d.shape)
        pg_masked     = np.where(boundary_mask, pg, np.nan)

        pg_filled = np.where(np.isnan(pg_masked), float(min_rx) - 50.0, pg_masked)
        ps        = gaussian_filter(pg_filled.astype(float), sigma=1.5)
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

        color_levels = self.settings.get('color_levels') \
                       if hasattr(self, 'settings') else None
        url = self._render_heatmap_image(ps, cm, boundary_mask, min_rx, color_levels)

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
            'step'    : step,
        }

    def heatmap_combined(self, gws, min_rx, step=0.0015,
                         cb=None, radius_km=12.0,
                         pr_min=None, pr_max=None):
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
        gw_idx_grid = np.full(len(px), -1, dtype=np.int32)

        n_workers = min(os.cpu_count() or 4, 16)
        from core.propagation import PathLossModel

        for gi, gw in enumerate(gws):
            if cb: cb(f"히트맵 계산 중... GW {gi+1}/{len(gws)}: {gw.callsign}")

            gx_arr, gy_arr = tr.transform(gw.lon, gw.lat)
            gx, gy   = float(gx_arr), float(gy_arr)
            eirp     = float(gw.pt_dbm + gw.gt_dbi - gw.lt_db)

            # ── 히트맵 전용 경량 모델 ────────────────────────
            N_SAMP_HM = max(30, min(50, self.n_samples // 2))  # ← 수정
            model = PathLossModel(
                self.spatial,
                h_station  = 1.5,
                hb_gw      = float(gw.hb_m),
                env        = self.env,
                fc         = self.fc,
                n_samples  = N_SAMP_HM,
                diff_order = 2,        # ← 수정
                prop_model = self.settings.get('prop_model', 'smartcity'),
            )

            deg_lon_gw = radius_km / (111.0 * np.cos(np.radians(gw.lat)))
            deg_lat_gw = radius_km / 111.0
            dist_deg   = np.sqrt(((fl - gw.lon) / deg_lon_gw)**2 +
                                 ((fa - gw.lat) / deg_lat_gw)**2) * radius_km
            gw_mask    = mask & (dist_deg <= radius_km)
            idx        = np.where(gw_mask)[0]

            px_idx = px_f[idx]
            py_idx = py_f[idx]

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

        pg_filled = np.where(np.isnan(pg_masked), float(min_rx) - 50.0, pg_masked)
        ps        = gaussian_filter(pg_filled.astype(float), sigma=1.5)
        ps        = np.where(boundary_mask, ps, float(min_rx) - 50.0)

        cm        = (ps >= min_rx) & boundary_mask
        gw_idx_2d = gw_idx_grid.reshape(lon2d.shape)

        color_levels = self.settings.get('color_levels') \
                       if hasattr(self, 'settings') else None
        gw_color_map = self.settings.get('gw_color_map', {}) \
                       if hasattr(self, 'settings') else {}

        url = self._render_heatmap_gw_colors(
            ps, cm, boundary_mask, min_rx,
            gw_idx_2d, gws, gw_color_map, color_levels)

        if cb: cb(f"합성 히트맵 완료 ({len(gws)}개 GW)")
        return {
            'url'     : url,
            'bounds'  : [[float(latmin), float(lmin)],
                         [float(latmax), float(lmax)]],
            'callsign': 'COMBINED',
            'type'    : 'combined',
            'min_rx'  : min_rx,
            'ps'      : ps,
            'cm'      : cm,
            'gw_idx'  : gw_idx_2d,
            'lon_min' : float(lmin),
            'lat_min' : float(latmin),
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
            'bounds'  : [[float(latmin), float(lmin)],
                         [float(latmax), float(lmax)]],
            'callsign': 'ENV_MAP',
            'type'    : 'env_map',
        }

    @staticmethod
    def _render_heatmap_image(ps, cm, boundary_mask, min_rx, color_levels=None):
        """단일 GW 히트맵 렌더링 (color_levels 기반 계단식 색상)."""
        import io, base64
        from PIL import Image
        import numpy as np

        rows, cols = ps.shape
        rgba = np.zeros((rows, cols, 4), dtype=np.uint8)

        if color_levels:
            levels    = sorted(color_levels, key=lambda x: -x['pr'])
            pr_min    = float(levels[-1]['pr'])
            pr_max_lv = float(levels[0]['pr'])

            cm_display = cm & boundary_mask & (ps >= pr_min)

            r_arr    = np.zeros((rows, cols), dtype=np.uint8)
            g_arr    = np.zeros((rows, cols), dtype=np.uint8)
            b_arr    = np.zeros((rows, cols), dtype=np.uint8)
            assigned = np.zeros((rows, cols), dtype=bool)

            for lv in reversed(levels):
                hx = lv['color'].lstrip('#')
                rv, gv, bv = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
                lv_mask = cm_display & (ps >= lv['pr'])
                r_arr[lv_mask] = rv
                g_arr[lv_mask] = gv
                b_arr[lv_mask] = bv
                assigned[lv_mask] = True

            denom = max(pr_max_lv - pr_min, 1.0)
            alpha = np.where(
                assigned,
                (0.5 + 0.35 * np.clip((ps - pr_min) / denom, 0, 1)) * 255,
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
            cm_display    = cm & boundary_mask & (ps >= vmin)

            cmap   = plt.colormaps['jet']
            norm   = mc.Normalize(vmin=vmin, vmax=vmax, clip=True)
            rgba_f = cmap(norm(ps)).astype(float)
            pn     = np.clip((ps - vmin) / (vmax - vmin), 0, 1)
            rgba_f[..., 3] = np.where(cm_display, 0.45 + 0.35 * pn, 0.0)
            rgba = (rgba_f * 255).astype(np.uint8)

        img = Image.fromarray(rgba[::-1, :, :], 'RGBA')
        w, h_img = img.size
        img = img.resize((w * 4, h_img * 4), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, 'PNG', optimize=True)
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    @staticmethod
    def _render_heatmap_gw_colors(ps, cm, boundary_mask, min_rx,
                                   gw_idx_2d, gws, gw_color_map,
                                   color_levels=None):
        """GW별 고유 색상으로 합성 히트맵 렌더링."""
        import io, base64
        import numpy as np
        from PIL import Image

        GW_HEX_COLORS = [
            '#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22',
            '#c0392b', '#2980b9', '#27ae60', '#8e44ad', '#17a589',
            '#e91e8c', '#5dade2', '#58d68d', '#f0e68c', '#2c3e50',
        ]

        if color_levels:
            levels = sorted(color_levels, key=lambda x: -x['pr'])
            pr_min = float(levels[-1]['pr'])
        else:
            pr_min = float(min_rx)

        rows, cols = ps.shape
        rgba       = np.zeros((rows, cols, 4), dtype=np.uint8)
        cm_display = cm & boundary_mask & (ps >= pr_min)

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
        img = img.resize((w * 4, h_img * 4), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, 'PNG', optimize=True)
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"