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
    callsign  : str   = "Node1"
    lon       : float = 127.10
    lat       : float = 37.40
    gr_dbi    : float = 2.15
    lr_db     : float = 0.0
    hm_m      : float = 1.5
    min_rx_dbm: float = -126.6

@dataclass
class LinkResult:
    covered   : bool  = False
    best_gw   : str   = ""
    best_pr   : float = -999.0
    gw_prs    : dict  = field(default_factory=dict)

@dataclass
class CoverageResult:
    nodes     : list = field(default_factory=list)
    gw_counts : dict = field(default_factory=dict)
    n_covered : int  = 0
    n_total   : int  = 0

    @property
    def coverage_pct(self):
        return self.n_covered / self.n_total * 100 if self.n_total else 0


class CoverageEngine:
    def __init__(self, spatial, env=2, fc=915.0, n_samples=100):
        self.spatial   = spatial
        self.env       = env
        self.fc        = fc
        self.n_samples = n_samples

    def _model(self, hb, hm):
        from core.propagation import PathLossModel
        return PathLossModel(self.spatial, h_station=hm,
                             hb_gw=hb, env=self.env,
                             fc=self.fc, n_samples=self.n_samples)

    def run(self, gws, nodes, cb=None):
        def _log(m):
            if cb: cb(m)

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

        for ni, nd in enumerate(nodes):
            nx, ny  = float(nd_xy[ni][0]), float(nd_xy[ni][1])
            gw_prs  = {}
            best_pr = -999.0
            best_gw = ""

            for gw in active:
                gx, gy = float(gw_xy[gw.callsign][0]), float(gw_xy[gw.callsign][1])
                model  = self._model(gw.hb_m, nd.hm_m)
                pl     = model.path_loss(gx, gy, nx, ny)
                pr     = gw.pt_dbm + gw.gt_dbi - gw.lt_db - pl + nd.gr_dbi - nd.lr_db
                gw_prs[gw.callsign] = round(float(pr), 1)
                if pr > best_pr:
                    best_pr, best_gw = pr, gw.callsign

            cov = best_pr >= nd.min_rx_dbm
            result.nodes.append(LinkResult(
                covered=cov, best_gw=best_gw,
                best_pr=round(best_pr, 1), gw_prs=gw_prs))
            if cov:
                result.n_covered += 1
                result.gw_counts[best_gw] = result.gw_counts.get(best_gw, 0) + 1

            if (ni + 1) % max(1, len(nodes) // 10) == 0:
                _log(f"  {ni+1}/{len(nodes)} ({(ni+1)/len(nodes)*100:.0f}%)")

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
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt, matplotlib.colors as mc
        from PIL import Image
        from pyproj import Transformer
        from scipy.ndimage import gaussian_filter, label as nd_label
        from core.propagation import SongsModel, DeygoutDiff

        b = self.spatial.bounds

        # GW 중심 반경으로 계산 범위 제한 + 성남시 경계 교집합
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
            import shapely
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
        px, py   = tr.transform(fl, fa)
        gx_arr, gy_arr = tr.transform(gw.lon, gw.lat)
        gx, gy   = float(gx_arr), float(gy_arr)

        sp           = self.spatial
        dem          = sp.dem
        ox, oy, res  = sp.ox, sp.oy, sp.res
        rows_, cols_ = sp.dem_rows, sp.dem_cols
        h_tx         = float(gw.hb_m)
        h_rx         = 1.5
        N_SAMP       = min(50, self.n_samples)

        songs   = SongsModel(fc=self.fc, hb=h_tx, hm=h_rx, env=self.env)
        deygout = DeygoutDiff(fc=self.fc, max_order=2)
        eirp    = float(gw.pt_dbm + gw.gt_dbi - gw.lt_db)

        gw_col  = int(np.clip((gx - ox) / res, 0, cols_ - 1))
        gw_row  = int(np.clip((oy - gy) / res, 0, rows_ - 1))
        gw_elev = dem[gw_row, gw_col]
        if np.isnan(gw_elev): gw_elev = 0.0
        gw_abs_h = gw_elev + h_tx

        lam     = 3e8 / (self.fc * 1e6)
        r_inf_m = np.sqrt(4 * h_tx * h_rx / lam)

        idx = np.where(mask)[0]
        pf  = np.full(len(px), float(min_rx) - 50.0)

        if cb: cb(f"히트맵 계산 중... ({len(idx)}개 격자점)")

        px_m = px[idx].astype(np.float64)
        py_m = py[idx].astype(np.float64)
        dx   = px_m - gx
        dy   = py_m - gy
        d_m  = np.maximum(np.hypot(dx, dy), 1.0)
        d_km = d_m / 1000.0

        pl_songs_arr = (39.25
                        + 35.15 * np.log10(self.fc)
                        - 19.21 * np.log10(h_tx)
                        + (42.5 - 5.2 * np.log10(h_tx)) * np.log10(d_km)
                        - songs.ahm)

        pr_vals = np.full(len(idx), float(min_rx) - 50.0)

        for k in range(len(idx)):
            x2, y2 = px_m[k], py_m[k]
            dm      = d_m[k]

            xs    = np.linspace(gx, x2, N_SAMP)
            ys    = np.linspace(gy, y2, N_SAMP)
            cs    = np.clip(((xs - ox) / res).astype(int), 0, cols_ - 1)
            rs    = np.clip(((oy - ys) / res).astype(int), 0, rows_ - 1)
            elevs = dem[rs, cs]
            elevs = np.where(np.isnan(elevs), 0.0, elevs)
            dists = np.linspace(0.0, dm, N_SAMP)

            rx_elev = elevs[-1] + h_rx
            sight   = np.linspace(gw_abs_h, rx_elev, N_SAMP)
            nlos    = bool(np.any(elevs > sight))

            if not nlos:
                pl = pl_songs_arr[k]
                if dm <= r_inf_m:
                    max_obs = float(np.max(elevs[1:-1])) if len(elevs) > 2 else 0.0
                    if max_obs > gw_abs_h:
                        pl += 20.0
            else:
                e_tx   = elevs[0]  + h_tx
                e_rx   = elevs[-1] + h_rx
                sight2 = np.linspace(e_tx, e_rx, N_SAMP)
                n_obs  = int(np.sum(elevs > sight2))

                if n_obs > max(3 * (N_SAMP // 50), 8):
                    d_km_v = max(dm / 1000.0, 0.001)
                    pl_fs  = (20.0 * np.log10(self.fc)
                              + 20.0 * np.log10(d_km_v) - 27.5492)
                    pl = pl_fs + 200.0
                else:
                    ld_t = deygout._deygout_recursive(
                        dists, elevs, 0, len(dists)-1,
                        h_tx, h_rx, deygout.max_order)
                    ld_t  = max(0.0, ld_t)
                    d_km_v = max(dm / 1000.0, 0.001)
                    pl_fs  = (20.0 * np.log10(self.fc)
                              + 20.0 * np.log10(d_km_v) - 27.5492)
                    dl = pl_fs + ld_t
                    pl = max(pl_songs_arr[k], dl)

            pr_vals[k] = eirp - pl

        pf[idx] = pr_vals

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

        pr_max_actual = float(np.nanmax(pg_masked)) if boundary_mask.any() else float(min_rx)
        vmin = float(pr_min) if pr_min is not None else float(min_rx)
        vmax = float(pr_max) if pr_max is not None else max(pr_max_actual, vmin + 1.0)

        cmap = plt.colormaps['jet']
        norm = mc.Normalize(vmin=vmin, vmax=vmax, clip=True)
        rgba = cmap(norm(ps)).astype(float)
        pn   = np.clip((ps - vmin) / (vmax - vmin), 0, 1)

        # pr_min 미만 픽셀 투명 처리
        cm_display = cm & (ps >= vmin)
        rgba[..., 3] = np.where(cm_display, 0.45 + 0.35 * pn, 0.0)

        img      = Image.fromarray((rgba[::-1, :, :] * 255).astype(np.uint8), 'RGBA')
        w, h_img = img.size
        img      = img.resize((w * 2, h_img * 2), Image.BILINEAR)
        buf      = io.BytesIO()
        img.save(buf, 'PNG', optimize=True)
        url      = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

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
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt, matplotlib.colors as mc
        from PIL import Image
        from pyproj import Transformer
        from scipy.ndimage import gaussian_filter, label as nd_label
        from core.propagation import SongsModel, DeygoutDiff

        b = self.spatial.bounds

        deg_lat   = radius_km / 111.0
        mean_lat  = float(np.mean([g.lat for g in gws]))
        deg_lon   = radius_km / (111.0 * np.cos(np.radians(mean_lat)))

        lmin   = max(b[0], min(g.lon for g in gws) - deg_lon)
        latmin = max(b[1], min(g.lat for g in gws) - deg_lat)
        lmax   = min(b[2], max(g.lon for g in gws) + deg_lon)
        latmax = min(b[3], max(g.lat for g in gws) + deg_lat)

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

        sp           = self.spatial
        dem          = sp.dem
        ox, oy, res  = sp.ox, sp.oy, sp.res
        rows_, cols_ = sp.dem_rows, sp.dem_cols
        N_SAMP       = min(50, self.n_samples)
        lam          = 3e8 / (self.fc * 1e6)

        pr_max_grid = np.full(len(px), float(min_rx) - 50.0)

        for gi, gw in enumerate(gws):
            if cb: cb(f"히트맵 계산 중... GW {gi+1}/{len(gws)}: {gw.callsign}")

            gx_arr, gy_arr = tr.transform(gw.lon, gw.lat)
            gx, gy   = float(gx_arr), float(gy_arr)
            h_tx     = float(gw.hb_m)
            h_rx     = 1.5
            eirp     = float(gw.pt_dbm + gw.gt_dbi - gw.lt_db)
            r_inf_m  = np.sqrt(4 * h_tx * h_rx / lam)

            songs   = SongsModel(fc=self.fc, hb=h_tx, hm=h_rx, env=self.env)
            deygout = DeygoutDiff(fc=self.fc, max_order=2)

            gw_col   = int(np.clip((gx - ox) / res, 0, cols_ - 1))
            gw_row   = int(np.clip((oy - gy) / res, 0, rows_ - 1))
            gw_elev  = dem[gw_row, gw_col]
            if np.isnan(gw_elev): gw_elev = 0.0
            gw_abs_h = gw_elev + h_tx

            deg_lon_gw = radius_km / (111.0 * np.cos(np.radians(gw.lat)))
            deg_lat_gw = radius_km / 111.0
            dist_deg   = np.sqrt(((fl - gw.lon) / deg_lon_gw)**2 +
                                 ((fa - gw.lat) / deg_lat_gw)**2) * radius_km
            gw_mask    = mask & (dist_deg <= radius_km)
            idx        = np.where(gw_mask)[0]

            px_m = px[idx].astype(np.float64)
            py_m = py[idx].astype(np.float64)
            dx   = px_m - gx
            dy   = py_m - gy
            d_m  = np.maximum(np.hypot(dx, dy), 1.0)
            d_km = d_m / 1000.0

            pl_songs_arr = (39.25
                            + 35.15 * np.log10(self.fc)
                            - 19.21 * np.log10(h_tx)
                            + (42.5 - 5.2 * np.log10(h_tx)) * np.log10(d_km)
                            - songs.ahm)

            for k in range(len(idx)):
                x2, y2 = px_m[k], py_m[k]
                dm      = d_m[k]

                xs    = np.linspace(gx, x2, N_SAMP)
                ys    = np.linspace(gy, y2, N_SAMP)
                cs    = np.clip(((xs - ox) / res).astype(int), 0, cols_ - 1)
                rs    = np.clip(((oy - ys) / res).astype(int), 0, rows_ - 1)
                elevs = dem[rs, cs]
                elevs = np.where(np.isnan(elevs), 0.0, elevs)
                dists = np.linspace(0.0, dm, N_SAMP)

                rx_elev = elevs[-1] + h_rx
                sight   = np.linspace(gw_abs_h, rx_elev, N_SAMP)
                nlos    = bool(np.any(elevs > sight))

                if not nlos:
                    pl = pl_songs_arr[k]
                    if dm <= r_inf_m:
                        max_obs = float(np.max(elevs[1:-1])) if len(elevs) > 2 else 0.0
                        if max_obs > gw_abs_h:
                            pl += 20.0
                else:
                    e_tx   = elevs[0]  + h_tx
                    e_rx   = elevs[-1] + h_rx
                    sight2 = np.linspace(e_tx, e_rx, N_SAMP)
                    n_obs  = int(np.sum(elevs > sight2))

                    if n_obs > max(3 * (N_SAMP // 50), 8):
                        d_km_v = max(dm / 1000.0, 0.001)
                        pl_fs  = (20.0 * np.log10(self.fc)
                                  + 20.0 * np.log10(d_km_v) - 27.5492)
                        pl = pl_fs + 200.0
                    else:
                        ld_t = deygout._deygout_recursive(
                            dists, elevs, 0, len(dists)-1,
                            h_tx, h_rx, deygout.max_order)
                        ld_t  = max(0.0, ld_t)
                        d_km_v = max(dm / 1000.0, 0.001)
                        pl_fs  = (20.0 * np.log10(self.fc)
                                  + 20.0 * np.log10(d_km_v) - 27.5492)
                        pl = max(pl_songs_arr[k], pl_fs + ld_t)

                pr = eirp - pl
                if pr > pr_max_grid[idx[k]]:
                    pr_max_grid[idx[k]] = pr

        pg            = pr_max_grid.reshape(lon2d.shape)
        boundary_mask = mask.reshape(lon2d.shape)
        pg_masked     = np.where(boundary_mask, pg, np.nan)

        pg_filled = np.where(np.isnan(pg_masked), float(min_rx) - 50.0, pg_masked)
        ps        = gaussian_filter(pg_filled.astype(float), sigma=0.5)
        ps        = np.where(boundary_mask, ps, float(min_rx) - 50.0)

        cm = (ps >= min_rx) & boundary_mask

        pr_max_actual = float(np.nanmax(pg_masked)) if boundary_mask.any() else float(min_rx)
        vmin = float(pr_min) if pr_min is not None else float(min_rx)
        vmax = float(pr_max) if pr_max is not None else max(pr_max_actual, vmin + 1.0)

        cmap = plt.colormaps['jet']
        norm = mc.Normalize(vmin=vmin, vmax=vmax, clip=True)
        rgba = cmap(norm(ps)).astype(float)
        pn   = np.clip((ps - vmin) / (vmax - vmin), 0, 1)

        # pr_min 미만 픽셀 투명 처리
        cm_display = cm & (ps >= vmin)
        rgba[..., 3] = np.where(cm_display, 0.45 + 0.35 * pn, 0.0)

        img      = Image.fromarray((rgba[::-1, :, :] * 255).astype(np.uint8), 'RGBA')
        w, h_img = img.size
        img      = img.resize((w * 2, h_img * 2), Image.BILINEAR)
        buf      = io.BytesIO()
        img.save(buf, 'PNG', optimize=True)
        url      = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

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