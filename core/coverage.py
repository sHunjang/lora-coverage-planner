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
    covered      : bool  = False
    best_gw      : str   = ""
    best_pr      : float = -999.0
    gw_prs       : dict  = field(default_factory=dict)
    macro_pr     : float = -999.0
    n_rx_gw      : int   = 0
    adr_sf       : int   = 0
    adr_dr       : int   = 0
    adr_toa_ms   : float = 0.0

@dataclass
class CoverageResult:
    nodes     : list = field(default_factory=list)
    gw_counts : dict = field(default_factory=dict)
    n_covered : int  = 0
    n_total   : int  = 0

    @property
    def coverage_pct(self):
        return self.n_covered / self.n_total * 100 if self.n_total else 0

    @property
    def macro_diversity_gain(self) -> float:
        gains = [n.macro_pr - n.best_pr
                 for n in self.nodes
                 if n.best_pr > -999 and n.n_rx_gw > 1]
        return float(np.mean(gains)) if gains else 0.0

    @property
    def avg_n_rx_gw(self) -> float:
        counts = [n.n_rx_gw for n in self.nodes]
        return float(np.mean(counts)) if counts else 0.0

    @property
    def adr_sf_distribution(self) -> dict:
        dist = {sf: 0 for sf in range(7, 13)}
        for nd in self.nodes:
            if nd.adr_sf > 0:
                dist[nd.adr_sf] = dist.get(nd.adr_sf, 0) + 1
        return dist

    @property
    def avg_toa_ms(self) -> float:
        toas = [nd.adr_toa_ms for nd in self.nodes if nd.adr_toa_ms > 0]
        return float(np.mean(toas)) if toas else 0.0


# SF별 감도/DR/ToA 상수
_SF_SENS  = {12:-137.0, 11:-134.5, 10:-132.0,
              9:-129.0,   8:-126.0,  7:-123.0}
_SF_TO_DR = {12:0, 11:1, 10:2, 9:3, 8:4, 7:5}
_SF_TOA   = {12:1482.8, 11:741.4, 10:370.7,
              9:185.3,   8:102.4,  7:61.7}
_ADR_MARGIN_DB = 15.0


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
                pr     = (gw.pt_dbm + gw.gt_dbi - gw.lt_db
                          - pl + nd.gr_dbi - nd.lr_db
                          - nd.indoor_loss_db)
                gw_prs[gw.callsign] = round(float(pr), 1)
                if pr > best_pr:
                    best_pr, best_gw = pr, gw.callsign

            cov = best_pr >= nd.min_rx_dbm

            # 매크로 다이버시티
            rx_gws   = [pr for pr in gw_prs.values() if pr >= nd.min_rx_dbm]
            n_rx     = len(rx_gws)
            macro_pr = (float(10 * np.log10(sum(10**(p/10) for p in rx_gws)))
                        if n_rx > 0 else best_pr)

            # ADR SF 결정 (마진 15dB 기준)
            adr_sf = 0
            for sf in sorted(_SF_SENS.keys(), reverse=True):
                if macro_pr >= _SF_SENS[sf] + _ADR_MARGIN_DB:
                    adr_sf = sf
                    break
            if adr_sf == 0 and macro_pr >= _SF_SENS[12]:
                adr_sf = 12

            result.nodes.append(LinkResult(
                covered=cov, best_gw=best_gw,
                best_pr=round(best_pr, 1), gw_prs=gw_prs,
                macro_pr=round(macro_pr, 1), n_rx_gw=n_rx,
                adr_sf=adr_sf,
                adr_dr=_SF_TO_DR.get(adr_sf, 0),
                adr_toa_ms=_SF_TOA.get(adr_sf, 0.0),
            ))
            if cov:
                result.n_covered += 1
                result.gw_counts[best_gw] = result.gw_counts.get(best_gw, 0) + 1

            if (ni + 1) % max(1, len(nodes) // 10) == 0:
                _log(f"  {ni+1}/{len(nodes)} ({(ni+1)/len(nodes)*100:.0f}%)")

        _log(f"완료: {result.n_covered}/{result.n_total}개 ({result.coverage_pct:.1f}%)")
        return result

    def heatmap(self, gw, min_rx, step=0.001, cb=None, use_deygout=False):
        import base64, io
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt, matplotlib.colors as mc
        from PIL import Image
        from pyproj import Transformer
        from scipy.ndimage import gaussian_filter, label as nd_label

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
        gx, gy = tr.transform(gw.lon, gw.lat)
        gx, gy = float(gx), float(gy)

        from core.propagation import SongsModel, DeygoutDiff

        idx = np.where(mask)[0]
        pf  = np.full(len(px), float(min_rx) - 50.0)

        if len(idx) > 0:
            px_m = px[idx].astype(np.float64)
            py_m = py[idx].astype(np.float64)

            dx = px_m - gx
            dy = py_m - gy
            d_m  = np.maximum(np.hypot(dx, dy), 1.0)
            d_km = d_m / 1000.0

            songs = SongsModel(fc=self.fc, hb=float(gw.hb_m),
                               hm=1.5, env=self.env)
            pl_songs = (39.25
                        + 35.15 * np.log10(self.fc)
                        - 19.21 * np.log10(gw.hb_m)
                        + (42.5 - 5.2 * np.log10(gw.hb_m)) * np.log10(d_km)
                        - songs.ahm)

            if use_deygout:
                N_DIFF  = min(30, self.n_samples)
                deygout = DeygoutDiff(fc=self.fc, max_order=1)
                sp      = self.spatial
                dem     = sp.dem
                ox, oy, res = sp.ox, sp.oy, sp.res
                rows_, cols_ = sp.dem_rows, sp.dem_cols
                h_tx, h_rx  = float(gw.hb_m), 1.5
                l_diff = np.zeros(len(idx), dtype=np.float64)
                for k in range(len(idx)):
                    x2, y2 = px_m[k], py_m[k]
                    xs = np.linspace(gx, x2, N_DIFF)
                    ys = np.linspace(gy, y2, N_DIFF)
                    cs = np.clip(((xs-ox)/res).astype(int), 0, cols_-1)
                    rs = np.clip(((oy-ys)/res).astype(int), 0, rows_-1)
                    elevs = dem[rs, cs]
                    elevs = np.where(np.isnan(elevs), 50.0, elevs)
                    dists = np.linspace(0, d_m[k], N_DIFF)
                    l_diff[k] = deygout.diffraction_loss(dists, elevs, h_tx, h_rx)
            else:
                l_diff = np.zeros(len(idx), dtype=np.float64)

            pl_total = pl_songs + l_diff
            eirp     = float(gw.pt_dbm + gw.gt_dbi - gw.lt_db)
            pf[idx]  = eirp - pl_total

        pg = pf.reshape(lon2d.shape)
        boundary_mask = mask.reshape(lon2d.shape)
        pg_masked = np.where(boundary_mask, pg, np.nan)

        pg_filled = np.where(np.isnan(pg_masked), float(min_rx)-50.0, pg_masked)
        ps = gaussian_filter(pg_filled.astype(float), sigma=1.5)
        ps = np.where(boundary_mask, ps, float(min_rx)-50.0)

        cov_raw = ps >= min_rx
        labeled, n = nd_label(cov_raw)

        if n > 0:
            gw_col = int(np.clip((gw.lon-lmin)/step, 0, lon2d.shape[1]-1))
            gw_row = int(np.clip((gw.lat-latmin)/step, 0, lon2d.shape[0]-1))
            if labeled[gw_row, gw_col] > 0:
                main_label = labeled[gw_row, gw_col]
            else:
                r0 = max(0, gw_row-10); r1 = min(lon2d.shape[0], gw_row+10)
                c0 = max(0, gw_col-10); c1 = min(lon2d.shape[1], gw_col+10)
                region = labeled[r0:r1, c0:c1]
                labels_near = region[region > 0]
                if len(labels_near) > 0:
                    main_label = int(np.bincount(labels_near).argmax())
                else:
                    sizes = np.bincount(labeled.ravel())[1:]
                    main_label = int(np.argmax(sizes)) + 1
            cm = (labeled == main_label) & boundary_mask
        else:
            cm = cov_raw & boundary_mask

        pr_max_actual = float(np.nanmax(pg_masked)) if boundary_mask.any() else float(min_rx)
        vmin = float(min_rx)
        vmax = max(pr_max_actual, vmin + 1.0)

        cmap = plt.colormaps['jet']
        norm = mc.Normalize(vmin=vmin, vmax=vmax, clip=True)
        rgba = cmap(norm(ps)).astype(float)
        pn   = np.clip((ps - vmin) / (vmax - vmin), 0, 1)
        rgba[..., 3] = np.where(cm, 0.45 + 0.35*pn, 0.0)

        img = Image.fromarray((rgba[::-1,:,:]*255).astype(np.uint8), 'RGBA')
        w, h = img.size
        img  = img.resize((w*2, h*2), Image.BILINEAR)
        buf  = io.BytesIO()
        img.save(buf, 'PNG', optimize=True)
        url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

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

        # 1=Dense Urban(빨강), 2=Urban(주황), 3=Suburban(노랑), 4=Open(초록)
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

        img = Image.fromarray(rgba[::-1,:,:], 'RGBA')
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