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

    def heatmap(self, gw, min_rx, step=0.001, cb=None):
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

        # 경계 마스크
        try:
            from shapely.vectorized import contains as sc
            mask = sc(self.spatial.polygon_4326, fl, fa)
        except Exception:
            from shapely.geometry import Point
            mask = np.array([self.spatial.polygon_4326.contains(
                Point(lo, la)) for lo, la in zip(fl, fa)])

        tr = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
        px, py = tr.transform(fl, fa)

        # GW 좌표도 동일한 Transformer 사용
        gx, gy = tr.transform(gw.lon, gw.lat)
        gx, gy = float(gx), float(gy)

        model = self._model(gw.hb_m, 1.5)
        idx   = np.where(mask)[0]
        pf    = np.full(len(px), float(min_rx) - 50.0)

        def _c(i):
            pl = model.path_loss(gx, gy, float(px[i]), float(py[i]))
            return i, float(gw.pt_dbm + gw.gt_dbi - gw.lt_db - pl)

        with ThreadPoolExecutor(max_workers=8) as pool:
            for i, pr in pool.map(_c, idx):
                pf[i] = pr

        pg = pf.reshape(lon2d.shape)
        boundary_mask = mask.reshape(lon2d.shape)
        pg_masked = np.where(boundary_mask, pg, np.nan)

        # 경계 안쪽만 blur
        pg_filled = np.where(np.isnan(pg_masked), float(min_rx) - 50.0, pg_masked)
        ps = gaussian_filter(pg_filled.astype(float), sigma=1.5)
        ps = np.where(boundary_mask, ps, float(min_rx) - 50.0)

        # 커버 마스크 — cov_raw 기준으로 컴포넌트 분리
        cov_raw = ps >= min_rx
        labeled, n = nd_label(cov_raw)   # ← 수정: cov_raw 사용

        if n > 0:
            gw_col = int(np.clip((gw.lon - lmin) / step, 0, lon2d.shape[1] - 1))
            gw_row = int(np.clip((gw.lat - latmin) / step, 0, lon2d.shape[0] - 1))

            if labeled[gw_row, gw_col] > 0:
                main_label = labeled[gw_row, gw_col]
            else:
                # GW 주변 10픽셀 반경에서 레이블 찾기
                r0 = max(0, gw_row - 10); r1 = min(lon2d.shape[0], gw_row + 10)
                c0 = max(0, gw_col - 10); c1 = min(lon2d.shape[1], gw_col + 10)
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

        # 컬러맵
        pr_max_actual = float(np.nanmax(pg_masked)) if boundary_mask.any() else float(min_rx)
        vmin = float(min_rx)
        vmax = max(pr_max_actual, vmin + 1.0)

        cmap = plt.colormaps['jet']
        norm = mc.Normalize(vmin=vmin, vmax=vmax, clip=True)
        rgba = cmap(norm(ps)).astype(float)
        pn   = np.clip((ps - vmin) / (vmax - vmin), 0, 1)
        rgba[..., 3] = np.where(cm, 0.45 + 0.35 * pn, 0.0)

        img = Image.fromarray(
            (rgba[::-1, :, :] * 255).astype(np.uint8), 'RGBA')
        w, h = img.size
        img  = img.resize((w * 2, h * 2), Image.BILINEAR)
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