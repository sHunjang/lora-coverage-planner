# core/dem_loader.py
"""
DEM + SHP 공간 데이터 로더
────────────────────────────────────────────────────────────
[역할]
  - 성남시 경계 SHP 파일 로드 (EPSG:3857 / 4326)
  - DEM (.img) 로드 및 성남시 경계로 마스킹
  - 좌표 변환 헬퍼 (4326 ↔ 3857)
  - 고도 조회 (벡터화 지원)
  - LOS(Line of Sight) 판별

[DEM 파일]
  dem_build_seongnam_3857-2.img
  - 포맷  : HFA (Erdas Imagine)
  - CRS   : EPSG:3857
  - 해상도: 10m/pixel
  - NoData: -9999
────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping
from pyproj import Transformer


class SpatialData:
    """
    성남시 공간 데이터 (DEM + SHP) 로드 및 보관.

    Parameters
    ----------
    shp_path : 성남시 경계 Shapefile 경로
    dem_path : DEM (.img) 파일 경로
    """

    def __init__(self, shp_path: str, dem_path: str):
        self.shp_path = shp_path
        self.dem_path = dem_path

        # SHP 관련
        self.gdf_3857     = None   # GeoDataFrame (EPSG:3857)
        self.gdf_4326     = None   # GeoDataFrame (EPSG:4326)
        self.polygon_3857 = None   # 성남시 경계 폴리곤 (3857)
        self.polygon_4326 = None   # 성남시 경계 폴리곤 (4326)
        self.bounds       = None   # [lon_min, lat_min, lon_max, lat_max]

        # DEM 관련
        self.dem          = None   # 2D float32 배열 (마스킹 완료)
        self.dem_transform = None  # rasterio Affine transform
        self.dem_rows     = 0
        self.dem_cols     = 0
        self.res          = 10.0  # 픽셀 해상도 (m)
        self.ox           = 0.0   # DEM 원점 X (EPSG:3857)
        self.oy           = 0.0   # DEM 원점 Y (EPSG:3857)

        # 좌표 변환기
        self._to_3857 = Transformer.from_crs(
            "EPSG:4326", "EPSG:3857", always_xy=True)
        self._to_4326 = Transformer.from_crs(
            "EPSG:3857", "EPSG:4326", always_xy=True)

    # ── 데이터 로드 ──────────────────────────────────────────
    def load(self, progress_cb=None):
        """
        SHP + DEM 파일 로드.

        Parameters
        ----------
        progress_cb : 진행 메시지를 받을 콜백 함수 (str → None)
        """
        def _log(msg: str):
            if progress_cb:
                progress_cb(msg)
            else:
                print(msg)

        # ── SHP 로드 ─────────────────────────────────────────
        _log("SHP 로드 중...")
        # pyogrio를 우선 엔진으로 사용 (PyInstaller 번들 환경 호환성)
        try:
            self.gdf_3857 = gpd.read_file(self.shp_path, engine='pyogrio')
        except Exception:
            self.gdf_3857 = gpd.read_file(self.shp_path)
        self.polygon_3857 = self.gdf_3857.geometry.iloc[0]
        self.gdf_4326     = self.gdf_3857.to_crs(epsg=4326)
        self.polygon_4326 = self.gdf_4326.geometry.iloc[0]
        self.bounds       = self.gdf_4326.total_bounds  # [xmin,ymin,xmax,ymax]
        _log(f"  경계 로드 완료: {self.polygon_4326.bounds}")

        # ── DEM 로드 + 마스킹 ────────────────────────────────
        _log("DEM 로드 및 마스킹 중...")
        with rasterio.open(self.dem_path) as src:
            out_image, self.dem_transform = rio_mask(
                src,
                [mapping(self.polygon_3857)],
                crop=True,
                nodata=np.nan,
            )
            raw = out_image[0].astype(np.float32)

        # NoData(-9999) 및 음수 고도 처리
        raw[raw <= -9998] = np.nan
        raw[raw < 0]      = np.nan
        self.dem = raw

        self.dem_rows, self.dem_cols = self.dem.shape
        self.res = self.dem_transform.a          # x방향 해상도 (m)
        self.ox  = self.dem_transform.c          # 좌상단 X
        self.oy  = self.dem_transform.f          # 좌상단 Y

        valid_px = int(np.sum(~np.isnan(self.dem)))
        _log(f"  DEM 로드 완료: {self.dem_rows}×{self.dem_cols}px "
             f"| 유효픽셀 {valid_px:,} "
             f"| 고도 {np.nanmin(self.dem):.0f}~{np.nanmax(self.dem):.0f} m")
        _log("공간 데이터 로드 완료")

    # ── 고도 조회 (단일 포인트) ──────────────────────────────
    def get_elevation(self, x3857: float, y3857: float) -> float:
        """
        EPSG:3857 좌표의 DEM 고도 반환.
        NaN 픽셀이면 50.0m 폴백.
        """
        col = int(np.clip((x3857 - self.ox) / self.res, 0, self.dem_cols - 1))
        row = int(np.clip((self.oy - y3857) / self.res, 0, self.dem_rows - 1))
        val = self.dem[row, col]
        return float(val) if not np.isnan(val) else 50.0

    # ── 고도 조회 (벡터화) ───────────────────────────────────
    def get_elevation_batch(self,
                            x3857: np.ndarray,
                            y3857: np.ndarray) -> np.ndarray:
        """
        EPSG:3857 좌표 배열의 DEM 고도를 한 번에 반환 (numpy 벡터화).

        Parameters
        ----------
        x3857, y3857 : ndarray (N,) — 3857 좌표 배열

        Returns
        -------
        ndarray (N,) — 고도값 (NaN → 50.0m)
        """
        cols = np.clip(
            ((x3857 - self.ox) / self.res).astype(int),
            0, self.dem_cols - 1)
        rows = np.clip(
            ((self.oy - y3857) / self.res).astype(int),
            0, self.dem_rows - 1)
        vals = self.dem[rows, cols]
        return np.where(np.isnan(vals), 50.0, vals)

    # ── LOS 판별 ─────────────────────────────────────────────
    def check_los(self,
                  x1: float, y1: float, h1: float,
                  x2: float, y2: float, h2: float) -> bool:
        """
        두 점 사이의 LOS(Line of Sight) 여부 판별.

        지형 단면을 따라 시선이 지형보다 높으면 LOS=True.

        Parameters
        ----------
        x1,y1 : 송신점 3857 좌표
        h1    : 송신 안테나 지상 높이 (m)
        x2,y2 : 수신점 3857 좌표
        h2    : 수신 안테나 지상 높이 (m)
        """
        c1 = int(np.clip((x1 - self.ox) / self.res, 0, self.dem_cols - 1))
        r1 = int(np.clip((self.oy - y1) / self.res, 0, self.dem_rows - 1))
        c2 = int(np.clip((x2 - self.ox) / self.res, 0, self.dem_cols - 1))
        r2 = int(np.clip((self.oy - y2) / self.res, 0, self.dem_rows - 1))

        n  = max(abs(c2 - c1), abs(r2 - r1)) + 1
        cs = np.clip(np.linspace(c1, c2, n).astype(int), 0, self.dem_cols - 1)
        rs = np.clip(np.linspace(r1, r2, n).astype(int), 0, self.dem_rows - 1)

        terrain = np.where(np.isnan(self.dem[rs, cs]), 0.0, self.dem[rs, cs])
        e1      = self.get_elevation(x1, y1) + h1
        e2      = self.get_elevation(x2, y2) + h2
        sight   = np.linspace(e1, e2, n)
        return bool(np.all(sight >= terrain))

    # ── 좌표 변환 헬퍼 ───────────────────────────────────────
    def lonlat_to_xy(self, lon, lat):
        """위경도(4326) → 3857 변환. 배열 입력 지원."""
        return self._to_3857.transform(lon, lat)

    def xy_to_lonlat(self, x, y):
        """3857 → 위경도(4326) 변환. 배열 입력 지원."""
        return self._to_4326.transform(x, y)