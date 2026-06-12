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
        self.gdf_3857     = None
        self.gdf_4326     = None
        self.polygon_3857 = None
        self.polygon_4326 = None
        self.bounds       = None

        # DEM 관련
        self.dem           = None
        self.dem_transform = None
        self.dem_rows      = 0
        self.dem_cols      = 0
        self.res           = 10.0
        self.ox            = 0.0
        self.oy            = 0.0

        # 좌표 변환기
        # DEM 좌표계 (load() 후 확정)
        self.dem_crs  = "EPSG:3857"
        # 좌표 변환기 (load()에서 DEM CRS 확정 후 생성)
        self._to_proj = None
        self._to_4326 = None
        # 토지피복 래스터 (set_landcover()로 설정, 없으면 DSM 추정 방식 사용)
        self._lc_raster = None   # 2D ndarray, DEM과 동일 shape
        self._lc_map    = {}     # {토지피복코드: env코드(1~4)}

    # ── 데이터 로드 ──────────────────────────────────────────
    # 변경 후
    def load(self, progress_cb=None):
        def _log(msg: str):
            if progress_cb:
                progress_cb(msg)
            else:
                print(msg)

        _log("SHP 로드 중...")
        try:
            gdf_raw = gpd.read_file(self.shp_path, engine='pyogrio')
        except Exception:
            gdf_raw = gpd.read_file(self.shp_path)

        if gdf_raw.crs is None:
            _log("  ⚠ SHP 좌표계(.prj) 없음 — EPSG:3857 가정")
            gdf_raw = gdf_raw.set_crs(epsg=3857)

        self.gdf_4326     = gdf_raw.to_crs(epsg=4326)
        self.polygon_4326 = self.gdf_4326.geometry.iloc[0]
        self.bounds       = self.gdf_4326.total_bounds
        _log(f"  경계 로드 완료: {self.polygon_4326.bounds}")

        _log("DEM 로드 및 마스킹 중...")
        with rasterio.open(self.dem_path) as src:
            # DEM 좌표계 자동 감지
            if src.crs is not None:
                self.dem_crs = src.crs.to_string()
            else:
                _log("  ⚠ DEM 좌표계 없음 — EPSG:3857 가정")
                self.dem_crs = "EPSG:3857"
            _log(f"  DEM 좌표계: {self.dem_crs}")

            # NoData 자동 감지
            src_nodata = src.nodata if src.nodata is not None else -9999.0

            # 폴리곤을 DEM 좌표계로 변환 후 마스킹
            gdf_dem           = gdf_raw.to_crs(self.dem_crs)
            self.gdf_3857     = gdf_dem          # 변수명 호환 유지
            self.polygon_3857 = gdf_dem.geometry.iloc[0]

            out_image, self.dem_transform = rio_mask(
                src,
                [mapping(self.polygon_3857)],
                crop=True,
                nodata=np.nan,
            )
            raw = out_image[0].astype(np.float32)

        # NoData 처리 (메타데이터 값 + 일반 음수)
        raw[raw == src_nodata] = np.nan
        raw[raw <= -9998]      = np.nan
        raw[raw < 0]           = np.nan
        self.dem = raw

        self.dem_rows, self.dem_cols = self.dem.shape
        self.res = self.dem_transform.a
        self.ox  = self.dem_transform.c
        self.oy  = self.dem_transform.f

        # 좌표 변환기 — DEM CRS 기반으로 생성
        self._to_proj = Transformer.from_crs(
            "EPSG:4326", self.dem_crs, always_xy=True)
        self._to_4326 = Transformer.from_crs(
            self.dem_crs, "EPSG:4326", always_xy=True)

        # 고도 분포 기반 환경 분류 기준 자동 조정
        valid = self.dem[~np.isnan(self.dem)]
        if len(valid) > 0:
            self._elev_p10  = float(np.percentile(valid, 10))
            self._elev_p50  = float(np.percentile(valid, 50))
            self._elev_p90  = float(np.percentile(valid, 90))
            # 산지 판별 기준: 지역 고도 분포에서 동적 계산
            # p90(상위10%) 기준으로 산지 임계값 결정
            self._mountain_thresh = max(
                self._elev_p10 + 20.0,
                self._elev_p50 * 0.8)
            _log(f"  고도 분포: "
                 f"p10={self._elev_p10:.0f}m "
                 f"p50={self._elev_p50:.0f}m "
                 f"p90={self._elev_p90:.0f}m "
                 f"| 산지 기준 >{self._mountain_thresh:.0f}m")
        else:
            self._elev_p10 = self._elev_p50 = self._elev_p90 = 50.0
            self._mountain_thresh = 60.0

        valid_px = int(np.sum(~np.isnan(self.dem)))
        _log(f"  DEM 로드 완료: {self.dem_rows}×{self.dem_cols}px "
             f"| 유효픽셀 {valid_px:,} "
             f"| 고도 {np.nanmin(self.dem):.0f}~{np.nanmax(self.dem):.0f} m")
        _log("공간 데이터 로드 완료")

    # ── 고도 조회 (단일 포인트) ──────────────────────────────
    def get_elevation(self, x3857: float, y3857: float) -> float:
        col = int(np.clip((x3857 - self.ox) / self.res, 0, self.dem_cols - 1))
        row = int(np.clip((self.oy - y3857) / self.res, 0, self.dem_rows - 1))
        val = self.dem[row, col]
        return float(val) if not np.isnan(val) else 50.0

    # ── 고도 조회 (벡터화) ───────────────────────────────────
    def get_elevation_batch(self,
                            x3857: np.ndarray,
                            y3857: np.ndarray) -> np.ndarray:
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
        return self._to_proj.transform(lon, lat)

    def xy_to_lonlat(self, x, y):
        return self._to_4326.transform(x, y)

    # ── 환경 자동 분류 (클러터) ──────────────────────────────
    def get_env_code(self, x3857: float, y3857: float,
                     radius_px: int = 10) -> int:
        """
        DSM 기반 주변 환경 자동 분류.
        반경 radius_px 픽셀(기본 100m) 내 고도 통계로 env 코드 결정.

        [분류 로직]
        1. 산지 판별 먼저 수행
           - 지형 기준선(하위 10%) 자체가 높고(>60m) 경사(std)가 크면 → Open
           - 산지는 DSM 자체가 높아 build_h/std가 커서 Dense Urban으로
             오분류되는 문제를 방지
        2. 건물 돌출 높이(build_h)와 표준편차(std)로 도시 밀도 분류
           - Dense Urban : 고층 건물 밀집 (아파트 단지 등)
           - Urban       : 일반 도심 (상업/업무 지역)
           - Suburban    : 저층 주거지역
           - Open        : 도로 / 공원 / 하천 / 개활지 / 산지

        Returns: 1=Dense Urban, 2=Urban, 3=Suburban, 4=Open
        """
        
        # ── 토지피복 래스터 우선 조회 ─────────────────────────
        if self._lc_raster is not None:
            col = int(np.clip((x3857 - self.ox) / self.res,
                              0, self.dem_cols - 1))
            row = int(np.clip((self.oy - y3857) / self.res,
                              0, self.dem_rows - 1))
            lc_code = int(self._lc_raster[row, col])
            if lc_code in self._lc_map:
                return self._lc_map[lc_code]
            # 매핑에 없는 코드면 DSM 추정으로 폴백

        col = int(np.clip((x3857 - self.ox) / self.res,
                          radius_px, self.dem_cols - radius_px - 1))
        row = int(np.clip((self.oy - y3857) / self.res,
                          radius_px, self.dem_rows - radius_px - 1))

        patch = self.dem[row - radius_px : row + radius_px,
                         col - radius_px : col + radius_px]
        valid = patch[~np.isnan(patch)]
        if len(valid) == 0:
            return 2  # 기본 Urban

        mean_h  = float(np.mean(valid))
        std_h   = float(np.std(valid))
        # 지형 기준선: 하위 10% 평균 (건물이 없는 지면 고도 추정)
        terrain = float(np.percentile(valid, 10))
        # 건물/구조물 돌출 높이 = 평균 DSM - 지형 기준선
        build_h = mean_h - terrain

        # ── 1단계: 산지/자연 지형 판별 ───────────────────────
        # 성남시 기준: 평지 고도 30~50m, 산지 60m 이상
        # 산지 특징: terrain 자체가 높고 경사(std)가 크다
        # 건물 특징: terrain은 낮고(평지) build_h와 std가 크다
        mt = getattr(self, '_mountain_thresh', 60.0)
        is_mountain = (terrain > mt and std_h > 8.0)
        if is_mountain:
            return 4  # Open — 산지/공원/자연 지형

        # ── 2단계: 건물 밀도 분류 ─────────────────────────────
        # terrain < 60 조건으로 산지 오분류 추가 방어
        if   build_h >= 18 and std_h >= 12 and terrain < 60:
            return 1  # Dense Urban (고층 아파트/빌딩 밀집)
        elif build_h >= 8  and std_h >= 5  and terrain < 80:
            return 2  # Urban (일반 도심/상업지역)
        elif build_h >= 3  and std_h >= 2:
            return 3  # Suburban (저층 주거지역)
        else:
            return 4  # Open (도로/공원/하천/개활지)

    def set_landcover(self, lc_raster: np.ndarray, lc_map: dict):
        """
        토지피복 래스터를 등록합니다. 등록 후 get_env_code()에서
        DSM 추정 대신 토지피복 코드를 우선 사용합니다.

        Parameters
        ----------
        lc_raster : DEM과 동일 shape의 2D 배열 (토지피복 코드값)
        lc_map    : {토지피복코드(int): env코드(1~4)}
                    예: {110: 1, 120: 2, 400: 4, 500: 4}
        """
        if lc_raster.shape != self.dem.shape:
            raise ValueError(
                f"토지피복 래스터 크기 {lc_raster.shape}가 "
                f"DEM 크기 {self.dem.shape}와 다릅니다.")
        self._lc_raster = lc_raster
        self._lc_map    = lc_map
        print(f"[토지피복] 등록 완료: {len(lc_map)}개 코드 매핑")