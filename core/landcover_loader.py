# core/landcover_loader.py
"""
토지피복(클러터) 데이터 로더 — 추후 데이터 확보 시 구현
────────────────────────────────────────────────────────────
[현재 상태]
  토지피복 데이터를 아직 확보하지 못한 상태. 이 파일은 나중에
  실제 데이터(SHP, GeoTIFF, 환경부 토지피복지도 등 무엇이든)를
  구했을 때, 그 데이터를 core/dem_loader.py의
  SpatialData.set_landcover()가 요구하는 표준 형태로 변환하는
  자리입니다.

[표준 형태] SpatialData.set_landcover(lc_raster, lc_map)
  - lc_raster : DEM과 동일 shape(dem_rows × dem_cols)의 2D ndarray.
                각 픽셀에 토지피복 코드값(int)이 들어있어야 함.
  - lc_map    : {토지피복코드(int): env코드(1~4)} 딕�셔너리.
                1=Dense Urban, 2=Urban, 3=Suburban, 4=Open

[사용 흐름 — 데이터 확보 후]
  1. 아래 load_landcover_*() 함수 중 데이터 형식에 맞는 것을 고르거나
     새로 작성. (현재는 빈 스텁 + 가이드 주석만 있음)
  2. main.py 또는 MainWindow에서 SpatialData.load() 직후 호출:
        from core.landcover_loader import load_landcover_shp
        lc_raster, lc_map = load_landcover_shp(path, spatial)
        spatial.set_landcover(lc_raster, lc_map)
  3. 이후 core/propagation.py, core/coverage.py 등은 전혀 수정할
     필요 없음 — get_env_code()가 자동으로 토지피복을 우선 사용함.
────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import numpy as np


# ══════════════════════════════════════════════════════════════
# 공통 헬퍼: 임의 좌표계/해상도의 데이터를 DEM 격자에 맞춰 정렬
# ══════════════════════════════════════════════════════════════

def resample_to_dem_grid(src_array: np.ndarray,
                         src_transform,
                         src_crs: str,
                         spatial) -> np.ndarray:
    """
    임의 해상도/좌표계의 래스터(src_array)를 DEM 격자
    (spatial.dem_rows × spatial.dem_cols, spatial.dem_crs)에
    맞춰 nearest-neighbor 리샘플링합니다.

    토지피복 데이터를 구한 뒤, 좌표계나 해상도가 DEM과 다를 때
    이 함수로 먼저 정렬해야 set_landcover()에 바로 넘길 수 있습니다.

    Parameters
    ----------
    src_array     : 원본 래스터 2D ndarray (토지피복 코드값)
    src_transform : 원본 래스터의 rasterio Affine transform
    src_crs       : 원본 래스터의 좌표계 (예: "EPSG:5179")
    spatial       : SpatialData 인스턴스 (DEM 격자 정보 제공)

    Returns
    -------
    DEM과 동일 shape의 2D ndarray (정렬된 토지피복 코드값)
    """
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import Affine

    dst_array = np.zeros(
        (spatial.dem_rows, spatial.dem_cols), dtype=src_array.dtype)

    dst_transform = Affine(
        spatial.res, 0.0, spatial.ox,
        0.0, -spatial.res, spatial.oy)

    reproject(
        source=src_array,
        destination=dst_array,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=spatial.dem_crs,
        resampling=Resampling.nearest,
    )
    return dst_array


# ══════════════════════════════════════════════════════════════
# 데이터 형식별 로더 — 실제 데이터 확보 후 구현
# ══════════════════════════════════════════════════════════════

def load_landcover_raster(tif_path: str, spatial) -> tuple[np.ndarray, dict]:
    """
    [스텁] GeoTIFF/IMG 등 래스터 형태의 토지피복 데이터를 로드합니다.
    예: 환경부 토지피복지도(래스터 배포본), 위성영상 분류 결과 등.

    구현 시 참고:
      - rasterio.open(tif_path)로 읽고, resample_to_dem_grid()로
        DEM 격자에 맞춰 정렬
      - lc_map은 실제 데이터의 코드 체계를 확인한 뒤 작성
        (예: 환경부 대분류 코드 100=주거, 200=공업, 300=상업 등)

    raise NotImplementedError로 남겨둠 — 데이터 확보 후 구현 필요.
    """
    raise NotImplementedError(
        "토지피복 래스터 데이터를 아직 확보하지 못했습니다. "
        "데이터 확보 후 이 함수를 구현하세요.")


def load_landcover_vector(shp_path: str, spatial,
                          attr_field: str = None) -> tuple[np.ndarray, dict]:
    """
    [스텁] SHP/GeoJSON 등 폴리곤 벡터 형태의 토지피복 데이터를
    로드합니다. 예: 건물 통합정보, 용도지역 폴리곤 등.

    구현 시 참고:
      - geopandas로 읽고, rasterio.features.rasterize()로
        DEM 격자 해상도에 맞춰 래스터화
      - attr_field: 폴리곤의 속성 컬럼 중 분류 코드로 쓸 컬럼명
        (예: '용도지역', '건물높이' 등 — 실제 데이터 확인 후 결정)

    raise NotImplementedError로 남겨둠 — 데이터 확보 후 구현 필요.
    """
    raise NotImplementedError(
        "토지피복 벡터 데이터를 아직 확보하지 못했습니다. "
        "데이터 확보 후 이 함수를 구현하세요.")