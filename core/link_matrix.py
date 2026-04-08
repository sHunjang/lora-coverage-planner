# core/link_matrix.py
"""
Station 간 경로 손실 행렬 계산 모듈
────────────────────────────────────────────────────────────
[역할]
  N개 station 간 N×N 경로 손실 행렬을 계산하고 CSV로 저장.

[행렬 구조]
       ST1    ST2    ST3   ...  STN
  ST1 [  0   PL12   PL13  ...  PL1N ]
  ST2 [ PL12   0    PL23  ...  PL2N ]
  ST3 [ PL13  PL23    0   ...  PL3N ]
  ...
  STN [ PL1N  PL2N  PL3N  ...    0  ]

  - 대각선 : 0 (자기 자신)
  - 링크 불가 : 0 (경로손실 > pl_limit)
  - 나머지 : Song's Model + Deygout 계산값 (dB)

[대칭성 활용]
  PL[i][j] == PL[j][i] 이므로
  실제 계산 횟수 = N×(N-1)/2 (상삼각 행렬만 계산 후 복사)

[PL_limit 자동 계산]
  MATLAB 코드와 동일한 방식:
    Margin    = 53.383*P_edge^3 - 80.075*P_edge^2 + 57.512*P_edge - 15.41
    PL_limit  = Pt_max - (sens_SF12 + Margin)

[성능]
  ThreadPoolExecutor로 행 단위 병렬 계산
  진행률 콜백으로 실시간 진행 상황 전달
────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

from .propagation import PathLossModel


# ── SF별 수신 감도 (dBm) ─────────────────────────────────────
# SF7 ~ SF12 순서
SENS_DBM = np.array([-123.0, -126.0, -129.0, -132.0, -134.5, -137.0])


# ── ProcessPoolExecutor 워커 함수 (모듈 최상위 필수) ──────────
# Windows spawn 방식에서 pickle 가능하려면 최상위 함수여야 함

_worker_model = None  # 각 워커 프로세스의 PathLossModel 인스턴스


def _worker_init(shp_path, dem_path, h_station, env, fc, n_samples):
    """워커 프로세스 초기화 — DEM을 한 번만 로드."""
    import sys, os
    # 프로젝트 루트를 sys.path에 추가 (상대 import 지원)
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)

    from core.dem_loader import SpatialData
    from core.propagation import PathLossModel as PLM

    global _worker_model
    sp = SpatialData(shp_path, dem_path)
    sp.load()
    _worker_model = PLM(sp, h_station=h_station, env=env,
                        fc=fc, n_samples=n_samples)


def _calc_row_proc(args):
    """행 i의 상삼각 경로 손실 계산 (ProcessPool 태스크)."""
    i, st_xi, st_yi, st_x_list, st_y_list, pl_limit_, N_ = args
    row = []
    for j in range(i + 1, N_):
        pl = _worker_model.path_loss(
            float(st_xi), float(st_yi),
            float(st_x_list[j]), float(st_y_list[j]))
        row.append((i, j, float(pl) if pl <= pl_limit_ else 0.0))
    return row


def calc_margin(p_edge: float) -> float:
    """
    셀 경계 커버리지 확률(p_edge) 기준 링크 마진 계산.

    MATLAB 코드와 동일:
      Margin = 53.383*P_edge^3 - 80.075*P_edge^2 + 57.512*P_edge - 15.41

    Parameters
    ----------
    p_edge : 셀 경계 커버리지 확률 (0~1), 기본 0.9

    Returns
    -------
    float : 마진 (dB)
    """
    return (53.383 * p_edge**3
            - 80.075 * p_edge**2
            + 57.512 * p_edge
            - 15.41)


def calc_pl_limit(pt_max: float, p_edge: float) -> float:
    """
    PL_limit 자동 계산.

    MATLAB 코드와 동일:
      PL_limit = Pt_max - (sens_SF12 + Margin)

    SF12(가장 감도 좋은 설정) 기준으로 계산하므로
    모든 SF를 포괄하는 최대 허용 경로 손실값.

    Parameters
    ----------
    pt_max : 송신 출력 (dBm)
    p_edge : 셀 경계 커버리지 확률

    Returns
    -------
    float : PL_limit (dB)
    """
    margin = calc_margin(p_edge)
    return pt_max - (SENS_DBM[-1] + margin)  # SENS_DBM[-1] = SF12 = -137 dBm


# ══════════════════════════════════════════════════════════════
# 파라미터 / 결과 데이터 클래스
# ══════════════════════════════════════════════════════════════

@dataclass
class LinkParams:
    """
    링크 행렬 계산 파라미터.

    PL_limit 결정 방식 (우선순위):
      1. pl_limit_override > 0 이면 해당 값을 직접 사용
      2. pl_limit_override <= 0 이면 pt_max + p_edge로 자동 계산
         PL_limit = Pt_max - (sens_SF12 + Margin)

    Parameters
    ----------
    num_stations    : station 수 (사용자 입력)
    pt_max          : 송신 출력 (dBm), 기본 14 dBm
    p_edge          : 셀 경계 커버리지 확률 (기본 0.9)
    pl_limit_override: 0 이하면 자동 계산, 양수면 직접 지정
    h_station       : station 안테나 지상 높이 (m), 기본 1.5 m
    env             : Song's Model 환경 (1=Dense Urban ~ 4=Open)
    fc              : 반송 주파수 (MHz)
    n_samples       : DEM 단면 샘플 수
    diff_order      : Deygout 재귀 깊이
    seed            : 랜덤 시드
    """
    num_stations     : int   = 100
    pt_max           : float = 14.0
    p_edge           : float = 0.9
    pl_limit_override: float = 0.0   # 0 이하 = 자동 계산
    h_station        : float = 1.5
    env              : int   = 2
    fc               : float = 915.0
    n_samples        : int   = 100
    diff_order       : int   = 2
    seed             : int   = 42

    @property
    def margin(self) -> float:
        """셀 경계 마진 (dB)."""
        return calc_margin(self.p_edge)

    @property
    def pl_limit(self) -> float:
        """
        실제 사용할 PL_limit (dB).
        pl_limit_override > 0 이면 그 값, 아니면 자동 계산.
        """
        if self.pl_limit_override > 0:
            return self.pl_limit_override
        return calc_pl_limit(self.pt_max, self.p_edge)

    def summary(self) -> str:
        """파라미터 요약 문자열."""
        mode = (f"직접 지정 ({self.pl_limit_override} dB)"
                if self.pl_limit_override > 0
                else f"자동 계산 (Pt={self.pt_max} dBm, P_edge={self.p_edge})")
        return (
            f"  Station 수      : {self.num_stations}개\n"
            f"  송신 출력 Pt    : {self.pt_max} dBm\n"
            f"  셀 경계 확률    : {self.p_edge} ({self.p_edge*100:.0f}%)\n"
            f"  Margin          : {self.margin:.4f} dB\n"
            f"  PL_limit        : {self.pl_limit:.4f} dB  [{mode}]\n"
            f"  안테나 높이     : {self.h_station} m\n"
            f"  환경            : {self.env} "
            f"({['','Dense Urban','Urban','Suburban','Open'][self.env]})\n"
            f"  주파수          : {self.fc} MHz\n"
            f"  DEM 샘플 수     : {self.n_samples}\n"
            f"  랜덤 시드       : {self.seed}"
        )


@dataclass
class LinkResult:
    """
    링크 행렬 계산 결과.

    Attributes
    ----------
    matrix    : (N, N) 경로 손실 행렬 (dB), 링크 불가 = 0
    st_lon    : station 경도 배열 (N,)
    st_lat    : station 위도 배열 (N,)
    st_elev   : station 고도 배열 (N,) (m)
    n_linked  : 유효 링크 수 (대칭 쌍 기준)
    n_total   : 전체 가능 링크 수 N*(N-1)/2
    link_ratio: 유효 링크 비율
    """
    matrix     : np.ndarray = field(default_factory=lambda: np.array([]))
    st_lon     : np.ndarray = field(default_factory=lambda: np.array([]))
    st_lat     : np.ndarray = field(default_factory=lambda: np.array([]))
    st_elev    : np.ndarray = field(default_factory=lambda: np.array([]))
    n_linked   : int   = 0
    n_total    : int   = 0
    link_ratio : float = 0.0


# ══════════════════════════════════════════════════════════════
# Station 배치
# ══════════════════════════════════════════════════════════════

def place_stations(spatial, num_stations: int,
                   seed: int = 42,
                   progress_cb: Callable[[str], None] | None = None
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    성남시 경계 안에 num_stations개 station을 랜덤 배치.

    배치 샘플링:
      bbox 안에서 필요량의 3배를 한 번에 생성 →
      shapely.contains 로 경계 내부 필터링 →
      부족하면 반복

    Returns
    -------
    st_lon, st_lat : 위경도 배열 (N,)
    st_x,   st_y   : EPSG:3857 배열 (N,)
    """
    from shapely.geometry import MultiPoint

    def _log(msg):
        if progress_cb:
            progress_cb(msg)
        else:
            print(msg)

    np.random.seed(seed)
    b = spatial.bounds  # [lon_min, lat_min, lon_max, lat_max]
    poly = spatial.polygon_4326

    lon_list, lat_list = [], []
    _log(f"  Station {num_stations}개 배치 중...")

    while len(lon_list) < num_stations:
        batch = max(num_stations * 3, 500)
        lons  = b[0] + (b[2] - b[0]) * np.random.rand(batch)
        lats  = b[1] + (b[3] - b[1]) * np.random.rand(batch)
        pts   = MultiPoint(list(zip(lons, lats)))
        mask  = np.array([poly.contains(pt) for pt in pts.geoms])
        lon_list.extend(lons[mask].tolist())
        lat_list.extend(lats[mask].tolist())

    st_lon = np.array(lon_list[:num_stations])
    st_lat = np.array(lat_list[:num_stations])
    st_x, st_y = spatial.lonlat_to_xy(st_lon, st_lat)

    _log(f"  Station 배치 완료: {num_stations}개")
    return st_lon, st_lat, st_x, st_y


# ══════════════════════════════════════════════════════════════
# 링크 행렬 계산기
# ══════════════════════════════════════════════════════════════

class LinkMatrixCalculator:
    """
    N×N station 간 경로 손실 행렬 계산기.

    사용법
    ------
    calc = LinkMatrixCalculator(spatial, params)
    result = calc.run(progress_cb=print)
    calc.save_csv(result, "output/matrix.csv")
    """

    def __init__(self, spatial, params: LinkParams):
        self.spatial = spatial
        self.params  = params
        self.model   = PathLossModel(
            spatial,
            h_station  = params.h_station,
            env        = params.env,
            fc         = params.fc,
            n_samples  = params.n_samples,
            diff_order = params.diff_order,
        )

    def run(self,
            progress_cb: Callable[[str], None] | None = None,
            cache_dir: str = "cache"
            ) -> LinkResult:
        """
        전체 링크 행렬 계산 실행.
        동일 파라미터가 있으면 캐시에서 즉시 로드.
        """
        import os, hashlib, json

        p = self.params

        def _log(msg: str):
            if progress_cb:
                progress_cb(msg)
            else:
                print(msg)

        # ── 캐시 키 생성 ──────────────────────────────────────
        # 결과에 영향을 주는 파라미터만 포함
        cache_params = {
            'num_stations' : p.num_stations,
            'h_station'    : p.h_station,
            'env'          : p.env,
            'fc'           : p.fc,
            'n_samples'    : p.n_samples,
            'diff_order'   : p.diff_order,
            'pl_limit'     : round(p.pl_limit, 4),
            'seed'         : p.seed,
        }
        key_str  = json.dumps(cache_params, sort_keys=True)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()[:12]
        cache_path = os.path.join(cache_dir, f"pl_matrix_{key_hash}.npz")

        # ── 캐시 히트 → 즉시 로드 ────────────────────────────
        if os.path.exists(cache_path):
            _log(f"[캐시] 이전 결과 로드 중... ({cache_path})")
            data = np.load(cache_path)
            result = LinkResult(
                matrix     = data['matrix'],
                st_lon     = data['st_lon'],
                st_lat     = data['st_lat'],
                st_elev    = data['st_elev'],
                n_linked   = int(data['n_linked']),
                n_total    = int(data['n_total']),
                link_ratio = float(data['link_ratio']),
            )
            upper = result.matrix[np.triu_indices(p.num_stations, k=1)]
            _log(f"[캐시] 로드 완료 | "
                 f"유효 링크 {result.n_linked:,}/{result.n_total:,} "
                 f"({result.link_ratio*100:.1f}%)")
            return result

        # ── ① station 배치 ────────────────────────────────────
        _log("=" * 50)
        _log(f"Station 수     : {p.num_stations}개")
        _log(f"전체 링크 수   : {p.num_stations*(p.num_stations-1)//2:,}개 (대칭 기준)")
        _log(f"Margin         : {p.margin:.4f} dB  (P_edge={p.p_edge})")
        _log(f"PL_limit       : {p.pl_limit:.4f} dB"
             + (" (자동 계산)" if p.pl_limit_override <= 0 else " (직접 지정)"))
        _log("=" * 50)

        st_lon, st_lat, st_x, st_y = place_stations(
            self.spatial, p.num_stations, p.seed, progress_cb)

        # ── ② station 고도 계산 (벡터화) ─────────────────────
        _log("Station 고도 계산 중...")
        st_elev = self.spatial.get_elevation_batch(st_x, st_y)
        _log(f"  고도 범위: {st_elev.min():.0f}~{st_elev.max():.0f} m")

        # ── ③ 행렬 계산 ───────────────────────────────────────
        N      = p.num_stations
        matrix = np.zeros((N, N), dtype=np.float32)

        total_pairs = N * (N - 1) // 2
        _log(f"경로 손실 행렬 계산 시작 ({total_pairs:,}쌍)...")

        # ── 병렬 계산: ProcessPoolExecutor + initializer ──────
        # 각 워커 프로세스가 시작할 때 DEM을 한 번만 로드
        import multiprocessing as _mp
        n_cpu = max(1, _mp.cpu_count() - 1)

        from concurrent.futures import ProcessPoolExecutor as _ProcPool

        # st_x, st_y를 list로 변환 (pickle 직렬화 용이)
        st_x_list = st_x.tolist()
        st_y_list = st_y.tolist()

        task_args = [
            (i, st_x_list[i], st_y_list[i],
             st_x_list, st_y_list, p.pl_limit, N)
            for i in range(N - 1)
        ]

        completed = [0]
        spatial_   = self.spatial
        params_    = p

        try:
            with _ProcPool(
                max_workers = n_cpu,
                initializer = _worker_init,
                initargs    = (
                    spatial_.shp_path,
                    spatial_.dem_path,
                    params_.h_station,
                    params_.env,
                    params_.fc,
                    params_.n_samples,
                ),
            ) as pool:
                for row_results in pool.map(
                        _calc_row_proc, task_args, chunksize=10):
                    for ri, rj, val in row_results:
                        matrix[ri, rj] = val
                        matrix[rj, ri] = val

                    completed[0] += 1
                    if (completed[0] % max(1, N // 10) == 0
                            or completed[0] == N - 1):
                        pct = completed[0] / (N - 1) * 100
                        _log(f"  진행: {completed[0]}/{N-1} 행 완료 "
                             f"({pct:.0f}%)  [Process×{n_cpu}]")

        except Exception as proc_err:
            # ProcessPool 실패 시 Thread로 fallback
            _log(f"  [Process 실패 → Thread fallback] {proc_err}")
            matrix = np.zeros((N, N), dtype=np.float32)
            completed[0] = 0

            def _calc_row_thread(i: int) -> tuple[int, list]:
                row_results = []
                for j in range(i + 1, N):
                    pl = self.model.path_loss(
                        float(st_x[i]), float(st_y[i]),
                        float(st_x[j]), float(st_y[j]))
                    val = float(pl) if pl <= p.pl_limit else 0.0
                    row_results.append((i, j, val))
                return i, row_results

            with ThreadPoolExecutor(max_workers=min(8, N-1)) as pool:
                futures = {
                    pool.submit(_calc_row_thread, i): i
                    for i in range(N - 1)}
                for fut in as_completed(futures):
                    i, row_results = fut.result()
                    for ri, rj, val in row_results:
                        matrix[ri, rj] = val
                        matrix[rj, ri] = val
                    completed[0] += 1
                    if (completed[0] % max(1, N // 10) == 0
                            or completed[0] == N - 1):
                        pct = completed[0] / (N - 1) * 100
                        _log(f"  진행: {completed[0]}/{N-1} 행 완료 "
                             f"({pct:.0f}%)  [Thread]")

        # ── ④ 결과 정리 ───────────────────────────────────────
        # 상삼각에서 유효 링크 수 (0이 아닌 값)
        upper = matrix[np.triu_indices(N, k=1)]
        n_linked   = int(np.sum(upper > 0))
        n_total    = total_pairs
        link_ratio = n_linked / n_total if n_total > 0 else 0.0

        _log("=" * 50)
        _log(f"계산 완료!")
        _log(f"  유효 링크: {n_linked:,} / {n_total:,} ({link_ratio*100:.1f}%)")
        _log(f"  링크 불가: {n_total - n_linked:,} ({(1-link_ratio)*100:.1f}%)")
        _log(f"  유효 PL 범위: "
             f"{upper[upper>0].min():.1f}~{upper[upper>0].max():.1f} dB"
             if n_linked > 0 else "  유효 링크 없음")
        _log("=" * 50)

        result = LinkResult(
            matrix     = matrix,
            st_lon     = st_lon,
            st_lat     = st_lat,
            st_elev    = st_elev,
            n_linked   = n_linked,
            n_total    = n_total,
            link_ratio = link_ratio,
        )

        # ── 캐시 저장 ─────────────────────────────────────────
        try:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(
                cache_path,
                matrix     = matrix,
                st_lon     = st_lon,
                st_lat     = st_lat,
                st_elev    = st_elev,
                n_linked   = np.array(n_linked),
                n_total    = np.array(n_total),
                link_ratio = np.array(link_ratio),
            )
            size_mb = os.path.getsize(cache_path) / 1024 / 1024
            _log(f"[캐시] 저장 완료: {cache_path} ({size_mb:.1f} MB)")
        except Exception as e:
            _log(f"[캐시] 저장 실패 (무시): {e}")

        return result

    # ── CSV 저장 ──────────────────────────────────────────────
    @staticmethod
    def save_csv(result: LinkResult, out_path: str,
                 progress_cb: Callable[[str], None] | None = None):
        """
        경로 손실 행렬을 CSV로 저장.

        파일 구조:
          행/열 헤더: ST1, ST2, ..., STN
          값: 경로 손실 (dB), 링크 불가 = 0

        Parameters
        ----------
        result   : LinkResult
        out_path : 저장 경로 (예: "output/pl_matrix_100.csv")
        """
        def _log(msg):
            if progress_cb:
                progress_cb(msg)
            else:
                print(msg)

        N       = result.matrix.shape[0]
        headers = [f"ST{i+1}" for i in range(N)]
        df      = pd.DataFrame(result.matrix,
                               index=headers, columns=headers)

        import os
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df.to_csv(out_path, float_format="%.2f")

        size_mb = os.path.getsize(out_path) / 1024 / 1024
        _log(f"CSV 저장 완료: {out_path} ({size_mb:.1f} MB)")

    # ── station 좌표 CSV 저장 ────────────────────────────────
    @staticmethod
    def save_stations_csv(result: LinkResult, out_path: str,
                          progress_cb: Callable[[str], None] | None = None):
        """
        Station 위치 정보를 별도 CSV로 저장.

        컬럼: id, longitude, latitude, elevation_m
        """
        def _log(msg):
            if progress_cb:
                progress_cb(msg)
            else:
                print(msg)

        N  = len(result.st_lon)
        df = pd.DataFrame({
            'id'          : np.arange(1, N + 1),
            'longitude'   : result.st_lon,
            'latitude'    : result.st_lat,
            'elevation_m' : result.st_elev,
        })

        import os
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df.to_csv(out_path, index=False, float_format="%.6f")
        _log(f"Station 좌표 저장 완료: {out_path}")
