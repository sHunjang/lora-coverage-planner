# core/gw_optimizer.py
"""
GW 최적 배치 모듈
────────────────────────────────────────────────────────────
[두 행렬 분리 구조]

  ① station↔station pl_matrix (이미 계산된 파일 입력)
     - 용도: 각 station의 연결 수 파악 → GW 후보 우선순위 결정
     - "어느 station이 가장 많이 통신되는가?"

  ② GW↔station gw_matrix (내부에서 자동 계산)
     - 용도: GW가 됐을 때 실제 커버 가능한 station 파악
     - GW 안테나 15m 기준
     - "GW를 여기 세우면 어느 station까지 커버되는가?"

[알고리즘 흐름]

  Step 1. GW 후보 우선순위 결정
    station↔station 연결 수 많은 순으로 GW 후보 정렬

  Step 2. GW↔station 커버 집합 계산
    각 station이 GW가 됐을 때 커버 가능한 station 집합 계산

  Step 3. Greedy Set Cover
    연결 수 많은 순으로 GW 선정
    GW↔station 커버 집합으로 실제 커버 처리
    고립 station → 그 자리에 GW 설치

  Step 4. K-means 클러스터링
    Greedy GW 수(k)로 전체 station 클러스터링

  Step 5. GW 위치 확정
    클러스터 무게중심 → 가장 가까운 station = GW 위치

  Step 6. 커버리지 검증
────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class GWResult:
    gw_indices      : np.ndarray = field(default_factory=lambda: np.array([]))
    gw_lon          : np.ndarray = field(default_factory=lambda: np.array([]))
    gw_lat          : np.ndarray = field(default_factory=lambda: np.array([]))
    gw_elev         : np.ndarray = field(default_factory=lambda: np.array([]))
    node_gw         : np.ndarray = field(default_factory=lambda: np.array([]))
    coverage        : float = 0.0
    num_gw          : int   = 0
    gw_cover_counts : np.ndarray = field(default_factory=lambda: np.array([]))
    cluster_labels  : np.ndarray = field(default_factory=lambda: np.array([]))


class GWOptimizer:

    def __init__(self,
                 pl_matrix  : np.ndarray,
                 stations   : pd.DataFrame,
                 spatial,
                 hb_gw      : float = 15.0,
                 hm         : float = 1.5,
                 env        : int   = 2,
                 fc         : float = 915.0,
                 pl_limit   : float = 140.59,
                 n_samples  : int   = 100,
                 kmeans_iter: int   = 100,
                 seed       : int   = 42,
                 min_cover  : int   = 3,
                 max_stations_per_gw: int = 0,   # 0=무제한
                 use_traffic_weight: bool = True, # 밀집도 가중치 사용
                 optimize_hb: bool = False,        # 안테나 높이 최적화
                 hb_candidates: list = None):      # 탐색할 hb 후보값(m)

        self.pl_matrix          = pl_matrix
        self.stations           = stations
        self.hb_gw              = hb_gw
        self.pl_limit           = pl_limit
        self.kmeans_iter        = kmeans_iter
        self.seed               = seed
        self.N                  = pl_matrix.shape[0]
        self.spatial            = spatial
        self.min_cover          = min_cover
        self.max_stations_per_gw = max_stations_per_gw
        self.optimize_hb        = optimize_hb
        self.hb_candidates      = hb_candidates or [10, 15, 20, 30]
        self.env                = env
        self.fc                 = fc
        self.n_samples          = n_samples
        self.hm                 = hm

        from .propagation import PathLossModel
        self.gw_model = PathLossModel(
            spatial,
            h_station = hm,
            hb_gw     = hb_gw,
            env       = env,
            fc        = fc,
            n_samples = n_samples,
        )

        lons = stations.longitude.values
        lats = stations.latitude.values
        self.st_x, self.st_y = spatial.lonlat_to_xy(lons, lats)

        # ── 트래픽 가중치: KDE 밀집도 기반 ──────────────────
        # 밀집한 구역의 station에 높은 가중치 → Greedy에서 밀집 구역 우선 커버
        if use_traffic_weight:
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(np.stack([lons, lats]))
                density = kde(np.stack([lons, lats]))
                # 0.5~1.5 범위로 정규화 (극단적 가중치 방지)
                d_min, d_max = density.min(), density.max()
                if d_max > d_min:
                    self.weights = 0.5 + (density - d_min) / (d_max - d_min)
                else:
                    self.weights = np.ones(self.N)
            except Exception:
                self.weights = np.ones(self.N)
        else:
            self.weights = np.ones(self.N)

        # ── GW 후보지 사전 계산 ──────────────────────────────
        self._gw_candidate_lonlat = self._compute_gw_candidates(spatial)

    @staticmethod
    def _compute_gw_candidates(spatial,
                                window: int = 5,
                                elev_min: float = 30.0) -> np.ndarray:
        """
        DSM 국소 최대 픽셀 좌표 추출 (GW 설치 가능 후보지).

        Parameters
        ----------
        window   : 국소 최대 탐색 윈도우 크기 (픽셀, ~window×10m 반경)
        elev_min : 최소 고도 필터 (m)

        Returns
        -------
        (M, 2) 배열: [[lon, lat], ...] EPSG:4326
        """
        from scipy.ndimage import maximum_filter
        from pyproj import Transformer

        dem   = spatial.dem
        valid = ~np.isnan(dem)

        # 국소 최대 마스크
        local_max = maximum_filter(
            np.where(valid, dem, -np.inf),
            size=window, mode='constant', cval=-np.inf)
        mask = (dem == local_max) & valid & (dem >= elev_min)

        # 픽셀 인덱스 → EPSG:3857 → EPSG:4326
        rows, cols = np.where(mask)
        t  = spatial.dem_transform
        xs = t.c + (cols + 0.5) * t.a   # 3857 x
        ys = t.f + (rows + 0.5) * t.e   # 3857 y

        tr = Transformer.from_crs('EPSG:3857', 'EPSG:4326', always_xy=True)
        lons, lats = tr.transform(xs, ys)

        return np.stack([lons, lats], axis=1)   # (M, 2)

    # ── Step 1 ──────────────────────────────────────────────
    def _rank_candidates(self):
        """
        GW 후보 우선순위 계산.
        기준: station↔station 연결 수 × 커버 station 평균 밀집도 가중치
        → 많이 연결되고 밀집한 구역을 커버하는 후보 우선
        """
        link_counts = np.sum(self.pl_matrix > 0, axis=1)

        # 각 station이 연결된 station들의 평균 밀집도 가중치
        # 연결 수 × 평균 가중치 = 밀집 구역 커버 점수
        scores = np.zeros(self.N)
        for i in range(self.N):
            neighbors = np.where(self.pl_matrix[i] > 0)[0]
            if len(neighbors) > 0:
                avg_w = self.weights[neighbors].mean()
            else:
                avg_w = self.weights[i]
            scores[i] = link_counts[i] * avg_w

        ranked = np.argsort(-scores, kind='stable')
        return ranked, link_counts

    # ── Step 2 ──────────────────────────────────────────────
    def _calc_coverage_set(self, gw_idx: int) -> set:
        """
        station i가 GW가 됐을 때 커버 집합 계산.
        optimize_hb=True이면 hb_candidates 중 커버 집합 최대인 높이 선택.
        """
        from .propagation import PathLossModel

        gx = float(self.st_x[gw_idx])
        gy = float(self.st_y[gw_idx])

        if not self.optimize_hb:
            # 기본: 고정 hb_gw
            covered = {gw_idx}
            for j in range(self.N):
                if j == gw_idx: continue
                pl = self.gw_model.path_loss(
                    gx, gy, float(self.st_x[j]), float(self.st_y[j]))
                if pl <= self.pl_limit:
                    covered.add(j)
            return covered

        # 안테나 높이 최적화: hb_candidates 중 커버 집합 가중합 최대인 것 선택
        best_set  = {gw_idx}
        best_score = 0.0

        for hb in self.hb_candidates:
            model_hb = PathLossModel(
                self.spatial,
                h_station = self.hm,
                hb_gw     = float(hb),
                env       = self.env,
                fc        = self.fc,
                n_samples = self.n_samples,
            )
            covered = {gw_idx}
            for j in range(self.N):
                if j == gw_idx: continue
                pl = model_hb.path_loss(
                    gx, gy, float(self.st_x[j]), float(self.st_y[j]))
                if pl <= self.pl_limit:
                    covered.add(j)
            # 커버된 station들의 가중치 합산
            score = sum(self.weights[j] for j in covered)
            if score > best_score:
                best_score = score
                best_set   = covered

        return best_set

    def _calc_all_coverage_sets(self,
                                 progress_cb: Callable | None = None
                                 ) -> dict[int, set]:
        def _log(msg):
            if progress_cb: progress_cb(msg)
            else: print(msg)

        results   = {}
        completed = [0]

        def _calc(i):
            return i, self._calc_coverage_set(i)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_calc, i): i for i in range(self.N)}
            for fut in as_completed(futures):
                i, cset = fut.result()
                results[i] = cset
                completed[0] += 1
                if completed[0] % max(1, self.N // 10) == 0 \
                        or completed[0] == self.N:
                    pct = completed[0] / self.N * 100
                    _log(f"  커버 집합 계산: {completed[0]}/{self.N} ({pct:.0f}%)")

        return results

    # ── Step 3 ──────────────────────────────────────────────
    def _greedy(self, ranked, link_counts, cov_sets, progress_cb):
        """
        Weighted Greedy Set Cover.
        - 트래픽 가중치: 밀집 구역 station을 먼저 커버
        - GW 용량 제한: max_stations_per_gw 초과 시 분할
        """
        def _log(msg):
            if progress_cb: progress_cb(msg)
            else: print(msg)

        cap      = self.max_stations_per_gw
        uncovered = set(range(self.N))
        gw_indices = []

        # ① 가중치 기반 Greedy
        for cand in ranked:
            if not uncovered:
                break
            if cand not in cov_sets:
                continue
            new = cov_sets[cand] & uncovered
            if not new:
                continue

            # 용량 제한: new가 cap 초과 시 이 GW는 cap개만 커버
            # 나머지는 다음 GW에서 처리
            if cap > 0 and len(new) > cap:
                # 가중치 높은 순으로 cap개만 선택
                new_sorted = sorted(new, key=lambda j: -self.weights[j])
                new = set(new_sorted[:cap])

            gw_indices.append(int(cand))
            uncovered -= new
            w_sum = sum(self.weights[j] for j in new)
            _log(f"  GW{len(gw_indices):3d} → ST{cand+1:4d} "
                 f"| 커버 +{len(new)}개 (가중합={w_sum:.1f})"
                 f"| 잔여 {len(uncovered)}개")

        # ② 고립 station → 각자 GW
        if uncovered:
            _log(f"  고립 station {len(uncovered)}개 → 각자 GW 설치")
            for iso in sorted(uncovered):
                gw_indices.append(int(iso))
                _log(f"  고립 GW → ST{iso+1}")

        _log(f"  Greedy 완료: GW {len(gw_indices)}개")
        return gw_indices

    # ── Step 4 ──────────────────────────────────────────────
    def _kmeans(self, k, init_indices, progress_cb):
        def _log(msg):
            if progress_cb: progress_cb(msg)
            else: print(msg)

        np.random.seed(self.seed)
        lons = self.stations.longitude.values
        lats = self.stations.latitude.values
        pts  = np.stack([lons, lats], axis=1)

        if len(init_indices) >= k:
            centers = pts[init_indices[:k]].copy().astype(float)
        else:
            extra   = [i for i in range(self.N)
                       if i not in set(init_indices)][:k-len(init_indices)]
            centers = pts[list(init_indices)+extra].copy().astype(float)

        labels = np.zeros(self.N, dtype=int)
        for it in range(self.kmeans_iter):
            dists      = np.sqrt(
                ((pts[:, None]-centers[None])**2).sum(axis=2))
            new_labels = np.argmin(dists, axis=1)
            new_centers = centers.copy()
            for c in range(k):
                mask = new_labels == c
                if mask.sum() > 0:
                    new_centers[c] = pts[mask].mean(axis=0)
            shift   = np.max(np.sqrt(
                np.sum((new_centers-centers)**2, axis=1)))
            centers = new_centers
            labels  = new_labels
            if shift < 1e-6:
                _log(f"  K-means: {it+1}번째 iteration 수렴")
                break
        else:
            _log(f"  K-means: {self.kmeans_iter}번 완료")

        return labels

    # ── Step 5 ──────────────────────────────────────────────
    def _assign_positions(self, k, labels):
        """
        K-means 무게중심에서 가장 가까운 GW 후보지(국소 최대 픽셀)로 확정.
        후보지가 없으면 기존 방식(가장 가까운 station)으로 fallback.
        """
        lons = self.stations.longitude.values
        lats = self.stations.latitude.values
        pts  = np.stack([lons, lats], axis=1)

        candidates = self._gw_candidate_lonlat  # (M, 2)
        use_candidates = len(candidates) > k

        gw_indices = []
        used_candidates = set()
        used_stations   = set()

        for c in range(k):
            mask     = labels == c
            centroid = pts[mask].mean(axis=0) \
                       if mask.sum() > 0 else pts.mean(axis=0)

            if use_candidates:
                # 무게중심에서 가장 가까운 미사용 후보지 픽셀 찾기
                dists = np.sqrt(np.sum(
                    (candidates - centroid)**2, axis=1))
                for ci in np.argsort(dists):
                    if ci not in used_candidates:
                        # 해당 후보지에서 가장 가까운 station을 GW 인덱스로
                        cand_pt = candidates[ci]
                        st_dists = np.sqrt(np.sum(
                            (pts - cand_pt)**2, axis=1))
                        for si in np.argsort(st_dists):
                            if int(si) not in used_stations:
                                gw_indices.append(int(si))
                                used_stations.add(int(si))
                                used_candidates.add(ci)
                                break
                        break
            else:
                # fallback: 가장 가까운 station
                dists = np.sqrt(np.sum((pts - centroid)**2, axis=1))
                for idx in np.argsort(dists):
                    if int(idx) not in used_stations:
                        gw_indices.append(int(idx))
                        used_stations.add(int(idx))
                        break

        return np.array(gw_indices, dtype=int)

    # ── Step 6 ──────────────────────────────────────────────
    def _verify(self, gw_indices, cov_sets):
        """
        GW ↔ station 할당.

        두 단계:
        1) 커버 가능한(cov_sets 기준) station → 가장 가까운 GW 배정
        2) 미커버 station → 최근접 GW 강제 배정 (K-NN)
           → GW 수 관계없이 100% 할당 보장
        """
        N       = self.N
        k       = len(gw_indices)
        node_gw = np.zeros(N, dtype=int)

        lons = self.stations.longitude.values
        lats = self.stations.latitude.values
        pts  = np.stack([lons, lats], axis=1)
        gw_pts = pts[gw_indices]   # (k, 2)

        # ── 1) 커버 가능 station → 가장 가까운 커버 가능 GW ──
        for j in range(N):
            best_gw = -1
            best_d  = np.inf
            for g_pos, g in enumerate(gw_indices):
                if j in cov_sets.get(g, set()):
                    d = float(np.sqrt(np.sum((pts[j] - gw_pts[g_pos])**2)))
                    if d < best_d:
                        best_d  = d
                        best_gw = g_pos
            if best_gw >= 0:
                node_gw[j] = best_gw + 1

        # ── 2) 미커버 station → K-NN 최근접 GW 강제 배정 ─────
        uncov_idx = np.where(node_gw == 0)[0]
        if len(uncov_idx) > 0:
            # 각 미커버 station과 모든 GW 간 거리 행렬
            diff = pts[uncov_idx][:, np.newaxis, :] - gw_pts[np.newaxis, :, :]
            dists = np.sqrt((diff ** 2).sum(axis=2))  # (n_uncov, k)
            nearest = np.argmin(dists, axis=1)
            for i, j in enumerate(uncov_idx):
                node_gw[j] = int(nearest[i]) + 1

        counts   = np.array(
            [np.sum(node_gw == g+1) for g in range(k)], dtype=int)
        coverage = float(np.sum(node_gw > 0) / N)
        return node_gw, counts, coverage

    # ── 메인 ────────────────────────────────────────────────
    def run(self, progress_cb: Callable | None = None) -> GWResult:
        def _log(msg):
            if progress_cb: progress_cb(msg)
            else: print(msg)

        _log("=" * 58)
        _log(f"  GW 최적 배치  |  GW 안테나 {self.hb_gw}m  |  "
             f"PL_limit {self.pl_limit:.2f} dB")
        _log("=" * 58)

        _log("\n[Step 1] station↔station 연결 수 기반 우선순위 정렬")
        ranked, link_counts = self._rank_candidates()
        _log(f"  최다 연결: ST{ranked[0]+1} ({link_counts[ranked[0]]}개) "
             f"| 연결 0개: {int(np.sum(link_counts==0))}개")

        _log(f"\n[Step 2] GW↔station 커버 집합 계산 (GW 안테나 {self.hb_gw}m)")
        cov_sets = self._calc_all_coverage_sets(progress_cb)
        sizes    = [len(v) for v in cov_sets.values()]
        _log(f"  커버 크기: 평균 {np.mean(sizes):.1f} | "
             f"최대 {max(sizes)} | 최소 {min(sizes)}")

        # 커버 집합이 min_cover 미만인 station은 GW 후보에서 제외
        # → 산 정상처럼 지형 차단이 심해 커버리지가 극히 좁은 위치 제거
        excluded = {g for g, s in cov_sets.items()
                    if len(s) < self.min_cover}
        if excluded:
            _log(f"  GW 후보 제외: {len(excluded)}개 "
                 f"(커버 집합 크기 < {self.min_cover})")
            cov_sets = {g: s for g, s in cov_sets.items()
                        if g not in excluded}

        _log("\n[Step 3] Greedy Set Cover")
        greedy_gws = self._greedy(ranked, link_counts, cov_sets, progress_cb)
        k = len(greedy_gws)
        _log(f"  → 필요 GW 수: {k}개")

        _log(f"\n[Step 4] K-means 클러스터링 (k={k})")
        labels = self._kmeans(k, greedy_gws, progress_cb)
        u, c   = np.unique(labels, return_counts=True)
        _log(f"  클러스터 크기: 최소 {c.min()} ~ 최대 {c.max()} "
             f"(평균 {c.mean():.1f})")

        _log("\n[Step 5] GW 위치 확정 (무게중심 → 최근접 station)")
        gw_indices = self._assign_positions(k, labels)
        for g_pos, g in enumerate(gw_indices):
            _log(f"  GW{g_pos+1:3d} → ST{g+1:4d} "
                 f"| ({self.stations.longitude.iloc[g]:.5f}, "
                 f"{self.stations.latitude.iloc[g]:.5f}) "
                 f"| 고도 {self.stations.elevation_m.iloc[g]:.0f}m "
                 f"| ST↔ST {link_counts[g]}개")

        _log("\n[Step 6] 커버리지 검증")
        # 확정 GW 커버 집합 (K-means 위치 변경 반영)
        final_cov = {}
        for g in gw_indices:
            final_cov[g] = cov_sets.get(g) or self._calc_coverage_set(g)

        node_gw, counts, coverage = self._verify(gw_indices, final_cov)
        uncov = int(np.sum(node_gw == 0))
        _log(f"  커버: {int(np.sum(node_gw>0)):,} / {self.N:,}개")
        _log(f"  미커버: {uncov}개")
        _log(f"  커버리지: {coverage*100:.2f}%")

        # ── Step 7: 미커버 station → 해당 위치에 GW 강제 추가 ──
        # K-means로 GW 위치가 바뀌면서 발생한 미커버 처리
        if uncov > 0:
            _log(f"\n[Step 7] 미커버 {uncov}개 → GW 강제 추가")
            extra_indices = list(gw_indices)
            extra_cov     = dict(final_cov)

            uncovered_idx = list(np.where(node_gw == 0)[0])
            for iso in uncovered_idx:
                extra_indices.append(int(iso))
                # 해당 station의 커버 집합 사용 (이미 계산된 경우)
                extra_cov[iso] = cov_sets.get(iso) or {iso}
                _log(f"  GW 추가 → ST{iso+1}")

            gw_indices = np.array(extra_indices, dtype=int)
            final_cov  = extra_cov
            node_gw, counts, coverage = self._verify(gw_indices, final_cov)
            k = len(gw_indices)
            _log(f"  → GW {k}개 | 커버리지 {coverage*100:.2f}%")

        _log(f"\n  GW당 평균: {counts.mean():.1f}개 | 최대: {counts.max()}개")

        # ── Step 8: ILP 우선, GA fallback ─────────────────────
        _log(f"\n[Step 8] ILP/GA 최적화 (현재 GW {k}개 → 최소화 시도)")

        ilp_result = self.solve_ilp(
            final_cov, progress_cb=progress_cb, time_limit=30)

        if ilp_result is not None and len(ilp_result) > 0:
            opt_indices = ilp_result
            opt_cov = {g: final_cov.get(g) or self._calc_coverage_set(g)
                       for g in opt_indices}
            node_gw, counts, coverage = self._verify(opt_indices, opt_cov)
            # 미커버 처리
            for iso in np.where(node_gw == 0)[0]:
                opt_indices = np.append(opt_indices, int(iso))
                opt_cov[iso] = cov_sets.get(iso) or {iso}
            if np.any(node_gw == 0):
                node_gw, counts, coverage = self._verify(
                    opt_indices, opt_cov)
            gw_indices = opt_indices
            final_cov  = opt_cov
            k          = len(gw_indices)
        else:
            # ILP 실패 → GA fallback
            gw_indices, final_cov, node_gw, counts, coverage, k = \
                self._ga_minimize(
                    gw_indices, final_cov, cov_sets,
                    progress_cb=progress_cb)

        _log("\n" + "=" * 58)
        _log(f"  완료: GW {k}개 | 커버리지 {coverage*100:.2f}%")
        _log("=" * 58)

        # ── Step 9: 소규모 GW 제거 ────────────────────────────
        # 커버 station이 min_cover 미만인 GW는 제거하고
        # 해당 station들을 인접 GW에 재배정
        gw_indices, final_cov, node_gw, counts, coverage, k = \
            self._remove_small_gws(
                gw_indices, final_cov, node_gw, counts,
                min_cover=self.min_cover, progress_cb=progress_cb)

        _log("\n" + "=" * 58)
        _log(f"  최종: GW {k}개 | 커버리지 {coverage*100:.2f}% "
             f"| 평균 {counts.mean():.1f}개/GW")
        _log("=" * 58)

        # ── 최종 커버 수 + node_gw PL 기준 재확정 ─────────────
        _log("  [최종] PL 직접 계산으로 커버 수 및 node_gw 재확정 중...")

        final_counts  = np.zeros(k, dtype=int)
        final_node_gw = np.zeros(self.N, dtype=int)

        for g_pos in range(k):
            g  = int(gw_indices[g_pos])
            gx = float(self.st_x[g])
            gy = float(self.st_y[g])
            for j in range(self.N):
                if int(node_gw[j]) != g_pos + 1:
                    continue
                pl = self.gw_model.path_loss(
                    gx, gy,
                    float(self.st_x[j]), float(self.st_y[j]))
                if pl <= self.pl_limit:
                    final_counts[g_pos] += 1
                    final_node_gw[j]    = g_pos + 1

        n_covered   = int(np.sum(final_node_gw > 0))
        n_uncovered = self.N - n_covered

        # ── 커버 불가 station 분류 ─────────────────────────────
        # pl_matrix에서 링크 수 0 = 어떤 GW로도 물리적 커버 불가
        link_counts_all = np.sum(self.pl_matrix > 0, axis=1)
        truly_isolated  = set(np.where(link_counts_all == 0)[0])
        n_isolated      = len(truly_isolated)

        # 커버 가능한데 현재 미커버인 station (링크는 있지만 배정 안 됨)
        recoverable_uncov = [
            j for j in np.where(final_node_gw == 0)[0]
            if j not in truly_isolated
        ]

        _log(f"  물리적 커버 불가 station: {n_isolated}개 (지형 차단)")
        _log(f"  커버 가능 미커버 station: {len(recoverable_uncov)}개")

        # ── 커버 가능 미커버 → 최근접 GW로 재배정 ──────────────
        # (물리적 커버 불가 station은 제외)
        if recoverable_uncov:
            _log(f"  커버 가능 미커버 {len(recoverable_uncov)}개 → 최근접 GW 재배정")
            lons = self.stations.longitude.values
            lats = self.stations.latitude.values
            pts  = np.stack([lons, lats], axis=1)
            gw_pts = pts[gw_indices]

            for j in recoverable_uncov:
                # 가장 가까운 GW에 재배정 시도
                dists = np.sqrt(np.sum((pts[j] - gw_pts)**2, axis=1))
                for g_pos in np.argsort(dists):
                    g  = int(gw_indices[g_pos])
                    gx = float(self.st_x[g])
                    gy = float(self.st_y[g])
                    pl = self.gw_model.path_loss(
                        gx, gy,
                        float(self.st_x[j]), float(self.st_y[j]))
                    if pl <= self.pl_limit:
                        final_node_gw[j]     = g_pos + 1
                        final_counts[g_pos] += 1
                        break

        n_covered     = int(np.sum(final_node_gw > 0))
        n_uncovered   = self.N - n_covered
        real_coverage = float(n_covered) / self.N

        # 커버 가능 station 대비 커버리지
        n_coverable   = self.N - n_isolated
        cov_of_coverable = (
            float(n_covered) / n_coverable if n_coverable > 0 else 1.0)

        _log(f"  [최종] 전체 커버리지: {n_covered}/{self.N}개 "
             f"({real_coverage*100:.1f}%)")
        _log(f"  [최종] 커버 가능 대비: {n_covered}/{n_coverable}개 "
             f"({cov_of_coverable*100:.1f}%)")
        _log(f"  [최종] 물리적 미커버: {n_isolated}개 (지형 차단, GW 추가로 해결 불가)")
        _log(f"  [최종] GW {k}개 | 평균 {final_counts.mean():.1f}개/GW")

        return GWResult(
            gw_indices      = gw_indices,
            gw_lon          = self.stations.longitude.values[gw_indices],
            gw_lat          = self.stations.latitude.values[gw_indices],
            gw_elev         = self.stations.elevation_m.values[gw_indices],
            node_gw         = final_node_gw,
            coverage        = real_coverage,
            num_gw          = k,
            gw_cover_counts = final_counts,
            cluster_labels  = labels,
        )

    # ── Step 9: 소규모 GW 제거 ──────────────────────────────
    def _remove_small_gws(self, gw_indices, final_cov,
                           node_gw, counts, min_cover=3,
                           progress_cb=None):
        """
        커버 station 수가 min_cover 미만인 GW 제거.
        제거 후 전체 station을 K-NN으로 잔존 GW에 재할당.

        Parameters
        ----------
        min_cover : 이 수 미만 GW 제거 (기본 3)
        """
        def _log(msg):
            if progress_cb: progress_cb(msg)
            else: print(msg)

        lons = self.stations.longitude.values
        lats = self.stations.latitude.values
        pts  = np.stack([lons, lats], axis=1)

        gw_list  = list(gw_indices)
        cov_dict = dict(final_cov)
        removed  = 0

        # min_cover 미만 GW 반복 제거
        while True:
            if len(gw_list) <= 1:
                break
            gw_arr = np.array(gw_list)
            gw_pts = pts[gw_arr]

            # 현재 K-NN 할당
            diff   = pts[:, np.newaxis, :] - gw_pts[np.newaxis, :, :]
            dists  = np.sqrt((diff**2).sum(axis=2))  # (N, k)
            assign = np.argmin(dists, axis=1)         # (N,)
            cnts   = np.bincount(assign, minlength=len(gw_arr))

            # 가장 작은 GW 찾기
            min_pos = int(np.argmin(cnts))
            if cnts[min_pos] >= min_cover:
                break  # 모두 min_cover 이상

            _log(f"  [Step 9] GW{min_pos+1}(ST{gw_arr[min_pos]+1}, "
                 f"{cnts[min_pos]}개) 제거")
            cov_dict.pop(gw_arr[min_pos], None)
            gw_list.pop(min_pos)
            removed += 1

        if removed == 0:
            _log(f"  [Step 9] 제거 대상 없음 (모두 {min_cover}개 이상)")
        else:
            _log(f"  [Step 9] 총 {removed}개 GW 제거")

        # 최종 K-NN 재할당
        gw_arr_f = np.array(gw_list)
        gw_pts_f = pts[gw_arr_f]
        diff_f   = pts[:, np.newaxis, :] - gw_pts_f[np.newaxis, :, :]
        dists_f  = np.sqrt((diff_f**2).sum(axis=2))
        assign_f = np.argmin(dists_f, axis=1)
        node_arr = assign_f + 1  # 1-based

        cnts_f   = np.array(
            [np.sum(node_arr == g+1) for g in range(len(gw_arr_f))],
            dtype=int)
        cov_f    = float(np.sum(node_arr > 0) / self.N)

        _log(f"  K-NN 재할당 완료: 평균 {cnts_f.mean():.1f}개 | "
             f"최소 {cnts_f.min()} | 최대 {cnts_f.max()}")

        return gw_arr_f, cov_dict, node_arr, cnts_f, cov_f, len(gw_list)
    def solve_ilp(self, cov_sets: dict,
                  progress_cb=None,
                  time_limit: int = 60) -> np.ndarray:
        """
        ILP(정수 선형 계획법)로 최소 GW 수를 구합니다.

        Minimize  Σ x_g
        s.t.      Σ_{g: i∈S_g} x_g >= 1  ∀ station i
                  x_g ∈ {0, 1}

        Parameters
        ----------
        cov_sets   : {gw_idx: set(station_idx)} 커버 집합
        time_limit : 최대 계산 시간 (초)

        Returns
        -------
        np.ndarray : 선택된 GW 인덱스 배열
        """
        def _log(msg):
            if progress_cb: progress_cb(msg)
            else: print(msg)

        try:
            import pulp
        except ImportError:
            _log("  [ILP] pulp 없음 → Greedy 결과 사용")
            return None

        _log(f"  [ILP] 최소 GW 집합 탐색 중... (최대 {time_limit}초)")

        N          = self.N
        candidates = list(cov_sets.keys())
        n_cand     = len(candidates)

        # 변수 생성
        prob = pulp.LpProblem("min_gw", pulp.LpMinimize)
        x    = [pulp.LpVariable(f"x{g}", cat='Binary')
                for g in candidates]
        idx_map = {g: i for i, g in enumerate(candidates)}

        # 목적함수: GW 수 최소화
        prob += pulp.lpSum(x)

        # 제약: 모든 station이 최소 1개 GW에 커버
        for i in range(N):
            covering = [x[idx_map[g]]
                        for g in candidates
                        if i in cov_sets.get(g, set())]
            if covering:
                prob += pulp.lpSum(covering) >= 1

        # 풀기
        solver = pulp.PULP_CBC_CMD(
            timeLimit=time_limit, msg=0)
        status = prob.solve(solver)

        if pulp.LpStatus[status] in ('Optimal', 'Not Solved'):
            selected = [candidates[i]
                        for i, v in enumerate(x)
                        if pulp.value(v) and pulp.value(v) > 0.5]
            _log(f"  [ILP] 완료: {len(selected)}개 GW "
                 f"(상태: {pulp.LpStatus[status]})")
            return np.array(selected, dtype=int)
        else:
            _log(f"  [ILP] 실패 ({pulp.LpStatus[status]}) → GA 결과 사용")
            return None

    # ── Step 8: 유전 알고리즘 ────────────────────────────────
    def _ga_minimize(self, gw_indices, final_cov, cov_sets,
                     progress_cb=None,
                     n_gen=30, pop_size=20, mutation_rate=0.1):
        """
        유전 알고리즘으로 GW 수 최소화 (커버리지 100% 유지).

        현재 GW 집합을 초기 해로 사용 → 불필요한 GW 제거.
        각 개체: 0/1 비트 배열 (1=GW 포함, 0=제외)
        적합도: 커버리지 100% 만족 시 GW 수 최소, 불만족 시 패널티

        Parameters
        ----------
        n_gen         : 세대 수 (기본 30)
        pop_size      : 개체군 크기 (기본 20)
        mutation_rate : 비트 반전 확률 (기본 0.1)
        """
        def _log(msg):
            if progress_cb: progress_cb(msg)
            else: print(msg)

        k = len(gw_indices)
        if k <= 1:
            node_gw, counts, coverage = self._verify(gw_indices, final_cov)
            return gw_indices, final_cov, node_gw, counts, coverage, k

        _log(f"\n[Step 8] GA 최적화 (초기 GW {k}개 → 최소화)")

        # 적합도 함수
        def _fitness(bits):
            """적합도: 커버리지 100%면 GW 수 최소화, 아니면 패널티."""
            sel_idx = gw_indices[bits.astype(bool)]
            if len(sel_idx) == 0:
                return -self.N
            sel_cov = {g: final_cov[g] for g in sel_idx if g in final_cov}
            _, _, cov = self._verify(sel_idx, sel_cov)
            n_sel = int(bits.sum())
            if cov >= 1.0:
                return -n_sel          # 커버 완전 → GW 수 최소화
            else:
                return -n_sel - self.N * (1.0 - cov)  # 미커버 패널티

        # 초기 개체군 생성
        rng = np.random.default_rng(self.seed)
        # 첫 번째: 전체 GW 포함 (현재 해)
        pop = [np.ones(k, dtype=np.int8)]
        # 나머지: 랜덤으로 일부 제거
        for _ in range(pop_size - 1):
            bits = np.ones(k, dtype=np.int8)
            # 랜덤하게 1~3개 GW 제거
            n_drop = rng.integers(1, min(4, k))
            drop_idx = rng.choice(k, n_drop, replace=False)
            bits[drop_idx] = 0
            pop.append(bits)

        best_bits  = pop[0].copy()
        best_score = _fitness(best_bits)

        for gen in range(n_gen):
            # 적합도 계산
            scores = np.array([_fitness(b) for b in pop])

            # 엘리트 선택 (상위 50%)
            elite_idx = np.argsort(-scores)[:pop_size // 2]
            new_pop   = [pop[i].copy() for i in elite_idx]

            # 교차 + 변이로 나머지 채우기
            while len(new_pop) < pop_size:
                p1, p2 = rng.choice(len(elite_idx), 2, replace=False)
                p1_bits = pop[elite_idx[p1]]
                p2_bits = pop[elite_idx[p2]]
                # 단순 교차
                cx = rng.integers(1, k)
                child = np.concatenate([p1_bits[:cx], p2_bits[cx:]])
                # 변이
                flip = rng.random(k) < mutation_rate
                child = np.where(flip, 1 - child, child)
                # 최소 1개 GW 보장
                if child.sum() == 0:
                    child[rng.integers(k)] = 1
                new_pop.append(child)

            pop = new_pop

            # 현재 세대 최선 업데이트
            gen_best_idx   = int(np.argmax(scores))
            gen_best_score = scores[gen_best_idx]
            if gen_best_score > best_score:
                best_score = gen_best_score
                best_bits  = pop[gen_best_idx].copy() \
                             if gen_best_idx < len(pop) \
                             else best_bits

            if (gen + 1) % 10 == 0:
                sel_k = int(best_bits.sum())
                _log(f"  Gen {gen+1:3d}/{n_gen} | "
                     f"최선 GW {sel_k}개 | score={best_score:.1f}")

        # 최종 결과 적용
        sel_mask    = best_bits.astype(bool)
        opt_indices = gw_indices[sel_mask]
        opt_cov     = {g: final_cov[g] for g in opt_indices if g in final_cov}
        node_gw, counts, coverage = self._verify(opt_indices, opt_cov)

        # 커버리지 100% 보장: 미커버 있으면 강제 추가
        uncov = int(np.sum(node_gw == 0))
        if uncov > 0:
            extra = list(opt_indices)
            for iso in np.where(node_gw == 0)[0]:
                extra.append(int(iso))
                opt_cov[iso] = cov_sets.get(iso) or {iso}
            opt_indices = np.array(extra, dtype=int)
            node_gw, counts, coverage = self._verify(opt_indices, opt_cov)

        k_opt = len(opt_indices)
        _log(f"  GA 완료: {k}개 → {k_opt}개 "
             f"(−{k-k_opt}개) | 커버리지 {coverage*100:.2f}%")

        return opt_indices, opt_cov, node_gw, counts, coverage, k_opt

    @staticmethod
    def save_gw_csv(result, out_path, progress_cb=None):
        import os
        def _log(msg):
            if progress_cb: progress_cb(msg)
            else: print(msg)
        df = pd.DataFrame({
            'gw_id'      : np.arange(1, result.num_gw+1),
            'station_idx': result.gw_indices + 1,
            'longitude'  : result.gw_lon,
            'latitude'   : result.gw_lat,
            'elevation_m': result.gw_elev,
            'cover_count': result.gw_cover_counts,
        })
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df.to_csv(out_path, index=False, float_format="%.6f")
        _log(f"GW 위치 저장: {out_path}")

    @staticmethod
    def save_assignment_csv(result, stations, out_path, progress_cb=None):
        import os
        def _log(msg):
            if progress_cb: progress_cb(msg)
            else: print(msg)
        N  = len(stations)
        df = pd.DataFrame({
            'station_id' : np.arange(1, N+1),
            'longitude'  : stations.longitude.values,
            'latitude'   : stations.latitude.values,
            'elevation_m': stations.elevation_m.values,
            'assigned_gw': result.node_gw,
            'cluster'    : result.cluster_labels,
        })
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df.to_csv(out_path, index=False, float_format="%.6f")
        _log(f"Station 할당 결과 저장: {out_path}")
