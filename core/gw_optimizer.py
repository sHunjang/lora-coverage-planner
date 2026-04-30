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
    truly_isolated  : set         = field(default_factory=set)   # 물리적 커버 불가 Node 인덱스


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
                 max_stations_per_gw: int = 0,
                 use_traffic_weight: bool = True,
                 optimize_hb: bool = False,
                 hb_candidates: list = None):

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
        self.st_x = np.array(self.st_x, dtype=np.float64)
        self.st_y = np.array(self.st_y, dtype=np.float64)

        # ── 거리 필터: Song's Model 역산으로 최대 커버 가능 거리 계산 ──
        # BPL = 39.25 + 35.15·log(fc) - 19.21·log(hb) + (42.5-5.2·log(hb))·log(d)
        # d_max: PL_limit = BPL → log(d) = (PL_limit - A) / B 역산
        _A = (39.25 + 35.15 * np.log10(fc)
              - 19.21 * np.log10(hb_gw))
        _B = 42.5 - 5.2 * np.log10(hb_gw)
        _log_d = (pl_limit - _A) / _B
        self._d_max_m = min(10 ** _log_d * 1000.0 * 1.2, 20000.0)  # 최대 30km 캡

        # ── 전체 station 간 거리 행렬 사전 계산 (scipy cdist, 빠름) ──
        from scipy.spatial.distance import cdist
        _pts = np.stack([self.st_x, self.st_y], axis=1)
        self._dist_matrix = cdist(_pts, _pts).astype(np.float32)

        if use_traffic_weight:
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(np.stack([lons, lats]))
                density = kde(np.stack([lons, lats]))
                d_min, d_max = density.min(), density.max()
                if d_max > d_min:
                    self.weights = 0.5 + (density - d_min) / (d_max - d_min)
                else:
                    self.weights = np.ones(self.N)
            except Exception:
                self.weights = np.ones(self.N)
        else:
            self.weights = np.ones(self.N)

        self._gw_candidate_lonlat = self._compute_gw_candidates(spatial)

    @staticmethod
    def _compute_gw_candidates(spatial, window: int = 5, elev_min: float = 30.0) -> np.ndarray:
        from scipy.ndimage import maximum_filter
        from pyproj import Transformer

        dem   = spatial.dem
        valid = ~np.isnan(dem)
        local_max = maximum_filter(
            np.where(valid, dem, -np.inf),
            size=window, mode='constant', cval=-np.inf)
        mask = (dem == local_max) & valid & (dem >= elev_min)

        rows, cols = np.where(mask)
        t  = spatial.dem_transform
        xs = t.c + (cols + 0.5) * t.a
        ys = t.f + (rows + 0.5) * t.e

        tr = Transformer.from_crs('EPSG:3857', 'EPSG:4326', always_xy=True)
        lons, lats = tr.transform(xs, ys)
        return np.stack([lons, lats], axis=1)

    def _rank_candidates(self):
        link_counts = np.sum(self.pl_matrix > 0, axis=1)
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

    def _calc_coverage_set(self, gw_idx: int) -> set:
        from .propagation import PathLossModel
        gx = float(self.st_x[gw_idx])
        gy = float(self.st_y[gw_idx])

        # 거리 필터: d_max_m 이내 station만 Deygout 계산 대상
        dists      = self._dist_matrix[gw_idx]
        candidates = np.where(dists <= self._d_max_m)[0]

        if not self.optimize_hb:
            covered = {gw_idx}
            for j in candidates:
                if j == gw_idx: continue
                pl = self.gw_model.path_loss(
                    gx, gy, float(self.st_x[j]), float(self.st_y[j]))
                if pl <= self.pl_limit:
                    covered.add(j)
            return covered

        best_set   = {gw_idx}
        best_score = 0.0
        for hb in self.hb_candidates:
            model_hb = PathLossModel(
                self.spatial, h_station=self.hm, hb_gw=float(hb),
                env=self.env, fc=self.fc, n_samples=self.n_samples)
            covered = {gw_idx}
            for j in candidates:
                if j == gw_idx: continue
                pl = model_hb.path_loss(
                    gx, gy, float(self.st_x[j]), float(self.st_y[j]))
                if pl <= self.pl_limit:
                    covered.add(j)
            score = sum(self.weights[j] for j in covered)
            if score > best_score:
                best_score = score
                best_set   = covered
        return best_set

    def _calc_all_coverage_sets(self, progress_cb=None):
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
                if completed[0] % max(1, self.N // 10) == 0 or completed[0] == self.N:
                    pct = completed[0] / self.N * 100
                    _log(f"  커버 집합 계산: {completed[0]}/{self.N} ({pct:.0f}%)")
        return results

    def _greedy(self, ranked, link_counts, cov_sets, progress_cb):
        def _log(msg):
            if progress_cb: progress_cb(msg)
            else: print(msg)

        cap       = self.max_stations_per_gw
        uncovered = set(range(self.N))
        gw_indices = []

        for cand in ranked:
            if not uncovered: break
            if cand not in cov_sets: continue
            new = cov_sets[cand] & uncovered
            if not new: continue
            if cap > 0 and len(new) > cap:
                new_sorted = sorted(new, key=lambda j: -self.weights[j])
                new = set(new_sorted[:cap])
            gw_indices.append(int(cand))
            uncovered -= new
            w_sum = sum(self.weights[j] for j in new)
            _log(f"  GW{len(gw_indices):3d} → ST{cand+1:4d} "
                 f"| 커버 +{len(new)}개 (가중합={w_sum:.1f})"
                 f"| 잔여 {len(uncovered)}개")

        if uncovered:
            _log(f"  고립 station {len(uncovered)}개 → 각자 GW 설치")
            for iso in sorted(uncovered):
                gw_indices.append(int(iso))
                _log(f"  고립 GW → ST{iso+1}")

        _log(f"  Greedy 완료: GW {len(gw_indices)}개")
        return gw_indices

    def _kmeans(self, k, init_indices, progress_cb):
        def _log(msg):
            if progress_cb: progress_cb(msg)
            else: print(msg)

        lons = self.stations.longitude.values
        lats = self.stations.latitude.values
        pts  = np.stack([lons, lats], axis=1)

        # 초기 중심점 준비
        if len(init_indices) >= k:
            init_centers = pts[init_indices[:k]].copy().astype(float)
        else:
            extra        = [i for i in range(self.N)
                            if i not in set(init_indices)][:k - len(init_indices)]
            init_centers = pts[list(init_indices) + extra].copy().astype(float)

        # sklearn KMeans 사용 (C++ 최적화, 순수 Python 대비 10배 빠름)
        try:
            from sklearn.cluster import KMeans
            km = KMeans(
                n_clusters  = k,
                init        = init_centers,
                n_init      = 1,
                max_iter    = self.kmeans_iter,
                random_state= self.seed,
                tol         = 1e-6,
            )
            labels = km.fit_predict(pts)
            _log(f"  K-means (sklearn): {km.n_iter_}번째 iteration 수렴")
        except ImportError:
            # sklearn 없으면 순수 numpy fallback
            _log("  K-means (numpy fallback): sklearn 없음")
            centers = init_centers.copy()
            labels  = np.zeros(self.N, dtype=int)
            for it in range(self.kmeans_iter):
                dists      = np.sqrt(((pts[:, None] - centers[None])**2).sum(axis=2))
                new_labels = np.argmin(dists, axis=1)
                new_centers = centers.copy()
                for c in range(k):
                    mask = new_labels == c
                    if mask.sum() > 0:
                        new_centers[c] = pts[mask].mean(axis=0)
                shift   = np.max(np.sqrt(np.sum((new_centers - centers)**2, axis=1)))
                centers = new_centers
                labels  = new_labels
                if shift < 1e-6:
                    _log(f"  K-means: {it+1}번째 iteration 수렴")
                    break
            else:
                _log(f"  K-means: {self.kmeans_iter}번 완료")

        u, c = np.unique(labels, return_counts=True)
        _log(f"  클러스터 크기: 최소 {c.min()} ~ 최대 {c.max()} (평균 {c.mean():.1f})")
        return labels

    def _assign_positions(self, k, labels):
        lons = self.stations.longitude.values
        lats = self.stations.latitude.values
        pts  = np.stack([lons, lats], axis=1)
        candidates     = self._gw_candidate_lonlat
        use_candidates = len(candidates) > k
        gw_indices      = []
        used_candidates = set()
        used_stations   = set()

        for c in range(k):
            mask     = labels == c
            centroid = pts[mask].mean(axis=0) if mask.sum() > 0 else pts.mean(axis=0)
            if use_candidates:
                dists = np.sqrt(np.sum((candidates - centroid)**2, axis=1))
                for ci in np.argsort(dists):
                    if ci not in used_candidates:
                        cand_pt  = candidates[ci]
                        st_dists = np.sqrt(np.sum((pts - cand_pt)**2, axis=1))
                        for si in np.argsort(st_dists):
                            if int(si) not in used_stations:
                                gw_indices.append(int(si))
                                used_stations.add(int(si))
                                used_candidates.add(ci)
                                break
                        break
            else:
                dists = np.sqrt(np.sum((pts - centroid)**2, axis=1))
                for idx in np.argsort(dists):
                    if int(idx) not in used_stations:
                        gw_indices.append(int(idx))
                        used_stations.add(int(idx))
                        break
        return np.array(gw_indices, dtype=int)

    def _verify(self, gw_indices, cov_sets):
        N      = self.N
        k      = len(gw_indices)
        node_gw = np.zeros(N, dtype=int)
        lons = self.stations.longitude.values
        lats = self.stations.latitude.values
        pts  = np.stack([lons, lats], axis=1)
        gw_pts = pts[gw_indices]

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

        uncov_idx = np.where(node_gw == 0)[0]
        if len(uncov_idx) > 0:
            diff  = pts[uncov_idx][:, np.newaxis, :] - gw_pts[np.newaxis, :, :]
            dists = np.sqrt((diff ** 2).sum(axis=2))
            nearest = np.argmin(dists, axis=1)
            for i, j in enumerate(uncov_idx):
                node_gw[j] = int(nearest[i]) + 1

        counts   = np.array([np.sum(node_gw == g+1) for g in range(k)], dtype=int)
        coverage = float(np.sum(node_gw > 0) / N)
        return node_gw, counts, coverage

    def run(self, progress_cb=None):
        def _log(msg):
            if progress_cb: progress_cb(msg)
            else: print(msg)

        _log("=" * 58)
        _log(f"  GW 최적 배치  |  GW 안테나 {self.hb_gw}m  |  PL_limit {self.pl_limit:.2f} dB")
        _log("=" * 58)

        _log("\n[Step 1] station↔station 연결 수 기반 우선순위 정렬")
        ranked, link_counts = self._rank_candidates()
        _log(f"  최다 연결: ST{ranked[0]+1} ({link_counts[ranked[0]]}개) "
             f"| 연결 0개: {int(np.sum(link_counts==0))}개")

        _log(f"\n[Step 2] GW↔station 커버 집합 계산 (GW 안테나 {self.hb_gw}m)")
        cov_sets = self._calc_all_coverage_sets(progress_cb)
        sizes    = [len(v) for v in cov_sets.values()]
        _log(f"  거리 필터: 최대 {self._d_max_m/1000:.1f}km 이내만 계산")
        _log(f"  커버 크기: 평균 {np.mean(sizes):.1f} | 최대 {max(sizes)} | 최소 {min(sizes)}")

        excluded = {g for g, s in cov_sets.items() if len(s) < self.min_cover}
        if excluded:
            _log(f"  GW 후보 제외: {len(excluded)}개 (커버 집합 크기 < {self.min_cover})")
            cov_sets = {g: s for g, s in cov_sets.items() if g not in excluded}

        _log("\n[Step 3] Greedy Set Cover")
        greedy_gws = self._greedy(ranked, link_counts, cov_sets, progress_cb)
        k = len(greedy_gws)
        _log(f"  → 필요 GW 수: {k}개")

        _log(f"\n[Step 4] K-means 클러스터링 (k={k})")
        labels = self._kmeans(k, greedy_gws, progress_cb)

        _log("\n[Step 5] GW 위치 확정 (무게중심 → 최근접 station)")
        gw_indices = self._assign_positions(k, labels)
        for g_pos, g in enumerate(gw_indices):
            _log(f"  GW{g_pos+1:3d} → ST{g+1:4d} "
                 f"| ({self.stations.longitude.iloc[g]:.5f}, "
                 f"{self.stations.latitude.iloc[g]:.5f}) "
                 f"| 고도 {self.stations.elevation_m.iloc[g]:.0f}m "
                 f"| ST↔ST {link_counts[g]}개")

        _log("\n[Step 6] 커버리지 검증")
        final_cov = {}
        for g in gw_indices:
            final_cov[g] = cov_sets.get(g) or self._calc_coverage_set(g)

        node_gw, counts, coverage = self._verify(gw_indices, final_cov)
        uncov = int(np.sum(node_gw == 0))
        _log(f"  커버: {int(np.sum(node_gw>0)):,} / {self.N:,}개")
        _log(f"  미커버: {uncov}개")
        _log(f"  커버리지: {coverage*100:.2f}%")

        if uncov > 0:
            _log(f"\n[Step 7] 미커버 {uncov}개 → GW 강제 추가")
            extra_indices = list(gw_indices)
            extra_cov     = dict(final_cov)
            for iso in list(np.where(node_gw == 0)[0]):
                extra_indices.append(int(iso))
                extra_cov[iso] = cov_sets.get(iso) or {iso}
                _log(f"  GW 추가 → ST{iso+1}")
            gw_indices = np.array(extra_indices, dtype=int)
            final_cov  = extra_cov
            node_gw, counts, coverage = self._verify(gw_indices, final_cov)
            k = len(gw_indices)
            _log(f"  → GW {k}개 | 커버리지 {coverage*100:.2f}%")

        _log(f"\n  GW당 평균: {counts.mean():.1f}개 | 최대: {counts.max()}개")

        _log(f"\n[Step 8] ILP/GA 최적화 (현재 GW {k}개 → 최소화 시도)")
        ilp_result = self.solve_ilp(final_cov, progress_cb=progress_cb, time_limit=30)

        if ilp_result is not None and len(ilp_result) > 0:
            opt_indices = ilp_result
            opt_cov = {g: final_cov.get(g) or self._calc_coverage_set(g) for g in opt_indices}
            node_gw, counts, coverage = self._verify(opt_indices, opt_cov)
            for iso in np.where(node_gw == 0)[0]:
                opt_indices = np.append(opt_indices, int(iso))
                opt_cov[iso] = cov_sets.get(iso) or {iso}
            if np.any(node_gw == 0):
                node_gw, counts, coverage = self._verify(opt_indices, opt_cov)
            gw_indices = opt_indices
            final_cov  = opt_cov
            k          = len(gw_indices)
        else:
            gw_indices, final_cov, node_gw, counts, coverage, k =                 self._ga_minimize(gw_indices, final_cov, cov_sets, progress_cb=progress_cb)

        _log("\n" + "=" * 58)
        _log(f"  완료: GW {k}개 | 커버리지 {coverage*100:.2f}%")
        _log("=" * 58)

        gw_indices, final_cov, node_gw, counts, coverage, k =             self._remove_small_gws(gw_indices, final_cov, node_gw, counts,
                                   min_cover=self.min_cover, progress_cb=progress_cb)

        _log("\n" + "=" * 58)
        _log(f"  최종: GW {k}개 | 커버리지 {coverage*100:.2f}% | 평균 {counts.mean():.1f}개/GW")
        _log("=" * 58)

        # ── 최종 커버 수 + node_gw PL 기준 재확정 ─────────────
        _log("  [최종] PL 직접 계산으로 커버 수 및 node_gw 재확정 중...")

        lons   = self.stations.longitude.values
        lats   = self.stations.latitude.values
        pts    = np.stack([lons, lats], axis=1)
        gw_pts = pts[gw_indices]

        final_counts  = np.zeros(k, dtype=int)
        final_node_gw = np.zeros(self.N, dtype=int)

        # ── 물리적 커버 불가 station 분류 ────────────────────────
        # final_cov(GW↔station 커버 집합) 기준으로 판단
        covered_by_any_gw = set()
        for _s in final_cov.values():
            covered_by_any_gw.update(_s)
        truly_isolated = set(range(self.N)) - covered_by_any_gw
        n_isolated     = len(truly_isolated)

        # ── 모든 Node를 커버 가능한 최근접 GW에 직접 배정 ──────
        # 기존: node_gw 배정 기준으로 PL 확인 → 커버 불가 GW에 배정된 Node 누락
        # 개선: 커버 집합(final_cov) 내 GW 중 가장 가까운 GW에 바로 배정
        for j in range(self.N):
            if j in truly_isolated:
                continue  # 물리적 커버 불가 → 스킵
            best_gw = -1
            best_d  = np.inf
            for g_pos, g in enumerate(gw_indices):
                if j in final_cov.get(g, set()):
                    d = float(np.sqrt(np.sum((pts[j] - gw_pts[g_pos])**2)))
                    if d < best_d:
                        best_d  = d
                        best_gw = g_pos
            if best_gw >= 0:
                # PL 최종 검증
                g  = int(gw_indices[best_gw])
                gx = float(self.st_x[g])
                gy = float(self.st_y[g])
                pl = self.gw_model.path_loss(
                    gx, gy, float(self.st_x[j]), float(self.st_y[j]))
                if pl <= self.pl_limit:
                    final_node_gw[j]       = best_gw + 1
                    final_counts[best_gw] += 1

        # ── 커버 가능인데 미배정된 경우 전체 GW 탐색 ────────────
        # (커버 집합에 있지만 PL 재확인에서 실패한 경우 대비)
        for j in range(self.N):
            if final_node_gw[j] > 0 or j in truly_isolated:
                continue
            dists = np.sqrt(np.sum((pts[j] - gw_pts)**2, axis=1))
            for g_pos in np.argsort(dists):
                g  = int(gw_indices[g_pos])
                gx = float(self.st_x[g])
                gy = float(self.st_y[g])
                pl = self.gw_model.path_loss(
                    gx, gy, float(self.st_x[j]), float(self.st_y[j]))
                if pl <= self.pl_limit:
                    final_node_gw[j]       = g_pos + 1
                    final_counts[g_pos]   += 1
                    break

        n_covered        = int(np.sum(final_node_gw > 0))
        real_coverage    = float(n_covered) / self.N
        n_coverable      = self.N - n_isolated
        n_uncov          = self.N - n_covered
        cov_of_coverable = float(n_covered) / n_coverable if n_coverable > 0 else 1.0

        _log(f"  물리적 커버 불가 station: {n_isolated}개 (지형 차단)")
        _log(f"  커버된 station: {n_covered}개 / 커버 가능: {n_coverable}개")
        _log(f"  [최종] 전체 커버리지: {n_covered}/{self.N}개 ({real_coverage*100:.1f}%)")
        _log(f"  [최종] 커버 가능 대비: {n_covered}/{n_coverable}개 ({cov_of_coverable*100:.1f}%)")
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
            truly_isolated  = truly_isolated,   # 물리적 커버 불가 Node 집합
        )

    def _remove_small_gws(self, gw_indices, final_cov, node_gw, counts,
                           min_cover=3, progress_cb=None):
        def _log(msg):
            if progress_cb: progress_cb(msg)
            else: print(msg)

        lons = self.stations.longitude.values
        lats = self.stations.latitude.values
        pts  = np.stack([lons, lats], axis=1)
        gw_list  = list(gw_indices)
        cov_dict = dict(final_cov)
        removed  = 0

        while True:
            if len(gw_list) <= 1: break
            gw_arr = np.array(gw_list)
            gw_pts = pts[gw_arr]
            diff   = pts[:, np.newaxis, :] - gw_pts[np.newaxis, :, :]
            dists  = np.sqrt((diff**2).sum(axis=2))
            assign = np.argmin(dists, axis=1)
            cnts   = np.bincount(assign, minlength=len(gw_arr))
            min_pos = int(np.argmin(cnts))
            if cnts[min_pos] >= min_cover: break
            _log(f"  [Step 9] GW{min_pos+1}(ST{gw_arr[min_pos]+1}, {cnts[min_pos]}개) 제거")
            cov_dict.pop(gw_arr[min_pos], None)
            gw_list.pop(min_pos)
            removed += 1

        if removed == 0:
            _log(f"  [Step 9] 제거 대상 없음 (모두 {min_cover}개 이상)")
        else:
            _log(f"  [Step 9] 총 {removed}개 GW 제거")

        gw_arr_f = np.array(gw_list)
        gw_pts_f = pts[gw_arr_f]

        # ── 커버 집합 기반 재할당 ──────────────────────────────
        # 순수 K-NN(거리) 대신 cov_dict 기준으로 배정
        # → 실제 PL 커버 가능한 GW에만 배정해서 미커버 최소화
        node_arr = np.zeros(self.N, dtype=int)

        # 1) 커버 가능한 station → 가장 가까운 커버 가능 GW 배정
        for j in range(self.N):
            best_gw = -1
            best_d  = np.inf
            for g_pos, g in enumerate(gw_arr_f):
                if j in cov_dict.get(g, set()):
                    d = float(np.sqrt(np.sum((pts[j] - gw_pts_f[g_pos])**2)))
                    if d < best_d:
                        best_d  = d
                        best_gw = g_pos
            if best_gw >= 0:
                node_arr[j] = best_gw + 1

        # 2) 커버 불가 station → 최근접 GW 강제 배정 (K-NN fallback)
        uncov_idx = np.where(node_arr == 0)[0]
        if len(uncov_idx) > 0:
            diff_u  = pts[uncov_idx][:, np.newaxis, :] - gw_pts_f[np.newaxis, :, :]
            dists_u = np.sqrt((diff_u**2).sum(axis=2))
            nearest = np.argmin(dists_u, axis=1)
            for i, j in enumerate(uncov_idx):
                node_arr[j] = int(nearest[i]) + 1

        cnts_f = np.array(
            [np.sum(node_arr == g+1) for g in range(len(gw_arr_f))], dtype=int)
        cov_f  = float(np.sum(node_arr > 0) / self.N)

        n_cov_assigned = int(np.sum(node_arr > 0)) - len(uncov_idx)
        _log(f"  커버 집합 기반 재할당 완료: "
             f"평균 {cnts_f.mean():.1f}개 | 최소 {cnts_f.min()} | 최대 {cnts_f.max()}")
        return gw_arr_f, cov_dict, node_arr, cnts_f, cov_f, len(gw_list)

    def solve_ilp(self, cov_sets: dict, progress_cb=None, time_limit: int = 60):
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
        prob       = pulp.LpProblem("min_gw", pulp.LpMinimize)
        x          = [pulp.LpVariable(f"x{g}", cat='Binary') for g in candidates]
        idx_map    = {g: i for i, g in enumerate(candidates)}

        prob += pulp.lpSum(x)
        for i in range(N):
            covering = [x[idx_map[g]] for g in candidates if i in cov_sets.get(g, set())]
            if covering:
                prob += pulp.lpSum(covering) >= 1

        import sys as _sys, os as _os
        if getattr(_sys, 'frozen', False):
            _cbc = _os.path.join(
                _sys._MEIPASS, 'pulp', 'solverdir', 'cbc', 'win', 'i64', 'cbc.exe')
            if _os.path.isfile(_cbc):
                solver = pulp.COIN_CMD(path=_cbc, timeLimit=time_limit, msg=0)
            else:
                _log("  [ILP] CBC 없음 → GA 사용")
                return None
        else:
            solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=0)

        status = prob.solve(solver)

        if pulp.LpStatus[status] in ('Optimal', 'Not Solved'):
            selected = [candidates[i] for i, v in enumerate(x)
                        if pulp.value(v) and pulp.value(v) > 0.5]
            _log(f"  [ILP] 완료: {len(selected)}개 GW (상태: {pulp.LpStatus[status]})")
            return np.array(selected, dtype=int)
        else:
            _log(f"  [ILP] 실패 ({pulp.LpStatus[status]}) → GA 결과 사용")
            return None

    def _ga_minimize(self, gw_indices, final_cov, cov_sets,
                     progress_cb=None, n_gen=30, pop_size=20, mutation_rate=0.1):
        def _log(msg):
            if progress_cb: progress_cb(msg)
            else: print(msg)

        k = len(gw_indices)
        if k <= 1:
            node_gw, counts, coverage = self._verify(gw_indices, final_cov)
            return gw_indices, final_cov, node_gw, counts, coverage, k

        _log(f"\n[Step 8] GA 최적화 (초기 GW {k}개 → 최소화)")

        def _fitness(bits):
            sel_idx = gw_indices[bits.astype(bool)]
            if len(sel_idx) == 0: return -self.N
            sel_cov = {g: final_cov[g] for g in sel_idx if g in final_cov}
            _, _, cov = self._verify(sel_idx, sel_cov)
            n_sel = int(bits.sum())
            return -n_sel if cov >= 1.0 else -n_sel - self.N * (1.0 - cov)

        rng = np.random.default_rng(self.seed)
        pop = [np.ones(k, dtype=np.int8)]
        for _ in range(pop_size - 1):
            bits = np.ones(k, dtype=np.int8)
            n_drop = rng.integers(1, min(4, k))
            drop_idx = rng.choice(k, n_drop, replace=False)
            bits[drop_idx] = 0
            pop.append(bits)

        best_bits  = pop[0].copy()
        best_score = _fitness(best_bits)

        for gen in range(n_gen):
            scores    = np.array([_fitness(b) for b in pop])
            elite_idx = np.argsort(-scores)[:pop_size // 2]
            new_pop   = [pop[i].copy() for i in elite_idx]

            while len(new_pop) < pop_size:
                p1, p2  = rng.choice(len(elite_idx), 2, replace=False)
                p1_bits = pop[elite_idx[p1]]
                p2_bits = pop[elite_idx[p2]]
                cx      = rng.integers(1, k)
                child   = np.concatenate([p1_bits[:cx], p2_bits[cx:]])
                flip    = rng.random(k) < mutation_rate
                child   = np.where(flip, 1 - child, child)
                if child.sum() == 0:
                    child[rng.integers(k)] = 1
                new_pop.append(child)

            pop = new_pop
            gen_best_idx   = int(np.argmax(scores))
            gen_best_score = scores[gen_best_idx]
            if gen_best_score > best_score:
                best_score = gen_best_score
                best_bits  = pop[gen_best_idx].copy() if gen_best_idx < len(pop) else best_bits

            if (gen + 1) % 10 == 0:
                _log(f"  Gen {gen+1:3d}/{n_gen} | 최선 GW {int(best_bits.sum())}개 | score={best_score:.1f}")

        sel_mask    = best_bits.astype(bool)
        opt_indices = gw_indices[sel_mask]
        opt_cov     = {g: final_cov[g] for g in opt_indices if g in final_cov}
        node_gw, counts, coverage = self._verify(opt_indices, opt_cov)

        uncov = int(np.sum(node_gw == 0))
        if uncov > 0:
            extra = list(opt_indices)
            for iso in np.where(node_gw == 0)[0]:
                extra.append(int(iso))
                opt_cov[iso] = cov_sets.get(iso) or {iso}
            opt_indices = np.array(extra, dtype=int)
            node_gw, counts, coverage = self._verify(opt_indices, opt_cov)

        k_opt = len(opt_indices)
        _log(f"  GA 완료: {k}개 → {k_opt}개 (−{k-k_opt}개) | 커버리지 {coverage*100:.2f}%")
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