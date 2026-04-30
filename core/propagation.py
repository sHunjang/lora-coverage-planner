# core/propagation.py
"""
전파 경로 손실 계산 모듈
────────────────────────────────────────────────────────────
[모델 구조]

  SongsModel   : 거리·환경·안테나 높이 기반 기본 경로 손실 (dB)
  DeygoutDiff  : 지형 회절 손실
  PathLossModel: 두 모델을 결합한 최종 경로 손실 계산기

[Song's Model]
  BPL = 39.25 + 35.15·log(fc) - 19.21·log(hb)
        + (42.5 - 5.2·log(hb))·log(d_km)
  PL  = BPL - ahm(환경 보정치)

  환경(env):
    1 = Dense Urban
    2 = Urban (기본)
    3 = Suburban
    4 = Open

[Deygout 알고리즘]
  1. 시선과 DEM의 차이로 Fresnel-Kirchhoff 파라미터 v 계산
     v = h_eff × sqrt(2(d1+d2) / (λ·d1·d2))
  2. v > 0인 주요 능선 P를 찾아 회절 손실 J(v) 계산
  3. P 기준으로 양쪽 구간 재귀 탐색 (max_order=2)
  4. 모든 손실 합산

[총 경로 손실]
  PL_total = PL_songs(d_total) + L_deygout

[링크 불가 조건]
  PL_total > pl_limit → 해당 링크는 연결 불가 (0 반환)
────────────────────────────────────────────────────────────

[전체 계산 흐름 요약]

LOS:
  PL = Song's Model
     (변곡점 내 장애물 → +20dB)

NLOS (장애물 1~3개):
  LD_t  = J(v1) [+ J(v2) [+ J(v3)]]   ← Deygout 재귀
  PL_FS = 20·log10(fc) + 20·log10(d_km) - 27.5492
  DL    = PL_FS + LD_t
  PL    = max(Song's, DL)

NLOS (장애물 3개 이상):
  PL    = PL_FS + 200dB  ← 사실상 커버 불가

"""

from __future__ import annotations
import numpy as np


# ══════════════════════════════════════════════════════════════
# Song's Model
# ══════════════════════════════════════════════════════════════

class SongsModel:
    """
    Song's Model 기반 기본 경로 손실.
    
    역할: 거리와 환경에 따른 기본 경로손실 계산

    BPL = 39.25 + 35.15·log(fc) - 19.21·log(hb)
        + (42.5 - 5.2·log(hb))·log(d_km)
    환경별 ahm:

    Dense Urban: 18.9·log(hm) - 1.29·log(fc) - 11.5
    Urban:       18.4·log(hm) - 0.99·log(fc) + 2.0
    Suburban:    17.2·log(hm) - 0.6·log(fc) - 2.7
    Open:        17.2·log(hm) + 13

    PL = BPL - ahm(환경보정)

    Parameters
    ----------
    fc  : 반송 주파수 (MHz), 기본 915 MHz
    hb  : 송신 안테나 높이 (m), 기본 1.5 m  ← station 간이므로 낮게
    hm  : 수신 안테나 높이 (m), 기본 1.5 m
    env : 환경 코드 (1~4)
    """

    def __init__(self, fc: float = 915.0, hb: float = 1.5,
                 hm: float = 1.5, env: int = 2):
        self.fc  = fc
        self.hb  = hb
        self.hm  = hm
        self.env = env

        # 환경별 단말 안테나 보정값 ahm
        env_map = {
            1: 18.9 * np.log10(hm) - 1.29 * np.log10(fc) - 11.5,  # Dense Urban
            2: 18.4 * np.log10(hm) - 0.99 * np.log10(fc) + 2.0,   # Urban
            3: 17.2 * np.log10(hm) - 0.6  * np.log10(fc) - 2.7,   # Suburban
            4: 17.2 * np.log10(hm) + 13,                            # Open
        }
        if env not in env_map:
            raise ValueError(f"env는 1~4 사이여야 합니다. 입력값: {env}")
        self.ahm = env_map[env]

    def bpl(self, d_km: float) -> float:
        """기본 경로 손실 (Basic Path Loss)."""
        d = max(d_km, 1e-3)
        return (39.25
                + 35.15 * np.log10(self.fc)
                - 19.21 * np.log10(self.hb)
                + (42.5 - 5.2 * np.log10(self.hb)) * np.log10(d))

    def path_loss(self, d_km: float) -> float:
        """최종 경로 손실 (dB)."""
        return self.bpl(d_km) - self.ahm


# ══════════════════════════════════════════════════════════════
# Deygout 회절 손실
# ══════════════════════════════════════════════════════════════

class DeygoutDiff:
    """
    Deygout 방법 기반 지형 회절 손실 계산기.
    
    역할: 지형/건물 때문에 신호가 꺾일 때의 추가 손실 계산


    v = h_eff × √(2(d1+d2) / (λ·d1·d2))

    h_eff = 장애물 높이 - 시선(LOS) 높이  (양수면 장애물이 시선 위로 튀어나옴)
    d1    = 송신기 ~ 장애물 거리
    d2    = 장애물 ~ 수신기 거리
    λ     = 파장

    Parameters
    ----------
    fc        : 반송 주파수 (MHz)
    max_order : 재귀 탐색 깊이 (기본 2)
    """

    def __init__(self, fc: float = 915.0, max_order: int = 2):
        self.fc        = fc
        self.lam       = 3e8 / (fc * 1e6)  # 파장 (m)
        self.max_order = max_order

    def _fresnel_v(self, h_eff: float, d1: float, d2: float) -> float:
        """Fresnel-Kirchhoff 파라미터 v."""
        if d1 <= 0 or d2 <= 0:
            return -np.inf
        return h_eff * np.sqrt(2 * (d1 + d2) / (self.lam * d1 * d2))

    @staticmethod
    def _jv(v: float) -> float:
        """
        회절 손실 J(v) (dB).
          v ≤ 0        : 0 (시선 여유)
          0 < v ≤ 2.4  : 6.02 + 9.11v + 1.27v²
          v > 2.4      : 13 + 20·log10(v)
        """
        if v <= 0.0:
            return 0.0
        if v <= 2.4:
            return 6.02 + 9.11 * v + 1.27 * v * v
        return 13.0 + 20.0 * np.log10(v)

    def _v_profile(self, dists, elevs, i_tx, i_rx, h_tx, h_rx) -> np.ndarray:
        """구간 [i_tx, i_rx] 내 Fresnel-v 배열."""
        n = i_rx - i_tx + 1
        if n < 3:
            return np.full(n, -np.inf)

        e_tx    = elevs[i_tx] + h_tx
        e_rx    = elevs[i_rx] + h_rx
        d_tx    = dists[i_tx]
        d_rx    = dists[i_rx]
        d_total = d_rx - d_tx
        if d_total <= 0.0:
            return np.full(n, -np.inf)

        vs = np.full(n, -np.inf)
        for k in range(1, n - 1):
            idx     = i_tx + k
            d1      = dists[idx] - d_tx
            d2      = d_rx - dists[idx]
            sight_h = e_tx + (e_rx - e_tx) * (d1 / d_total)
            h_eff   = elevs[idx] - sight_h
            vs[k]   = self._fresnel_v(h_eff, d1, d2)
        return vs

    def _deygout_recursive(self, dists, elevs,
                           i_tx, i_rx, h_tx, h_rx, order) -> float:
        """재귀 Deygout 회절 손실 누적."""
        if i_rx <= i_tx + 1:
            return 0.0

        vs    = self._v_profile(dists, elevs, i_tx, i_rx, h_tx, h_rx)
        k_max = int(np.argmax(vs))

        if vs[k_max] <= 0.0:
            return 0.0

        i_peak = i_tx + k_max
        loss   = self._jv(float(vs[k_max]))

        if order > 0:
            loss += self._deygout_recursive(
                dists, elevs, i_tx, i_peak, h_tx, 0.0, order - 1)
            loss += self._deygout_recursive(
                dists, elevs, i_peak, i_rx, 0.0, h_rx, order - 1)
        return loss

    def diffraction_loss(self, dists, elevs, h_tx, h_rx) -> float:
        """
        NLOS 회절 경로 손실
        DL = PL_FS + LD_t

        PL_FS: 전체 경로 자유공간 손실
            = 20·log10(fc[MHz]) + 20·log10(d[km]) - 27.5492
        LD_t:  Deygout knife-edge 누적 손실 = J(v1) + J(v2) + ...

        장애물 3개 이상: PDF 명세에 따라 최저값(매우 큰 손실) 반환
        """
        if len(dists) < 3:
            return 0.0

        # ── 장애물 수 사전 파악 ──────────────────────────────────
        # 시선(LOS line) 위로 튀어나온 점 = 잠재적 장애물
        e_tx   = elevs[0]  + h_tx
        e_rx   = elevs[-1] + h_rx
        sight  = np.linspace(e_tx, e_rx, len(elevs))
        n_obs  = int(np.sum(elevs > sight))  # 시선 위 샘플 수

        # 시선 위 샘플이 많으면 실질적 장애물 3개 이상으로 간주
        # (샘플 수 기반이므로 임계값은 샘플 밀도 고려: 3개 × 최소 간격)
        if n_obs > max(3 * (len(dists) // 50), 8):
            # PDF 명세: 장애물 3개 이상 → 최저값 처리
            d_total_km = max(float(dists[-1]) / 1000.0, 0.001)
            pl_fs = (20.0 * np.log10(self.fc)
                    + 20.0 * np.log10(d_total_km)
                    - 27.5492)
            return pl_fs + 200.0  # 사실상 커버 불가

        # ── LD_t: Deygout knife-edge 누적 손실 ───────────────────
        ld_t = self._deygout_recursive(
            dists, elevs, 0, len(dists) - 1, h_tx, h_rx, self.max_order)
        ld_t = max(0.0, ld_t)

        # ── PL_FS: 전체 경로 자유공간 손실 ───────────────────────
        d_total_km = max(float(dists[-1]) / 1000.0, 0.001)
        pl_fs = (20.0 * np.log10(self.fc)
                + 20.0 * np.log10(d_total_km)
                - 27.5492)

        return pl_fs + ld_t


# ══════════════════════════════════════════════════════════════
# 결합 모델: Song's Model + Deygout
# ══════════════════════════════════════════════════════════════

class PathLossModel:
    """
    Song's Model + Deygout 회절 손실 결합 계산기.

    station ↔ station 간 경로 손실 계산에 사용.
    송수신 안테나 높이가 동일(hb = hm = h_station)하다고 가정.

    Parameters
    ----------
    spatial    : SpatialData 인스턴스
    h_station  : station 안테나 지상 높이 (m), 기본 1.5 m
    env        : Song's Model 환경 코드 (1~4)
    fc         : 반송 주파수 (MHz)
    n_samples  : DEM 단면 샘플 수 (기본 100)
    diff_order : Deygout 재귀 깊이 (기본 2)
    """

    def __init__(self, spatial,
                 h_station: float = 1.5,
                 env: int = 2,
                 fc: float = 915.0,
                 n_samples: int = 100,
                 diff_order: int = 2,
                 hb_gw: float | None = None):
        """
        Parameters
        ----------
        h_station : station 안테나 높이 hm (m), 기본 1.5m
        hb_gw     : GW 안테나 높이 hb (m). None이면 h_station과 동일.
                    GW↔station 링크 계산 시 반드시 GW 안테나 높이를 지정해야
                    Song's Model이 올바른 hb/hm으로 PL을 계산함.
        """
        self.spatial    = spatial
        self.h_station  = h_station
        self.n_samples  = n_samples

        # hb: GW 안테나 높이, hm: station 안테나 높이
        hb = hb_gw if hb_gw is not None else h_station
        hm = h_station

        self._auto_env = (env == 0)     # env=0 이면 경로별 자동 분류
        _env = env if env != 0 else 2   # 초기 기본값 Urban
        self.songs = SongsModel(fc=fc, hb=hb, hm=hm, env=_env)

        self.deygout = DeygoutDiff(fc=fc, max_order=diff_order)

        # Deygout 단면 샘플링 시 송수신 안테나 높이
        self._h_tx = float(hb)   # GW 안테나 높이
        self._h_rx = float(hm)   # station 안테나 높이

        # DEM 배열 직접 참조
        self._dem  = spatial.dem
        self._ox   = spatial.ox
        self._oy   = spatial.oy
        self._res  = spatial.res
        self._rows = spatial.dem_rows
        self._cols = spatial.dem_cols

    # ── DEM 단면 샘플링 (벡터화) ─────────────────────────────
    def _sample_profile(self,
                        x1: float, y1: float,
                        x2: float, y2: float
                        ) -> tuple[np.ndarray, np.ndarray]:
        """
        두 점 사이의 DEM 고도 단면을 numpy 배열 인덱싱으로 샘플링.

        Returns
        -------
        dists : (n_samples,) — 수평 거리 배열 (m)
        elevs : (n_samples,) — 고도 배열 (m), NaN → 50m 폴백
        """
        n  = self.n_samples
        xs = np.linspace(x1, x2, n)
        ys = np.linspace(y1, y2, n)

        cols = np.clip(
            ((xs - self._ox) / self._res).astype(int), 0, self._cols - 1)
        rows = np.clip(
            ((self._oy - ys) / self._res).astype(int), 0, self._rows - 1)

        elevs = self._dem[rows, cols]
        elevs = np.where(np.isnan(elevs), 50.0, elevs)

        d_tot = np.hypot(x2 - x1, y2 - y1)
        dists = np.linspace(0, d_tot, n)
        return dists, elevs

    # ── 단일 링크 경로 손실 ──────────────────────────────────
    def path_loss(self, x1, y1, x2, y2) -> float:
        d_m  = max(np.hypot(x2 - x1, y2 - y1), 1.0)
        d_km = d_m / 1000.0

        pl_songs = self.songs.path_loss(d_km)
        dists, elevs = self._sample_profile(x1, y1, x2, y2)

        # ── LOS/NLOS 판단 ─────────────────────────────────────
        gw_abs  = elevs[0] + self._h_tx   # GW 절대 고도
        rx_abs  = elevs[-1] + self._h_rx  # 단말 절대 고도
        sight   = np.linspace(gw_abs, rx_abs, len(elevs))
        nlos    = bool(np.any(elevs > sight))

        if not nlos:
            # 가시거리(LOS): Song's Model만
            # 변곡점 거리 내 GW 높이보다 높은 장애물 → +20dB
            lam     = 3e8 / (self.songs.fc * 1e6)
            r_inf_m = np.sqrt(4 * self._h_tx * self._h_rx / lam)
            if d_m <= r_inf_m and len(elevs) > 2:
                if float(np.max(elevs[1:-1])) > gw_abs:
                    return pl_songs + 20.0
            return pl_songs
        else:
            # NLOS: DL = PL_FS + LD_t
            # max(Song's Model, DL) 적용
            l_diff = self.deygout.diffraction_loss(
                dists, elevs, h_tx=self._h_tx, h_rx=self._h_rx)
            return max(pl_songs, l_diff)

    # ── 상세 분해 (디버그/분석용) ────────────────────────────
    def path_loss_detail(self, x1, y1, x2, y2) -> dict:
        d_m  = max(np.hypot(x2 - x1, y2 - y1), 1.0)
        d_km = d_m / 1000.0

        pl_songs = self.songs.path_loss(d_km)
        dists, elevs = self._sample_profile(x1, y1, x2, y2)

        gw_abs = elevs[0] + self._h_tx
        rx_abs = elevs[-1] + self._h_rx
        sight  = np.linspace(gw_abs, rx_abs, len(elevs))
        nlos   = bool(np.any(elevs > sight))

        lam     = 3e8 / (self.songs.fc * 1e6)
        r_inf_m = np.sqrt(4 * self._h_tx * self._h_rx / lam)

        if not nlos:
            extra = 0.0
            if d_m <= r_inf_m and len(elevs) > 2:
                if float(np.max(elevs[1:-1])) > gw_abs:
                    extra = 20.0
            l_diff    = 0.0
            pl_fs_val = 0.0   # ← 추가
            ld_t      = 0.0   # ← 추가
            pl_total  = pl_songs + extra
        else:
            ld_t = self.deygout._deygout_recursive(
                dists, elevs, 0, len(dists)-1,
                self._h_tx, self._h_rx, self.deygout.max_order)
            ld_t = max(0.0, ld_t)

            d_km_val  = max(float(dists[-1]) / 1000.0, 0.001)
            pl_fs_val = (20.0 * np.log10(self.songs.fc)
                        + 20.0 * np.log10(d_km_val)
                        - 27.5492)
            l_diff   = pl_fs_val + ld_t
            pl_total = max(pl_songs, l_diff)

        return {
            'pl_total': pl_total,
            'pl_songs': pl_songs,
            'l_diff'  : l_diff,
            'pl_fs'   : pl_fs_val,
            'ld_t'    : ld_t,
            'nlos'    : nlos,
            'd_km'    : d_km,
            'dists'   : dists,
            'elevs'   : elevs,
        }