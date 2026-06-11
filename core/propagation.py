# core/propagation.py
"""
전파 경로 손실 계산 모듈
────────────────────────────────────────────────────────────
[모델 구조]

  SongsModel    : SmartCity LoRaScape Model (Song's 기반)
  COST231Model  : COST-231 Hata Model
  DeygoutDiff   : 지형 회절 손실 (v1.9a — 버그 수정)
  PathLossModel : 모델 결합 최종 경로 손실 계산기

[전체 계산 흐름]

LOS (h_eff_max <= 0 → 시선선 아래 장애물 없음):
  PL = 기본모델(d, env_경로평균)
     (근거리 변곡점 장애물 → +20dB)

준-LOS (h_eff_max > 0 이지만 v_max <= 0):
  PL = 기본모델 (회절 손실 없음 — Fresnel Zone 미침범)

NLOS (h_eff_max > 0 AND v_max > 0):
  LD_t  = J(v1)+13 [+ J(v2)+7]   ← Deygout 재귀 + 차수별 보정
  PL_FS = 20·log10(fc) + 20·log10(d_km) - 27.5492
  DL    = PL_FS + LD_t
  PL    = max(기본모델, DL)

다중 장애물 (n_obs > obs_threshold):
  PL = PL_FS + min(n_obs × 8dB, 80dB)   ← 단계적 손실 (최대 80dB)
  ※ 기존 200dB 일률 → 개선

[v1.9a 버그 수정 내용]
  Bug 1: 평가 결과 V_LOS_THRESHOLD = 0.0 이 올바른 값으로 유지.
         (평지 v_max ≈ -0.33, 능선 v_max ≈ +0.85 → 0.0 기준 명확히 구분)

  Bug 2: diffraction_loss() 내부에 h_eff_max / v_max 이중 LOS 체크 제거
         path_loss()에서 이미 NLOS임을 확인하고 diffraction_loss()를
         호출하므로, 내부에서 또 LOS 체크 → 0 반환하면 항상 0이 됨.
         diffraction_loss()는 이제 NLOS 전용 함수로 단순화.

  Bug 3: _pl_base()가 env=0(자동)일 때 songs/cost231 캐시 오브젝트를
         fixed_env=2(Urban)로 생성해 놓고, 실제 env≠2인 경우는
         매번 새 모델 인스턴스를 생성함 → 느리지만 정확도는 OK.
         이는 버그는 아니나 최적화 포인트로 주석 추가.

[클러터 환경 자동 분류 — v1.8 유지]
  env=0(자동) 모드에서 경로 중간 구간의 DSM 고도 통계를
  get_env_code()로 샘플링하여 경로별 최빈 환경 코드를 결정.
────────────────────────────────────────────────────────────
"""

from __future__ import annotations
from collections import Counter
import numpy as np

# ── Fresnel v 임계값 ──────────────────────────────────────
# v <= 0: 장애물이 1차 Fresnel Zone 밖 → 회절 손실 없음 (J(v)=0)
# v >  0: Fresnel Zone 침범 → Deygout 적용
#
# 측정 검증: 평지 v_max≈-0.33, 능선(+20m) v_max≈+0.85
# → 0.0이 LOS/NLOS를 명확히 구분하는 올바른 경계값
V_LOS_THRESHOLD = 0.0

# 다중 장애물당 추가 손실 (dB) — 경험값
MULTI_OBS_LOSS_PER = 8.0
MULTI_OBS_LOSS_MAX = 80.0


# ══════════════════════════════════════════════════════════════
# SmartCity LoRaScape Model (Song's Model)
# ══════════════════════════════════════════════════════════════

class SongsModel:
    """
    SmartCity LoRaScape Model (Song's Model 기반).

    BPL = 39.25 + 35.15·log(fc) - 19.21·log(hb)
        + (42.5 - 5.2·log(hb))·log(d_km)
    PL  = BPL - ahm

    환경별 ahm:
      Dense Urban: 18.9·log(hm) - 1.29·log(fc) - 11.5
      Urban:       18.4·log(hm) - 0.99·log(fc) + 2.0
      Suburban:    17.2·log(hm) - 0.6·log(fc) - 2.7
      Open:        17.2·log(hm) + 13
    """

    def __init__(self, fc: float = 915.0, hb: float = 1.5,
                 hm: float = 1.5, env: int = 2):
        self.fc  = fc
        self.hb  = hb
        self.hm  = hm
        self.env = env

        env_map = {
            1: 18.9 * np.log10(hm) - 1.29 * np.log10(fc) - 11.5,
            2: 18.4 * np.log10(hm) - 0.99 * np.log10(fc) + 2.0,
            3: 17.2 * np.log10(hm) - 0.6  * np.log10(fc) - 2.7,
            4: 17.2 * np.log10(hm) + 13,
        }
        if env not in env_map:
            raise ValueError(f"env는 1~4 사이여야 합니다. 입력값: {env}")
        self.ahm = env_map[env]

    def bpl(self, d_km: float) -> float:
        d = max(d_km, 1e-3)
        return (39.25
                + 35.15 * np.log10(self.fc)
                - 19.21 * np.log10(self.hb)
                + (42.5 - 5.2 * np.log10(self.hb)) * np.log10(d))

    def path_loss(self, d_km: float) -> float:
        return self.bpl(d_km) - self.ahm


# ══════════════════════════════════════════════════════════════
# COST-231 Hata Model
# ══════════════════════════════════════════════════════════════

class COST231Model:
    """
    COST-231 Hata 모델 기반 경로 손실.

    PL = 46.3 + 33.9·log(f) - 13.82·log(hb) - a(hm)
       + (44.9 - 6.55·log(hb))·log(d) + Cm
    """

    def __init__(self, fc: float = 915.0, hb: float = 15.0,
                 hm: float = 1.5, env: int = 2):
        self.fc  = fc
        self.hb  = max(hb, 1.0)
        self.hm  = max(hm, 0.1)
        self.env = env

    def _ahm(self) -> float:
        fc, hm = self.fc, self.hm
        if self.env == 1:
            if fc <= 300:
                return 8.29 * (np.log10(1.54 * hm)) ** 2 - 1.1
            else:
                return 3.2 * (np.log10(11.75 * hm)) ** 2 - 4.97
        else:
            return ((1.1 * np.log10(fc) - 0.7) * hm
                    - (1.56 * np.log10(fc) - 0.8))

    def path_loss(self, d_km: float) -> float:
        d   = max(d_km, 1e-3)
        fc  = self.fc
        hb  = self.hb
        ahm = self._ahm()

        if self.env in (1, 2):
            Cm = 3.0 if self.env == 1 else 0.0
            return (46.3
                    + 33.9 * np.log10(fc)
                    - 13.82 * np.log10(hb)
                    - ahm
                    + (44.9 - 6.55 * np.log10(hb)) * np.log10(d)
                    + Cm)
        else:
            ahm_urban = ((1.1 * np.log10(fc) - 0.7) * self.hm
                         - (1.56 * np.log10(fc) - 0.8))
            pl_urban = (46.3
                        + 33.9 * np.log10(fc)
                        - 13.82 * np.log10(hb)
                        - ahm_urban
                        + (44.9 - 6.55 * np.log10(hb)) * np.log10(d))
            if self.env == 3:
                return pl_urban - 2 * (np.log10(fc / 28)) ** 2 - 5.4
            else:
                return (pl_urban
                        - 4.78 * (np.log10(fc)) ** 2
                        + 18.33 * np.log10(fc)
                        - 40.94)


# ══════════════════════════════════════════════════════════════
# Deygout 회절 손실 — v1.9a
# ══════════════════════════════════════════════════════════════

class DeygoutDiff:
    """
    Deygout 방법 기반 지형 회절 손실 계산기 (v1.9a).

    [v1.9a 수정 내용]
    1. V_LOS_THRESHOLD = 0.0 유지 (검증 완료)
       측정 결과: 평지 v_max≈-0.33, 능선(+20m) v_max≈+0.85
       → 0.0이 LOS/NLOS를 명확히 구분하는 올바른 값.

    2. diffraction_loss() 내부 LOS 재확인 로직 제거 [핵심 버그 수정]
       [Bug 2 수정] path_loss()가 이미 NLOS임을 확인하고 호출하는데,
       diffraction_loss() 내부에서 또 h_eff_max / v_max를 체크해
       return 0.0 하면 → 항상 0이 반환되어 히트맵이 완전 원형이 됨.
       diffraction_loss()는 이제 NLOS 전용으로 단순화:
         - 다중 장애물 단계적 손실 체크 (200dB → n×8dB)
         - Deygout 재귀 계산

    3. 회절 차수 보정 유지
       - 1차: J(v) + 13 dB
       - 2차: J(v) + 7 dB
    """

    def __init__(self, fc: float = 915.0, max_order: int = 2):
        self.fc        = fc
        self.lam       = 3e8 / (fc * 1e6)
        self.max_order = max_order

    def _fresnel_v(self, h_eff: float, d1: float, d2: float) -> float:
        if d1 <= 0 or d2 <= 0:
            return -np.inf
        return h_eff * np.sqrt(2 * (d1 + d2) / (self.lam * d1 * d2))

    @staticmethod
    def _jv(v: float) -> float:
        if v <= 0.0:
            return 0.0
        if v <= 2.4:
            return 6.02 + 9.11 * v + 1.27 * v * v
        return 13.0 + 20.0 * np.log10(v)

    def _v_profile(self, dists, elevs, i_tx, i_rx,
                   h_tx, h_rx) -> np.ndarray:
        """경로 구간의 모든 점에 대한 Fresnel v 배열 계산."""
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

    def _v_max_full(self, dists, elevs, h_tx, h_rx) -> float:
        """
        전체 경로에서 Fresnel v 최댓값 반환.
        LOS/NLOS 판단에 사용 (path_loss() 에서 호출).
        """
        if len(dists) < 3:
            return -np.inf

        e_tx    = elevs[0]  + h_tx
        e_rx    = elevs[-1] + h_rx
        d_total = float(dists[-1])
        if d_total <= 0:
            return -np.inf

        v_max = -np.inf
        for k in range(1, len(dists) - 1):
            d1      = float(dists[k])
            d2      = d_total - d1
            sight_h = e_tx + (e_rx - e_tx) * (d1 / d_total)
            h_eff   = float(elevs[k]) - sight_h
            v       = self._fresnel_v(h_eff, d1, d2)
            if v > v_max:
                v_max = v
        return v_max

    def _deygout_recursive(self, dists, elevs,
                           i_tx, i_rx, h_tx, h_rx, order) -> float:
        """
        재귀 Deygout 회절 손실 누적.
        1차 회절 (order == max_order): J(v) + 13 dB
        2차 회절 (order == max_order-1): J(v) + 7 dB
        """
        if i_rx <= i_tx + 1:
            return 0.0

        vs    = self._v_profile(dists, elevs, i_tx, i_rx, h_tx, h_rx)
        k_max = int(np.argmax(vs))

        if vs[k_max] <= 0.0:
            return 0.0

        i_peak = i_tx + k_max
        loss   = self._jv(float(vs[k_max]))

        if order == self.max_order:
            loss += 13.0
        elif order == self.max_order - 1:
            loss += 7.0

        if order > 0:
            loss += self._deygout_recursive(
                dists, elevs, i_tx, i_peak, h_tx, 0.0, order - 1)
            loss += self._deygout_recursive(
                dists, elevs, i_peak, i_rx, 0.0, h_rx, order - 1)
        return loss

    def diffraction_loss(self, dists, elevs, h_tx, h_rx) -> float:
        """
        NLOS 전용 회절 경로 손실 계산 (v1.9a).

        [Bug 2 수정] 이 함수는 path_loss()가 NLOS임을 확인한 후에만 호출.
        따라서 내부에서 LOS 재확인(h_eff_max 체크 / v_max 체크 후 return 0)을
        하면 안 됨 — 그렇게 하면 항상 0을 반환해 히트맵이 완전 원형이 됨.

        이 함수의 역할:
          1. 다중 장애물이면 단계적 손실(n_obs × 8dB, 최대 80dB) 반환
          2. 그 외에는 Deygout 재귀 계산 결과 반환
        """
        if len(dists) < 3:
            return 0.0

        d_total_km = max(float(dists[-1]) / 1000.0, 0.001)
        pl_fs = (20.0 * np.log10(self.fc)
                 + 20.0 * np.log10(d_total_km)
                 - 27.5492)

        # ── 다중 장애물 단계적 손실 ───────────────────────────
        # (LOS 체크는 path_loss()에서 이미 완료됨 — 여기서는 하지 않음)
        e_tx  = elevs[0]  + h_tx
        e_rx  = elevs[-1] + h_rx
        sight = np.linspace(e_tx, e_rx, len(elevs))
        n_obs = int(np.sum(elevs > sight))

        obs_threshold = max(3 * (len(dists) // 50), 8)

        if n_obs > obs_threshold:
            # 200dB 일률 → 장애물 수 비례 단계적 손실
            extra = min(n_obs * MULTI_OBS_LOSS_PER, MULTI_OBS_LOSS_MAX)
            return pl_fs + extra

        # ── Deygout 회절 손실 계산 ────────────────────────────
        ld_t = self._deygout_recursive(
            dists, elevs, 0, len(dists) - 1,
            h_tx, h_rx, self.max_order)
        ld_t = max(0.0, ld_t)

        return pl_fs + ld_t


# ══════════════════════════════════════════════════════════════
# 클러터 환경 샘플링 헬퍼
# ══════════════════════════════════════════════════════════════

_CLUTTER_SAMPLES = 5
_CLUTTER_RADIUS  = 10


def _get_path_env(spatial, xs: np.ndarray, ys: np.ndarray) -> int:
    """
    경로 중간 1/4~3/4 구간을 _CLUTTER_SAMPLES개 샘플링하여
    최빈 환경 코드(1~4)를 반환.
    """
    if spatial is None or not hasattr(spatial, 'get_env_code'):
        return 2

    n     = len(xs)
    mid_s = n // 4
    mid_e = n * 3 // 4
    step  = max(1, (mid_e - mid_s) // _CLUTTER_SAMPLES)

    codes = []
    for i in range(mid_s, mid_e, step):
        try:
            code = spatial.get_env_code(
                float(xs[i]), float(ys[i]),
                radius_px=_CLUTTER_RADIUS)
            codes.append(code)
        except Exception:
            codes.append(2)

    if not codes:
        return 2
    return int(Counter(codes).most_common(1)[0][0])

def _get_path_ahm_avg(spatial, xs, ys, hm, fc):
    """
    경로 중간 구간을 샘플링해 각 점의 env별 ahm을 평균.
    최빈값 1개 대신 ahm 평균을 써서 격자 간 전이를 부드럽게 함.
    반환: ahm 평균값 (float)
    """
    if spatial is None or not hasattr(spatial, 'get_env_code'):
        # Urban 기본 ahm
        return 18.4 * np.log10(hm) - 0.99 * np.log10(fc) + 2.0

    def _ahm(env):
        m = {
            1: 18.9 * np.log10(hm) - 1.29 * np.log10(fc) - 11.5,
            2: 18.4 * np.log10(hm) - 0.99 * np.log10(fc) + 2.0,
            3: 17.2 * np.log10(hm) - 0.6  * np.log10(fc) - 2.7,
            4: 17.2 * np.log10(hm) + 13,
        }
        return m.get(env, m[2])

    n     = len(xs)
    mid_s = n // 4
    mid_e = n * 3 // 4
    step  = max(1, (mid_e - mid_s) // _CLUTTER_SAMPLES)

    ahms = []
    for i in range(mid_s, mid_e, step):
        try:
            code = spatial.get_env_code(
                float(xs[i]), float(ys[i]), radius_px=_CLUTTER_RADIUS)
            ahms.append(_ahm(code))
        except Exception:
            ahms.append(_ahm(2))

    if not ahms:
        return _ahm(2)
    return float(np.mean(ahms))

# ══════════════════════════════════════════════════════════════
# 결합 모델: PathLossModel
# ══════════════════════════════════════════════════════════════

class PathLossModel:
    """
    SmartCity LoRaScape / COST-231 + Deygout(v1.9a) + 클러터 결합 경로 손실 계산기.

    Parameters
    ----------
    spatial    : SpatialData 인스턴스
    h_station  : 단말 안테나 지상 높이 (m)
    env        : 환경 코드 (0=자동, 1~4)
                 0=자동 → 경로별 DSM 클러터 샘플링으로 env 자동 결정
    fc         : 반송 주파수 (MHz)
    n_samples  : DEM 단면 샘플 수
    diff_order : Deygout 재귀 깊이
    hb_gw      : GW 안테나 높이 (m)
    prop_model : 'smartcity' | 'cost231'
    """

    def __init__(self, spatial,
                 h_station: float = 1.5,
                 env: int = 2,
                 fc: float = 915.0,
                 n_samples: int = 100,
                 diff_order: int = 2,
                 hb_gw: float | None = None,
                 prop_model: str = 'smartcity'):

        self.spatial    = spatial
        self.h_station  = h_station
        self.n_samples  = n_samples
        self.prop_model = prop_model

        hb = hb_gw if hb_gw is not None else h_station
        hm = h_station

        self._auto_env  = (env == 0)
        self._fixed_env = env if env != 0 else 2

        self.songs   = SongsModel(fc=fc,   hb=hb, hm=hm, env=self._fixed_env)
        self.cost231 = COST231Model(fc=fc, hb=hb, hm=hm, env=self._fixed_env)
        self.deygout = DeygoutDiff(fc=fc, max_order=diff_order)

        self._h_tx = float(hb)
        self._h_rx = float(hm)
        self._fc   = fc
        self._hb   = hb
        self._hm   = hm

        self._dem  = spatial.dem
        self._ox   = spatial.ox
        self._oy   = spatial.oy
        self._res  = spatial.res
        self._rows = spatial.dem_rows
        self._cols = spatial.dem_cols

    # ── DEM 단면 샘플링 ──────────────────────────────────────
    def _sample_profile(self, x1: float, y1: float,
                        x2: float, y2: float
                        ) -> tuple[np.ndarray, np.ndarray]:
        
        d_dot = np.hypot(x2 - x1, y2 - y1)
        n = int(np.clip(d_dot / self._res, self.n_samples, 1000))
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

    # ── 경로 환경 코드 결정 ──────────────────────────────────
    def _resolve_env(self, x1: float, y1: float,
                     x2: float, y2: float) -> int:
        if not self._auto_env:
            return self._fixed_env
        n  = self.n_samples
        xs = np.linspace(x1, x2, n)
        ys = np.linspace(y1, y2, n)
        return _get_path_env(self.spatial, xs, ys)

    # ── 환경 코드로 기본 모델 경로 손실 계산 ─────────────────
    def _pl_base(self, d_km: float, env: int) -> float:
        # env가 cached fixed_env와 같으면 미리 생성된 인스턴스 재사용 (빠름)
        # env가 다르면 (자동 분류에서 env 바뀜) 새 인스턴스 생성 (정확)
        if env == self._fixed_env:
            if self.prop_model == 'cost231':
                return self.cost231.path_loss(d_km)
            else:
                return self.songs.path_loss(d_km)
        else:
            if self.prop_model == 'cost231':
                return COST231Model(
                    fc=self._fc, hb=self._hb,
                    hm=self._hm, env=env).path_loss(d_km)
            else:
                return SongsModel(
                    fc=self._fc, hb=self._hb,
                    hm=self._hm, env=env).path_loss(d_km)

    # ── 단일 링크 경로 손실 ──────────────────────────────────
    def path_loss(self, x1, y1, x2, y2) -> float:
        d_m  = max(np.hypot(x2 - x1, y2 - y1), 1.0)
        d_km = d_m / 1000.0

        # 클러터 환경 결정
        if self._auto_env and self.prop_model != 'cost231':
            # [방안 B] 경로 ahm 가중평균으로 부드러운 전이 (smartcity 전용)
            xs = np.linspace(x1, x2, self.n_samples)
            ys = np.linspace(y1, y2, self.n_samples)
            ahm_avg = _get_path_ahm_avg(
                self.spatial, xs, ys, self._hm, self._fc)
            pl_base = self.songs.bpl(d_km) - ahm_avg
            env     = 0  # 자동 (가중평균 적용됨 표시)
        else:
            env     = self._resolve_env(x1, y1, x2, y2)
            pl_base = self._pl_base(d_km, env)

        dists, elevs = self._sample_profile(x1, y1, x2, y2)

        gw_abs = elevs[0]  + self._h_tx
        rx_abs = elevs[-1] + self._h_rx

        # ── LOS/NLOS 판단 (v1.9a) ────────────────────────────
        # Step 1: h_eff_max — 시선선 위로 돌출한 지형이 있는지 먼저 확인
        sight_    = np.linspace(gw_abs, rx_abs, len(elevs))
        h_eff_arr = elevs - sight_
        h_eff_max = (float(np.max(h_eff_arr[1:-1]))
                     if len(h_eff_arr) > 2 else -np.inf)

        if h_eff_max <= 0.0:
            # 시선선 아래 → 완전 LOS
            # 단, 근거리 변곡점(r_inf) 이내 장애물이 GW보다 높으면 +20dB
            lam     = 3e8 / (self._fc * 1e6)
            r_inf_m = np.sqrt(4 * self._h_tx * self._h_rx / lam)
            if d_m <= r_inf_m and len(elevs) > 2:
                if float(np.max(elevs[1:-1])) > gw_abs:
                    return pl_base + 20.0
            return pl_base

        # Step 2: h_eff_max > 0 이면 Fresnel v 확인
        # v_max <= V_LOS_THRESHOLD(0.0): Fresnel Zone 여유 있음 → LOS 처리
        # v_max >  V_LOS_THRESHOLD(0.0): Fresnel Zone 침범 → NLOS
        v_max = self.deygout._v_max_full(dists, elevs, self._h_tx, self._h_rx)

        if v_max <= V_LOS_THRESHOLD:
            # 준-LOS: 장애물이 시선선 위에 있지만 Fresnel Zone은 여유
            return pl_base

        # 순수 지형 회절 손실(자유공간 제외분)만 추출해서 pl_base에 가산
        l_diff = self.deygout.diffraction_loss(
            dists, elevs,
            h_tx=self._h_tx,
            h_rx=self._h_rx)
        d_km_fs = max(d_km, 0.001)
        pl_fs   = (20.0 * np.log10(self._fc)
                   + 20.0 * np.log10(d_km_fs) - 27.5492)
        # diffraction_loss = pl_fs + 회절분 → 회절분만 분리
        diff_excess = max(0.0, l_diff - pl_fs)
        diff_excess = min(diff_excess, 30.0)
        return pl_base + diff_excess

    # ── 상세 분해 (디버그/분석용) ────────────────────────────
    def path_loss_detail(self, x1, y1, x2, y2) -> dict:
        d_m  = max(np.hypot(x2 - x1, y2 - y1), 1.0)
        d_km = d_m / 1000.0

        env     = self._resolve_env(x1, y1, x2, y2)
        pl_base = self._pl_base(d_km, env)

        dists, elevs = self._sample_profile(x1, y1, x2, y2)

        gw_abs = elevs[0]  + self._h_tx
        rx_abs = elevs[-1] + self._h_rx

        sight_    = np.linspace(gw_abs, rx_abs, len(elevs))
        h_eff_arr = elevs - sight_
        h_eff_max = (float(np.max(h_eff_arr[1:-1]))
                     if len(h_eff_arr) > 2 else -np.inf)

        lam     = 3e8 / (self._fc * 1e6)
        r_inf_m = np.sqrt(4 * self._h_tx * self._h_rx / lam)

        if h_eff_max <= 0.0:
            extra = 0.0
            if d_m <= r_inf_m and len(elevs) > 2:
                if float(np.max(elevs[1:-1])) > gw_abs:
                    extra = 20.0
            v_max    = self.deygout._v_max_full(dists, elevs,
                                                self._h_tx, self._h_rx)
            l_diff   = 0.0
            pl_fs_val = 0.0
            ld_t      = 0.0
            nlos      = False
            pl_total  = pl_base + extra
        else:
            v_max = self.deygout._v_max_full(dists, elevs,
                                             self._h_tx, self._h_rx)
            nlos  = v_max > V_LOS_THRESHOLD

            if not nlos:
                l_diff    = 0.0
                pl_fs_val = 0.0
                ld_t      = 0.0
                pl_total  = pl_base
            else:
                ld_t = self.deygout._deygout_recursive(
                    dists, elevs, 0, len(dists) - 1,
                    self._h_tx, self._h_rx, self.deygout.max_order)
                ld_t = max(0.0, ld_t)

                d_km_val  = max(float(dists[-1]) / 1000.0, 0.001)
                pl_fs_val = (20.0 * np.log10(self._fc)
                             + 20.0 * np.log10(d_km_val)
                             - 27.5492)
                l_diff   = pl_fs_val + ld_t
                pl_total = max(pl_base, l_diff)

        return {
            'pl_total'  : pl_total,
            'pl_base'   : pl_base,
            'l_diff'    : l_diff,
            'pl_fs'     : pl_fs_val,
            'ld_t'      : ld_t,
            'nlos'      : nlos,
            'h_eff_max' : h_eff_max,
            'v_max'     : v_max,
            'd_km'      : d_km,
            'env'       : env,
            'auto_env'  : self._auto_env,
            'gw_abs'    : gw_abs,
            'rx_abs'    : rx_abs,
            'dists'     : dists,
            'elevs'     : elevs,
        }