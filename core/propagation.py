# core/propagation.py
"""
전파 경로 손실 계산 모듈
────────────────────────────────────────────────────────────
[모델 구조]

  SongsModel    : SmartCity LoRaScape Model (Song's 기반)
  COST231Model  : COST-231 Hata Model
  DeygoutDiff   : 지형 회절 손실
  PathLossModel : 모델 결합 최종 경로 손실 계산기

[전체 계산 흐름]

LOS:
  PL = 기본모델(d)
     (변곡점 내 장애물 → +20dB)

NLOS (장애물 1~3개):
  LD_t  = J(v1)+13 [+ J(v2)+7]   ← Deygout 재귀 + 차수별 보정
  PL_FS = 20·log10(fc) + 20·log10(d_km) - 27.5492
  DL    = PL_FS + LD_t
  PL    = max(기본모델, DL)

NLOS (장애물 3개 이상):
  PL    = PL_FS + 200dB
────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import numpy as np


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
# Deygout 회절 손실
# ══════════════════════════════════════════════════════════════

class DeygoutDiff:
    """
    Deygout 방법 기반 지형 회절 손실 계산기.

    1차 회절: J(v) + 13 dB
    2차 회절: J(v) + 7 dB
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

        # 회절 차수별 보정값
        if order == self.max_order:
            loss += 13.0       # 1차 회절
        elif order == self.max_order - 1:
            loss += 7.0        # 2차 회절

        if order > 0:
            loss += self._deygout_recursive(
                dists, elevs, i_tx, i_peak, h_tx, 0.0, order - 1)
            loss += self._deygout_recursive(
                dists, elevs, i_peak, i_rx, 0.0, h_rx, order - 1)
        return loss

    def diffraction_loss(self, dists, elevs, h_tx, h_rx) -> float:
        """
        NLOS 회절 경로 손실.
        장애물 수가 많으면 PL_FS + 200dB (사실상 커버 불가).
        """
        if len(dists) < 3:
            return 0.0

        e_tx  = elevs[0]  + h_tx
        e_rx  = elevs[-1] + h_rx
        sight = np.linspace(e_tx, e_rx, len(elevs))
        n_obs = int(np.sum(elevs > sight))

        d_total_km = max(float(dists[-1]) / 1000.0, 0.001)
        pl_fs = (20.0 * np.log10(self.fc)
                 + 20.0 * np.log10(d_total_km)
                 - 27.5492)

        if n_obs > max(3 * (len(dists) // 50), 8):
            return pl_fs + 200.0

        ld_t = self._deygout_recursive(
            dists, elevs, 0, len(dists) - 1,
            h_tx, h_rx, self.max_order)
        ld_t = max(0.0, ld_t)

        return pl_fs + ld_t


# ══════════════════════════════════════════════════════════════
# 결합 모델: PathLossModel
# ══════════════════════════════════════════════════════════════

class PathLossModel:
    """
    SmartCity LoRaScape / COST-231 + Deygout 결합 경로 손실 계산기.

    Parameters
    ----------
    spatial    : SpatialData 인스턴스
    h_station  : 단말 안테나 지상 높이 (m)
    env        : 환경 코드 (0=자동, 1~4)
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

        self._auto_env = (env == 0)
        _env = env if env != 0 else 2

        self.songs   = SongsModel(fc=fc, hb=hb, hm=hm, env=_env)
        self.cost231 = COST231Model(fc=fc, hb=hb, hm=hm, env=_env)
        self.deygout = DeygoutDiff(fc=fc, max_order=diff_order)

        self._h_tx = float(hb)   # GW 안테나 높이
        self._h_rx = float(hm)   # 단말 안테나 높이
        self._fc   = fc

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

        # 기본 경로손실
        if self.prop_model == 'cost231':
            pl_base = self.cost231.path_loss(d_km)
        else:
            pl_base = self.songs.path_loss(d_km)

        dists, elevs = self._sample_profile(x1, y1, x2, y2)

        # GW/단말 절대 고도 (DSM 고도 + 안테나 높이)
        gw_abs = elevs[0]  + self._h_tx
        rx_abs = elevs[-1] + self._h_rx

        # LOS/NLOS 판단
        sight = np.linspace(gw_abs, rx_abs, len(elevs))
        nlos  = bool(np.any(elevs > sight))

        if not nlos:
            # LOS: 기본 모델
            # 변곡점 거리 내 GW보다 높은 장애물 → +20dB
            lam     = 3e8 / (self._fc * 1e6)
            r_inf_m = np.sqrt(4 * self._h_tx * self._h_rx / lam)
            if d_m <= r_inf_m and len(elevs) > 2:
                if float(np.max(elevs[1:-1])) > gw_abs:
                    return pl_base + 20.0
            return pl_base
        else:
            # NLOS: max(기본모델, PL_FS + Deygout)
            l_diff = self.deygout.diffraction_loss(
                dists, elevs,
                h_tx=self._h_tx,
                h_rx=self._h_rx)
            return max(pl_base, l_diff)

    # ── 상세 분해 (디버그/분석용) ────────────────────────────
    def path_loss_detail(self, x1, y1, x2, y2) -> dict:
        d_m  = max(np.hypot(x2 - x1, y2 - y1), 1.0)
        d_km = d_m / 1000.0

        if self.prop_model == 'cost231':
            pl_base = self.cost231.path_loss(d_km)
        else:
            pl_base = self.songs.path_loss(d_km)

        dists, elevs = self._sample_profile(x1, y1, x2, y2)

        gw_abs = elevs[0]  + self._h_tx
        rx_abs = elevs[-1] + self._h_rx
        sight  = np.linspace(gw_abs, rx_abs, len(elevs))
        nlos   = bool(np.any(elevs > sight))

        lam     = 3e8 / (self._fc * 1e6)
        r_inf_m = np.sqrt(4 * self._h_tx * self._h_rx / lam)

        if not nlos:
            extra = 0.0
            if d_m <= r_inf_m and len(elevs) > 2:
                if float(np.max(elevs[1:-1])) > gw_abs:
                    extra = 20.0
            l_diff    = 0.0
            pl_fs_val = 0.0
            ld_t      = 0.0
            pl_total  = pl_base + extra
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
            'pl_total': pl_total,
            'pl_base' : pl_base,
            'l_diff'  : l_diff,
            'pl_fs'   : pl_fs_val,
            'ld_t'    : ld_t,
            'nlos'    : nlos,
            'd_km'    : d_km,
            'gw_abs'  : gw_abs,
            'rx_abs'  : rx_abs,
            'dists'   : dists,
            'elevs'   : elevs,
        }