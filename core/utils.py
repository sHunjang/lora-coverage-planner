# core/utils.py — 공통 유틸리티 함수
from __future__ import annotations
import numpy as np


def haversine(lon1: float, lat1: float,
              lon2: float, lat2: float) -> float:
    """두 위경도 간 지표면 거리 (km). Haversine 공식."""
    R    = 6371.0
    phi1 = np.radians(lat1);  phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a    = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return float(R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def bearing(lon1: float, lat1: float,
            lon2: float, lat2: float) -> float:
    """방위각 (0=북, 시계방향, 도 단위)."""
    phi1 = np.radians(lat1);  phi2 = np.radians(lat2)
    dlam = np.radians(lon2 - lon1)
    x    = np.sin(dlam) * np.cos(phi2)
    y    = np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(dlam)
    return float((np.degrees(np.arctan2(x, y)) + 360) % 360)


def pr_to_sf(pr_dbm: float) -> int:
    """
    수신전력(dBm)으로 최적 ADR SF 결정.
    낮은 SF = 빠른 전송 (SF7이 가장 빠름).
    """
    SF_SENS = {
        7: -123.0, 8: -126.0, 9: -129.0,
        10: -132.0, 11: -134.5, 12: -137.0,
    }
    for sf in sorted(SF_SENS.keys()):
        if pr_dbm >= SF_SENS[sf]:
            return sf
    return 12


def toa_ms(sf: int, payload_bytes: int = 20,
           bw_khz: float = 125.0, cr: int = 1) -> float:
    """
    LoRa 패킷 ToA (Time on Air) 계산 (ms).

    Args:
        sf           : Spreading Factor (7~12)
        payload_bytes: 페이로드 크기 (바이트)
        bw_khz       : 대역폭 (kHz, 기본 125kHz)
        cr           : 코딩률 (1=4/5, 2=4/6, 3=4/7, 4=4/8)
    """
    # Preamble: 8 심볼
    t_sym   = (2 ** sf) / (bw_khz * 1000.0) * 1000.0  # ms/심볼
    t_pre   = (8 + 4.25) * t_sym                         # ms

    # Payload 심볼 수
    n_pay   = max(
        8,
        8 + np.ceil(
            max(4 * payload_bytes - 4 * sf + 28 + 16, 0)
            / (4 * sf)
        ) * (cr + 4)
    )
    t_pay   = n_pay * t_sym
    return round(float(t_pre + t_pay), 2)


def eirp_dbm(pt_dbm: float, gt_dbi: float, lt_db: float) -> float:
    """EIRP 계산 (dBm)."""
    return pt_dbm + gt_dbi - lt_db


def link_budget(pt_dbm: float, gt_dbi: float, lt_db: float,
                pl_db: float, gr_dbi: float, lr_db: float,
                indoor_db: float = 0.0) -> float:
    """링크 버짓 기반 수신전력 계산 (dBm)."""
    return pt_dbm + gt_dbi - lt_db - pl_db + gr_dbi - lr_db - indoor_db