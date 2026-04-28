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
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def bearing(lon1: float, lat1: float,
            lon2: float, lat2: float) -> float:
    """방위각 (0=북, 시계방향, 도 단위)."""
    phi1 = np.radians(lat1);  phi2 = np.radians(lat2)
    dlam = np.radians(lon2 - lon1)
    x    = np.sin(dlam) * np.cos(phi2)
    y    = np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(dlam)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360