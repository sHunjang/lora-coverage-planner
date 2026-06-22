# core/data_validator.py — 데이터 정합성 검증
"""
GW/Node 데이터 검증 모듈
────────────────────────────────────────────────────────────
프로그램 운영 중 GW·Node 목록에 누적될 수 있는 문제를 점검합니다.
UI와 분리된 순수 로직이라 어디서든 재사용 가능합니다
(예: 커버리지 분석 직전 자동 점검, 또는 별도 '데이터 검증' 메뉴).

검증 항목
  1. 좌표 범위 — 위경도가 지구 좌표 범위(±90/±180)를 벗어났는지
  2. DEM 영역 밖 — GW/Node가 로드된 DEM(지형) 범위 밖에 있는지
  3. 중복 좌표 — 같은 위치에 GW 또는 Node가 중복 등록되었는지
  4. 중복 Callsign — 이름이 같은 GW 또는 Node가 있는지
  5. 비정상 파라미터 — 송신출력·안테나 높이 등이 상식 밖 범위인지
  6. 거리 이상치 — GW-Node 거리가 비정상적으로 짧거나(0m) 너무 먼지
────────────────────────────────────────────────────────────
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math


@dataclass
class ValidationIssue:
    level: str       # 'ERROR' | 'WARN'
    category: str     # 검증 항목 분류
    message: str
    target: str = ""  # 관련 GW/Node callsign (있으면)


def _haversine_m(lon1, lat1, lon2, lat2) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (math.sin(dphi/2)**2
         + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def validate(gws, nodes, spatial=None) -> list[ValidationIssue]:
    """
    GW/Node 목록을 검증하여 ValidationIssue 리스트를 반환.

    Parameters
    ----------
    gws     : list[GWEntry]
    nodes   : list[NodeEntry]
    spatial : SpatialData | None — DEM 범위 검사에 사용 (없으면 생략)
    """
    issues: list[ValidationIssue] = []

    # ── 1. 좌표 범위 ─────────────────────────────────────────
    for g in gws:
        if not (-90 <= g.lat <= 90) or not (-180 <= g.lon <= 180):
            issues.append(ValidationIssue(
                'ERROR', '좌표 범위',
                f"GW '{g.callsign}' 좌표가 유효 범위를 벗어남 "
                f"(lon={g.lon}, lat={g.lat})", g.callsign))
    for n in nodes:
        if not (-90 <= n.lat <= 90) or not (-180 <= n.lon <= 180):
            issues.append(ValidationIssue(
                'ERROR', '좌표 범위',
                f"Node '{n.callsign}' 좌표가 유효 범위를 벗어남 "
                f"(lon={n.lon}, lat={n.lat})", n.callsign))

    # ── 2. DEM 영역 밖 ───────────────────────────────────────
    if spatial is not None and getattr(spatial, 'bounds', None) is not None:
        b = spatial.bounds  # (lon_min, lat_min, lon_max, lat_max)
        for g in gws:
            if not (b[0] <= g.lon <= b[2] and b[1] <= g.lat <= b[3]):
                issues.append(ValidationIssue(
                    'WARN', 'DEM 영역 밖',
                    f"GW '{g.callsign}'이 로드된 DEM 영역 밖에 있음 "
                    f"— 지형 계산이 부정확할 수 있음", g.callsign))
        for n in nodes:
            if not (b[0] <= n.lon <= b[2] and b[1] <= n.lat <= b[3]):
                issues.append(ValidationIssue(
                    'WARN', 'DEM 영역 밖',
                    f"Node '{n.callsign}'이 로드된 DEM 영역 밖에 있음 "
                    f"— 지형 계산이 부정확할 수 있음", n.callsign))

    # ── 3. 중복 좌표 ─────────────────────────────────────────
    def _dup_coords(entries, label):
        seen = {}
        for e in entries:
            key = (round(e.lon, 6), round(e.lat, 6))
            seen.setdefault(key, []).append(e.callsign)
        for key, names in seen.items():
            if len(names) > 1:
                issues.append(ValidationIssue(
                    'WARN', '중복 좌표',
                    f"{label} {len(names)}개가 동일 좌표에 위치함: "
                    f"{', '.join(names)}", names[0]))
    _dup_coords(gws, "GW")
    _dup_coords(nodes, "Node")

    # ── 4. 중복 Callsign ─────────────────────────────────────
    def _dup_names(entries, label):
        seen = {}
        for e in entries:
            seen.setdefault(e.callsign, 0)
            seen[e.callsign] += 1
        for name, cnt in seen.items():
            if cnt > 1:
                issues.append(ValidationIssue(
                    'ERROR', '중복 이름',
                    f"{label} Callsign '{name}'이 {cnt}개 중복됨 — "
                    f"분석 결과 매칭 오류 유발 가능", name))
    _dup_names(gws, "GW")
    _dup_names(nodes, "Node")

    # ── 5. 비정상 파라미터 ───────────────────────────────────
    for g in gws:
        if not (-10 <= g.pt_dbm <= 30):
            issues.append(ValidationIssue(
                'WARN', '파라미터 범위',
                f"GW '{g.callsign}' 송신출력이 일반 범위(-10~30dBm) 밖: "
                f"{g.pt_dbm}dBm", g.callsign))
        if not (1 <= g.hb_m <= 200):
            issues.append(ValidationIssue(
                'WARN', '파라미터 범위',
                f"GW '{g.callsign}' 안테나 높이가 일반 범위(1~200m) 밖: "
                f"{g.hb_m}m", g.callsign))
    for n in nodes:
        if not (0.1 <= n.hm_m <= 50):
            issues.append(ValidationIssue(
                'WARN', '파라미터 범위',
                f"Node '{n.callsign}' 안테나 높이가 일반 범위(0.1~50m) 밖: "
                f"{n.hm_m}m", n.callsign))

    # ── 6. 거리 이상치 ───────────────────────────────────────
    for g in gws:
        for n in nodes:
            d = _haversine_m(g.lon, g.lat, n.lon, n.lat)
            if d < 0.5:
                issues.append(ValidationIssue(
                    'WARN', '거리 이상치',
                    f"GW '{g.callsign}'와 Node '{n.callsign}'가 "
                    f"사실상 같은 위치({d:.2f}m)", n.callsign))

    return issues


def summarize(issues: list[ValidationIssue]) -> dict:
    n_err  = sum(1 for i in issues if i.level == 'ERROR')
    n_warn = sum(1 for i in issues if i.level == 'WARN')
    return {'n_error': n_err, 'n_warn': n_warn, 'n_total': len(issues)}
