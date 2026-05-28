# ui/map_widget.py
# Folium 기반 지도 위젯
# - GW/Node 마커 표시
# - 격자 히트맵 (GW 주변 전파 세기 분포, 참고용)
# - 커버리지 분석 결과 기반 수신전력 분포 (정확한 커버 시각화)
# - 등고선 / SF 레이어
# - 중첩 커버 / 음영 지역 레이어
# - 거리 측정선
# - 클릭 / 드래그 이벤트 브릿지

import folium, tempfile
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

BOUNDS = (127.02772, 37.33338, 127.19584, 37.47482)

# GW별 고유 색상 팔레트 (Folium 지원 색상명)
GW_COLORS = [
    'red', 'blue', 'green', 'purple', 'orange',
    'darkred', 'darkblue', 'darkgreen', 'darkpurple', 'cadetblue',
    'pink', 'lightblue', 'lightgreen', 'beige', 'black',
]

# 수신전력 → 색상 매핑 (범례와 동일한 기준)
PR_COLOR_LEVELS = [
    (-90,  '#FF2020'),   # 매우 강함 — 빨강
    (-100, '#FF8C00'),   # 강함 — 주황
    (-110, '#FFD700'),   # 보통 — 노랑
    (-120, '#00C94A'),   # 약함 — 초록
    (-999, '#4f8ef7'),   # 매우 약함 (범례 밖) — 파랑
]


def _pr_to_color(pr: float) -> str:
    """수신전력(dBm)을 색상 hex 코드로 변환."""
    for threshold, color in PR_COLOR_LEVELS:
        if pr >= threshold:
            return color
    return '#4f8ef7'


class MapBridge(QObject):
    """JavaScript ↔ Python 브릿지 (클릭/드래그 이벤트 수신)."""
    clicked       = pyqtSignal(float, float)
    right_clicked = pyqtSignal(float, float)
    gw_dragged    = pyqtSignal(str, float, float)
    nd_dragged    = pyqtSignal(str, float, float)

    @pyqtSlot(float, float)
    def mapClicked(self, lon, lat):
        self.clicked.emit(lon, lat)

    @pyqtSlot(float, float)
    def mapRightClicked(self, lon, lat):
        self.right_clicked.emit(lon, lat)

    @pyqtSlot(str, float, float)
    def gwDragged(self, callsign, lon, lat):
        self.gw_dragged.emit(callsign, lon, lat)

    @pyqtSlot(str, float, float)
    def nodeDragged(self, callsign, lon, lat):
        self.nd_dragged.emit(callsign, lon, lat)


class MapWidget(QWidget):
    """Folium 기반 지도 위젯."""
    sig_map_clicked       = pyqtSignal(float, float)
    sig_map_right_clicked = pyqtSignal(float, float)
    sig_gw_dragged        = pyqtSignal(str, float, float)
    sig_nd_dragged        = pyqtSignal(str, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.view = QWebEngineView()
        lay.addWidget(self.view)

        self.channel = QWebChannel()
        self.bridge  = MapBridge()
        self.channel.registerObject("bridge", self.bridge)
        self.view.page().setWebChannel(self.channel)
        self.bridge.clicked.connect(self.sig_map_clicked)
        self.bridge.right_clicked.connect(self.sig_map_right_clicked)
        self.bridge.gw_dragged.connect(self.sig_gw_dragged)
        self.bridge.nd_dragged.connect(self.sig_nd_dragged)

        self.refresh()

    def refresh(self, gws=None, nodes=None, result=None,
                heatmaps=None, selected_gws=None, map_tile=None,
                measure_pts=None):
        """
        지도를 새로 렌더링합니다.

        레이어 순서 (아래 → 위):
          1. 격자 히트맵 이미지 (GW 전파 세기 분포, 참고용)
          2. 등고선 레이어
          3. SF 레이어
          4. 수신전력 분포 (커버리지 분석 결과 기반, 정확한 커버 시각화)
          5. 중첩 커버 영역
          6. 음영 지역
          7. Node 마커
          8. GW 마커
          9. 거리 측정선
        """
        c    = [(BOUNDS[1] + BOUNDS[3]) / 2, (BOUNDS[0] + BOUNDS[2]) / 2]
        tile = map_tile or "CartoDB Voyager"
        m    = folium.Map(location=c, zoom_start=12,
                          tiles=tile, prefer_canvas=True)

        # ── GW별 색상 맵 생성 ────────────────────────────────
        gw_color_map = {}
        if gws:
            active_gws = [g for g in gws if g.enabled]
            for i, gw in enumerate(active_gws):
                gw_color_map[gw.callsign] = GW_COLORS[i % len(GW_COLORS)]

        # ── 거리 측정선 ──────────────────────────────────────
        if measure_pts and len(measure_pts) >= 1:
            import math

            def _haversine(p1, p2):
                R = 6371.0
                la1, lo1 = math.radians(p1[1]), math.radians(p1[0])
                la2, lo2 = math.radians(p2[1]), math.radians(p2[0])
                dlat = la2 - la1; dlon = lo2 - lo1
                a = (math.sin(dlat/2)**2
                     + math.cos(la1)*math.cos(la2)*math.sin(dlon/2)**2)
                return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

            for i, (lon, lat) in enumerate(measure_pts):
                folium.CircleMarker(
                    location=[lat, lon], radius=6,
                    color='#FFD700', fill=True,
                    fill_color='#FFD700', fill_opacity=1.0,
                    tooltip=f"P{i+1} ({lat:.5f}, {lon:.5f})",
                ).add_to(m)

            for i in range(len(measure_pts) - 1):
                p1 = measure_pts[i]; p2 = measure_pts[i+1]
                dist = _haversine(p1, p2)
                folium.PolyLine(
                    locations=[[p1[1], p1[0]], [p2[1], p2[0]]],
                    color='#FFD700', weight=2.5,
                    dash_array='8 4', opacity=0.9,
                ).add_to(m)
                mid_lat = (p1[1] + p2[1]) / 2
                mid_lon = (p1[0] + p2[0]) / 2
                folium.Marker(
                    location=[mid_lat, mid_lon],
                    icon=folium.DivIcon(
                        html=f'''<div style="
                            background:#1e2130cc;color:#FFD700;
                            border:1px solid #FFD700;border-radius:4px;
                            padding:2px 6px;font-size:11px;
                            font-weight:bold;white-space:nowrap;
                            ">{dist:.3f} km</div>''',
                        icon_size=(90, 24), icon_anchor=(45, 12),
                    ),
                ).add_to(m)

        # ── 격자 히트맵 이미지 ───────────────────────────────
        # GW 주변 전파 세기 분포를 면(面) 형태로 표시 (참고용)
        # 정확한 커버/미커버 판단은 아래 '수신전력 분포' 레이어 사용
        if heatmaps:
            for hm in heatmaps:
                hm_type = hm.get('type', '')

                # 격자 히트맵 이미지 표시 (환경 분류 지도 포함)
                if 'url' in hm:
                    lyr_name = (
                        f"{hm['callsign']} 히트맵"
                        if hm_type == 'env_map'
                        else f"{hm['callsign']} 전파 세기 (격자)")
                    lyr = folium.FeatureGroup(name=lyr_name, show=True)
                    folium.raster_layers.ImageOverlay(
                        image=hm['url'],
                        bounds=hm['bounds'],
                        opacity=0.65,        # 반투명 — Node 마커 가리지 않도록
                        interactive=False,
                        cross_origin=False,
                        zindex=2,
                    ).add_to(lyr)
                    lyr.add_to(m)

                # 등고선 레이어 (수신전력 등고선)
                if 'contours' in hm:
                    for cl in hm['contours']:
                        cl_lyr = folium.FeatureGroup(
                            name=f"{hm['callsign']} {cl['label']} 등고선",
                            show=True)
                        for seg in cl['segments']:
                            folium.PolyLine(
                                locations=seg,
                                color=cl['color'],
                                weight=cl['weight'],
                                opacity=0.9,
                                tooltip=cl['label'],
                                dash_array='6 4',
                            ).add_to(cl_lyr)
                        # 등고선 값 라벨
                        for lp in cl.get('label_pts', []):
                            lh = (
                                f'<div style="background:{cl["color"]}22;'
                                f'border:1px solid {cl["color"]};'
                                f'border-radius:4px;padding:1px 5px;'
                                f'font-size:10px;font-weight:bold;'
                                f'color:{cl["color"]};white-space:nowrap;'
                                f'pointer-events:none;">{lp["text"]}</div>')
                            folium.Marker(
                                location=[lp['lat'], lp['lon']],
                                icon=folium.DivIcon(
                                    html=lh, icon_size=(70, 20),
                                    icon_anchor=(35, 10)),
                            ).add_to(cl_lyr)
                        cl_lyr.add_to(m)

                # SF 레이어 (Spreading Factor별 커버리지 경계)
                if 'sf_layers' in hm:
                    for sl in hm['sf_layers']:
                        sf_lyr = folium.FeatureGroup(
                            name=f"{hm['callsign']} {sl['label']}",
                            show=False)
                        for seg in sl['segments']:
                            folium.PolyLine(
                                locations=seg,
                                color=sl['color'],
                                weight=2.5,
                                opacity=0.85,
                                tooltip=sl['label'],
                                dash_array='8 4',
                            ).add_to(sf_lyr)
                        sf_lyr.add_to(m)

        # ── 커버리지 분석 결과 기반 시각화 ───────────────────
        # 격자 히트맵과 달리 커버리지 분석(run())의 정확한 결과를 사용
        # → Node 마커 색상과 100% 일치 보장
        if result and nodes:

            # 수신전력 분포 레이어
            # 커버된 Node 위치에 수신전력에 비례하는 색상 원을 그림
            cov_hm_lyr = folium.FeatureGroup(
                name="수신전력 분포 (분석 결과)", show=True)

            for ni, nd in enumerate(nodes):
                if ni >= len(result.nodes):
                    break
                info = result.nodes[ni]
                if not info.covered:
                    continue

                pr    = info.best_pr
                color = _pr_to_color(pr)
                tip   = (f"{nd.callsign} | Pr={pr:.1f}dBm | "
                         f"연결 GW: {info.best_gw or '없음'} | "
                         f"수신 GW: {info.n_rx_gw}개")

                # 반경: 수신전력에 비례 (강할수록 크게, 14~22px)
                radius = max(14, min(22, int(14 + (pr + 120) / 5)))

                folium.CircleMarker(
                    location=[nd.lat, nd.lon],
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.40,
                    weight=0,
                    tooltip=tip,
                ).add_to(cov_hm_lyr)

            cov_hm_lyr.add_to(m)

            # 중첩 커버 레이어
            # 2개 이상 GW에 커버되는 Node를 보라색 원으로 표시
            ovlp_lyr = folium.FeatureGroup(
                name="중첩 커버 영역", show=False)
            for ni, nd in enumerate(nodes):
                if ni >= len(result.nodes):
                    break
                info = result.nodes[ni]
                if info.covered and info.n_rx_gw >= 2:
                    # 수신 GW 수에 비례하여 원 크기 증가
                    radius = 18 + (info.n_rx_gw - 2) * 6
                    tip    = (f"{nd.callsign} | 중첩 커버 | "
                              f"수신 GW: {info.n_rx_gw}개 | "
                              f"Pr={info.best_pr:.1f}dBm")
                    folium.CircleMarker(
                        location=[nd.lat, nd.lon],
                        radius=radius,
                        color='#9B59B6',
                        fill=True,
                        fill_color='#9B59B6',
                        fill_opacity=0.25,
                        weight=1.5,
                        tooltip=tip,
                    ).add_to(ovlp_lyr)
            ovlp_lyr.add_to(m)

            # 음영 지역 레이어
            # 미커버 Node를 빨간 원으로 표시
            shadow_lyr = folium.FeatureGroup(
                name="음영 지역 (미커버)", show=False)
            for ni, nd in enumerate(nodes):
                if ni >= len(result.nodes):
                    break
                info = result.nodes[ni]
                if not info.covered:
                    tip = (f"{nd.callsign} | ✗ 미커버 | "
                           f"최대 Pr={info.best_pr:.1f}dBm")
                    folium.CircleMarker(
                        location=[nd.lat, nd.lon],
                        radius=10,
                        color='#FF4444',
                        fill=True,
                        fill_color='#FF4444',
                        fill_opacity=0.30,
                        weight=1.5,
                        tooltip=tip,
                    ).add_to(shadow_lyr)
            shadow_lyr.add_to(m)

        # ── Node 마커 ────────────────────────────────────────
        if nodes:
            nd_lyr = folium.FeatureGroup(name="Nodes", show=True)
            for ni, nd in enumerate(nodes):
                if result and ni < len(result.nodes):
                    info = result.nodes[ni]
                    cov  = info.covered
                    pr   = info.best_pr
                    n_rx = info.n_rx_gw
                    tip  = (f"{nd.callsign} | "
                            f"{'✓ 커버' if cov else '✗ 미커버'} | "
                            f"최대 Pr={pr:.1f}dBm | "
                            f"연결 GW: {info.best_gw or '없음'} "
                            f"({n_rx}개 수신)")
                    marker_color = (gw_color_map.get(info.best_gw, 'gray')
                                    if cov and info.best_gw else 'gray')
                else:
                    marker_color = 'gray'
                    tip          = nd.callsign

                folium.Marker(
                    location=[nd.lat, nd.lon],
                    tooltip=tip,
                    icon=folium.Icon(
                        color=marker_color,
                        icon_color='white',
                        icon='mobile',
                        prefix='fa',
                    ),
                    draggable=True,
                ).add_to(nd_lyr)
            nd_lyr.add_to(m)

        # ── GW 마커 ──────────────────────────────────────────
        if gws:
            gw_lyr = folium.FeatureGroup(name="Gateway", show=True)
            for gw in gws:
                if not gw.enabled:
                    continue
                marker_color = gw_color_map.get(gw.callsign, 'gray')
                cnt = result.gw_counts.get(gw.callsign, 0) if result else 0
                tip = (f"{gw.callsign} | "
                       f"Pt={gw.pt_dbm}dBm Gt={gw.gt_dbi}dBi "
                       f"h={gw.hb_m}m | 담당 Node: {cnt}개")
                folium.Marker(
                    location=[gw.lat, gw.lon],
                    tooltip=tip,
                    icon=folium.Icon(
                        color=marker_color,
                        icon_color='white',
                        icon='broadcast-tower',
                        prefix='fa',
                    ),
                    draggable=True,
                ).add_to(gw_lyr)
            gw_lyr.add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)

        # ── JavaScript 브릿지 ────────────────────────────────
        map_name = m.get_name()
        m.get_root().html.add_child(folium.Element("""
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<script>
var _bridge = null;
new QWebChannel(qt.webChannelTransport, function(ch){
    _bridge = ch.objects.bridge;
});
</script>"""))

        m.get_root().script.add_child(folium.Element(f"""
(function waitMap(){{
    var mapObj = window['{map_name}'];
    if(!mapObj){{ setTimeout(waitMap, 100); return; }}

    mapObj.on('click', function(e){{
        if(_bridge) _bridge.mapClicked(e.latlng.lng, e.latlng.lat);
    }});

    mapObj.on('contextmenu', function(e){{
        L.DomEvent.preventDefault(e);
        L.DomEvent.stopPropagation(e);
        if(_bridge) _bridge.mapRightClicked(e.latlng.lng, e.latlng.lat);
    }});

    mapObj.eachLayer(function(layer){{
        if(layer instanceof L.Marker && layer.options.draggable){{
            layer.on('dragend', function(e){{
                var ll      = e.target.getLatLng();
                var tip     = e.target.getTooltip();
                if(!tip) return;
                var content = tip.getContent();
                var text    = content.replace(/<[^>]*>/g, '').trim();
                var cs      = text.split(' | ')[0].trim();
                if(content.indexOf('Pt=') !== -1){{
                    if(_bridge) _bridge.gwDragged(cs, ll.lng, ll.lat);
                }} else {{
                    if(_bridge) _bridge.nodeDragged(cs, ll.lng, ll.lat);
                }}
            }});
        }}
    }});
}})();"""))

        tmp = tempfile.NamedTemporaryFile(
            suffix='.html', delete=False, mode='w', encoding='utf-8')
        m.save(tmp.name)
        from PyQt5.QtCore import QUrl
        self.view.setUrl(QUrl.fromLocalFile(tmp.name))