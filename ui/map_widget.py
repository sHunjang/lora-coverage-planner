# ui/map_widget.py
import folium, tempfile
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

BOUNDS = (127.02772, 37.33338, 127.19584, 37.47482)


class MapBridge(QObject):
    clicked    = pyqtSignal(float, float)
    gw_dragged = pyqtSignal(str, float, float)
    nd_dragged = pyqtSignal(str, float, float)

    @pyqtSlot(float, float)
    def mapClicked(self, lon, lat):
        self.clicked.emit(lon, lat)

    @pyqtSlot(str, float, float)
    def gwDragged(self, callsign, lon, lat):
        self.gw_dragged.emit(callsign, lon, lat)

    @pyqtSlot(str, float, float)
    def nodeDragged(self, callsign, lon, lat):
        self.nd_dragged.emit(callsign, lon, lat)


class MapWidget(QWidget):
    sig_map_clicked = pyqtSignal(float, float)
    sig_gw_dragged  = pyqtSignal(str, float, float)
    sig_nd_dragged  = pyqtSignal(str, float, float)

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
        self.bridge.gw_dragged.connect(self.sig_gw_dragged)
        self.bridge.nd_dragged.connect(self.sig_nd_dragged)

        self.refresh()

    def refresh(self, gws=None, nodes=None, result=None,
                heatmaps=None, selected_gws=None):

        c = [(BOUNDS[1] + BOUNDS[3]) / 2, (BOUNDS[0] + BOUNDS[2]) / 2]
        m = folium.Map(location=c, zoom_start=12,
                       tiles="CartoDB Voyager", prefer_canvas=True)

        # ── 히트맵 ──────────────────────────────────────────
        if heatmaps:
            for hm in heatmaps:
                lyr = folium.FeatureGroup(
                    name=f"{hm['callsign']} 히트맵", show=True)
                folium.raster_layers.ImageOverlay(
                    image=hm['url'], bounds=hm['bounds'],
                    opacity=1.0, interactive=False,
                    cross_origin=False, zindex=2,
                ).add_to(lyr)
                lyr.add_to(m)

                # 등고선
                if 'contours' in hm:
                    for cl in hm['contours']:
                        cl_lyr = folium.FeatureGroup(
                            name=f"{hm['callsign']} {cl['label']} 등고선",
                            show=True)
                        for seg in cl['segments']:
                            folium.PolyLine(
                                locations=seg, color=cl['color'],
                                weight=cl['weight'], opacity=0.9,
                                tooltip=cl['label'], dash_array='6 4',
                            ).add_to(cl_lyr)
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

                # SF별 커버리지 레이어
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

        # ── Node 마커 ────────────────────────────────────────
        if nodes:
            nd_lyr = folium.FeatureGroup(name="Nodes", show=True)
            for ni, nd in enumerate(nodes):
                if result and ni < len(result.nodes):
                    info = result.nodes[ni]
                    cov  = info.covered
                    pr   = info.best_pr
                    tip  = (f"{nd.callsign} | "
                            f"{'✓ 커버' if cov else '✗ 미커버'} | "
                            f"최대 Pr={pr:.1f}dBm")
                    marker_color = 'green' if cov else 'red'
                else:
                    marker_color = 'blue'
                    tip = nd.callsign
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
            try:
                sel = set(selected_gws) if selected_gws else set()
            except TypeError:
                sel = set()
            for gw in gws:
                if not gw.enabled:
                    continue
                is_sel = gw.callsign in sel
                is_opt = gw.callsign.startswith("OPT-")
                if is_opt:
                    icon_color   = '#00FF88'
                    marker_color = 'darkgreen'
                elif is_sel:
                    icon_color   = '#FFD700'
                    marker_color = 'orange'
                else:
                    icon_color   = 'white'
                    marker_color = 'gray'
                tip = (f"{gw.callsign} | "
                       f"Pt={gw.pt_dbm}dBm Gt={gw.gt_dbi}dBi "
                       f"h={gw.hb_m}m")
                folium.Marker(
                    location=[gw.lat, gw.lon],
                    tooltip=tip,
                    icon=folium.Icon(
                        color=marker_color,
                        icon_color=icon_color,
                        icon='signal',
                        prefix='fa',
                    ),
                    draggable=True,
                ).add_to(gw_lyr)
            gw_lyr.add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)

        # ── 클릭/드래그 스크립트 ─────────────────────────────
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

    mapObj.eachLayer(function(layer){{
        if(layer instanceof L.Marker && layer.options.draggable){{
            layer.on('dragend', function(e){{
                var ll  = e.target.getLatLng();
                var tip = e.target.getTooltip();
                if(!tip) return;
                var content = tip.getContent();
                // HTML 태그 제거 후 첫 번째 ' | ' 앞부분 추출
                var text = content.replace(/<[^>]*>/g, '').trim();
                var cs = text.split(' | ')[0].trim();
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