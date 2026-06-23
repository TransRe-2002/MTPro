"""
Map widget using Leaflet.js embedded in QWebEngineView.
Default tile source: 高德卫星图 + 高德注记叠加 (GCJ-02).
"""

from enum import Enum, auto
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtCore import QObject, Signal, Slot

INITIAL_CENTER = (39.9042, 116.4074)
INITIAL_ZOOM = 5

LEAFLET_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  * { margin: 0; padding: 0; }
  html, body, #map { width: 100%; height: 100%; }
  .leaflet-control-layers { display: none; }
</style>
</head>
<body>
<div id="map"></div>
<script>
  var map = L.map('map').setView([LAT, LNG], ZOOM);

  // ---- tile sources ----
  function gdSatellite() {
    return L.tileLayer('https://webst0{s}.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}', {
      subdomains: ['1','2','3','4'], maxZoom: 18,
      attribution: '&copy; 高德'
    });
  }
  function gdLabel() {
    return L.tileLayer('https://webst0{s}.is.autonavi.com/appmaptile?style=8&x={x}&y={y}&z={z}', {
      subdomains: ['1','2','3','4'], maxZoom: 18,
    });
  }
  function gdNormal() {
    return L.tileLayer('https://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}', {
      subdomains: ['1','2','3','4'], maxZoom: 18,
      attribution: '&copy; 高德'
    });
  }
  function tdNormal() {
    return L.tileLayer('https://t{s}.tianditu.gov.cn/vec_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=vec&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&tk=', {
      subdomains: ['0','1','2','3','4','5','6','7'], maxZoom: 18,
      attribution: '&copy; 天地图'
    });
  }

  // ---- layer groups ----
  var satGroup = L.layerGroup();
  var labelGroup = L.layerGroup();
  var baseGroup = L.layerGroup();

  function setSatellite() {
    baseGroup.clearLayers(); labelGroup.clearLayers();
    satGroup.addLayer(gdSatellite());  // dummy, actually add to base
    baseGroup.addLayer(gdSatellite());
    labelGroup.addLayer(gdLabel());
    baseGroup.addTo(map);
    labelGroup.addTo(map);
  }
  function setNormal() {
    baseGroup.clearLayers(); labelGroup.clearLayers();
    baseGroup.addLayer(gdNormal());
    baseGroup.addTo(map);
  }
  function setTerrain() {
    baseGroup.clearLayers(); labelGroup.clearLayers();
    baseGroup.addLayer(tdNormal());
    baseGroup.addTo(map);
  }

  // default: satellite + labels
  setSatellite();

  // ---- markers ----
  var markers = {};

  new QWebChannel(qt.webChannelTransport, function(channel) {
    var h = channel.objects.handler;

    h.addMarker.connect(function(key, lat, lng, label) {
      if (markers[key]) { map.removeLayer(markers[key]); }
      var m = L.marker([lat, lng]).addTo(map);
      if (label) { m.bindPopup(label); }
      markers[key] = m;
      map.setView([lat, lng], map.getZoom());
    });

    h.clearMarkers.connect(function() {
      Object.values(markers).forEach(function(m) { map.removeLayer(m); });
      markers = {};
    });

    h.setView.connect(function(lat, lng, zoom) {
      map.setView([lat, lng], zoom);
    });

    h.switchTile.connect(function(name) {
      if (name === 'satellite') setSatellite();
      else if (name === 'normal') setNormal();
      else if (name === 'terrain') setTerrain();
    });
  });
</script>
</body>
</html>
"""


class TileSource(Enum):
    SATELLITE = auto()   # 高德卫星图 + 注记
    NORMAL = auto()      # 高德矢量地图
    TERRAIN = auto()     # 天地图矢量 (等高线需 TK)

    def _js_name(self):
        return {TileSource.SATELLITE: "satellite",
                TileSource.NORMAL: "normal",
                TileSource.TERRAIN: "terrain"}[self]


class _MapHandler(QObject):
    addMarker = Signal(str, float, float, str)
    clearMarkers = Signal()
    setView = Signal(float, float, int)
    switchTile = Signal(str)

    @Slot(str, float, float, str)
    def add_marker(self, key, lat, lng, label=""):
        self.addMarker.emit(key, lat, lng, label)

    @Slot()
    def clear_markers(self):
        self.clearMarkers.emit()

    @Slot(float, float, int)
    def set_view(self, lat, lng, zoom):
        self.setView.emit(lat, lng, zoom)

    @Slot(str)
    def switch_tile(self, name):
        self.switchTile.emit(name)


class MapWidget(QWidget):
    """Leaflet map widget. Defaults to 高德卫星图 + 注记.

    Coordinate system note: 高德/天地图 use GCJ-02. If your input is
    WGS-84 (GPS raw), coordinates in China will appear offset by 100-700m.
    Convert via ``coord_transform.gcj02_from_wgs84`` or similar before
    calling add_marker().
    """

    def __init__(self, tile_source=TileSource.SATELLITE, parent=None):
        super().__init__(parent)
        self._handler = _MapHandler(self)
        self._counter = 0
        self._current_zoom = INITIAL_ZOOM

        self._view = QWebEngineView(self)
        self._channel = QWebChannel(self._view.page())
        self._channel.registerObject("handler", self._handler)
        self._view.page().setWebChannel(self._channel)

        html = LEAFLET_HTML.replace("LAT", str(INITIAL_CENTER[0])) \
                           .replace("LNG", str(INITIAL_CENTER[1])) \
                           .replace("ZOOM", str(INITIAL_ZOOM))
        self._view.setHtml(html)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

    def add_marker(self, lat, lng, label=""):
        """Add a marker at (lat, lng). Returns key for future reference."""
        self._counter += 1
        key = f"m{self._counter}"
        self._handler.add_marker(key, lat, lng, label)
        return key

    def clear(self):
        self._handler.clear_markers()
        self._counter = 0

    def fly_to(self, lat, lng, zoom=None):
        z = zoom if zoom is not None else self._current_zoom
        self._current_zoom = z
        self._handler.set_view(lat, lng, z)

    def set_tile_source(self, source: TileSource):
        self._handler.switch_tile(source._js_name())


if __name__ == "__main__":
    import sys, random
    from PySide6.QtWidgets import (QApplication, QPushButton, QHBoxLayout,
                                    QVBoxLayout, QComboBox, QWidget)

    app = QApplication(sys.argv)

    window = QWidget()
    window.setWindowTitle("MapWidget 调试 — 高德卫星图")
    window.resize(1000, 700)

    root = QVBoxLayout(window)
    map_widget = MapWidget(TileSource.SATELLITE)
    root.addWidget(map_widget)

    # ---- toolbar ----
    bar = QHBoxLayout()

    src_combo = QComboBox()
    src_combo.addItem("高德卫星图", TileSource.SATELLITE)
    src_combo.addItem("高德矢量图", TileSource.NORMAL)
    src_combo.addItem("天地图矢量", TileSource.TERRAIN)
    src_combo.currentIndexChanged.connect(
        lambda: map_widget.set_tile_source(src_combo.currentData()))

    def add_cities():
        map_widget.add_marker(39.9042, 116.4074, "北京")
        map_widget.add_marker(31.2304, 121.4737, "上海")
        map_widget.add_marker(22.5431, 114.0579, "深圳")

    def add_random():
        lat = random.uniform(18, 53)
        lng = random.uniform(73, 135)
        map_widget.add_marker(lat, lng, f"({lat:.2f}, {lng:.2f})")

    bar.addWidget(src_combo)
    btn1 = QPushButton("北上深")
    btn1.clicked.connect(add_cities)
    btn2 = QPushButton("随机点")
    btn2.clicked.connect(add_random)
    btn3 = QPushButton("清除")
    btn3.clicked.connect(map_widget.clear)
    bar.addWidget(btn1)
    bar.addWidget(btn2)
    bar.addWidget(btn3)
    root.addLayout(bar)

    window.show()
    sys.exit(app.exec())
