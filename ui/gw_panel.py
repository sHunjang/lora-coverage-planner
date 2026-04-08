# ui/gw_panel.py  — GW 관리 패널
import csv
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QFileDialog, QLabel, QCheckBox,
)
from PyQt5.QtCore import Qt, pyqtSignal
from core.coverage import GWEntry

COLS = ['선택','활성','Callsign','경도','위도','Pt(dBm)','Gt(dBi)','Lt(dB)','높이(m)']
STYLE_TBL = ("QTableWidget{background:#1c1f26;color:#e0e4ef;"
             "gridline-color:#2a2f3b;}"
             "QHeaderView::section{background:#252930;color:#a0a8be;"
             "border:none;padding:4px;}")
STYLE_BTN  = ("QPushButton{background:#1c2a3a;color:#7ab8e8;"
              "border:1px solid #2a4a6a;border-radius:4px;"
              "padding:3px 8px;font-size:11px;}"
              "QPushButton:hover{background:#254d78;}")


class GWPanel(QWidget):
    sig_selection_changed = pyqtSignal(list)   # 선택된 callsign 목록
    sig_heatmap_request   = pyqtSignal(object) # GWEntry

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4,4,4,4)
        lay.setSpacing(4)

        lbl = QLabel("Gateway 목록")
        lbl.setStyleSheet("color:#a0a8be;font-size:12px;font-weight:bold;")
        lay.addWidget(lbl)

        self.tbl = QTableWidget(0, len(COLS))
        self.tbl.setHorizontalHeaderLabels(COLS)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setEditTriggers(QAbstractItemView.DoubleClicked)
        self.tbl.setStyleSheet(STYLE_TBL)
        self.tbl.itemChanged.connect(self._on_changed)
        lay.addWidget(self.tbl)

        btn_l = QHBoxLayout()
        for txt, fn in [("+ 추가", self.add_default),
                         ("- 삭제", self.del_selected),
                         ("CSV 가져오기", self.import_csv),
                         ("CSV 내보내기", self.export_csv),
                         ("히트맵", self.request_heatmap)]:
            b = QPushButton(txt)
            b.setStyleSheet(STYLE_BTN)
            b.clicked.connect(fn)
            btn_l.addWidget(b)
        lay.addLayout(btn_l)

    def _make_row(self, gw: GWEntry, row: int):
        self.tbl.blockSignals(True)
        # 선택 체크박스
        sel = QTableWidgetItem()
        sel.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        sel.setCheckState(Qt.Unchecked)
        self.tbl.setItem(row, 0, sel)
        # 활성 체크박스
        act = QTableWidgetItem()
        act.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        act.setCheckState(Qt.Checked if gw.enabled else Qt.Unchecked)
        self.tbl.setItem(row, 1, act)
        for c, v in enumerate([gw.callsign, f"{gw.lon:.6f}",
                                f"{gw.lat:.6f}", f"{gw.pt_dbm:.1f}",
                                f"{gw.gt_dbi:.2f}", f"{gw.lt_db:.2f}",
                                f"{gw.hb_m:.1f}"], 2):
            self.tbl.setItem(row, c, QTableWidgetItem(str(v)))
        self.tbl.blockSignals(False)

    def add_gw(self, gw: GWEntry):
        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        self._make_row(gw, r)

    def add_default(self):
        n = self.tbl.rowCount() + 1
        self.add_gw(GWEntry(callsign=f"GW{n}", lon=127.10, lat=37.40))

    def del_selected(self):
        rows = sorted({i.row() for i in self.tbl.selectedItems()}, reverse=True)
        for r in rows: self.tbl.removeRow(r)

    def get_gws(self) -> list[GWEntry]:
        out = []
        for r in range(self.tbl.rowCount()):
            try:
                out.append(GWEntry(
                    enabled  = self.tbl.item(r,1).checkState()==Qt.Checked,
                    callsign = self.tbl.item(r,2).text(),
                    lon      = float(self.tbl.item(r,3).text()),
                    lat      = float(self.tbl.item(r,4).text()),
                    pt_dbm   = float(self.tbl.item(r,5).text()),
                    gt_dbi   = float(self.tbl.item(r,6).text()),
                    lt_db    = float(self.tbl.item(r,7).text()),
                    hb_m     = float(self.tbl.item(r,8).text()),
                ))
            except Exception: continue
        return out

    def get_selected_callsigns(self) -> list[str]:
        out = []
        for r in range(self.tbl.rowCount()):
            it = self.tbl.item(r,0)
            if it and it.checkState()==Qt.Checked:
                cs = self.tbl.item(r,2)
                if cs: out.append(cs.text())
        return out

    def set_coord(self, lon: float, lat: float):
        """선택된 행의 경도/위도 설정 (지도 클릭 시 호출)."""
        rows = list({i.row() for i in self.tbl.selectedItems()})
        if not rows: return
        r = rows[0]
        self.tbl.blockSignals(True)
        self.tbl.setItem(r, 3, QTableWidgetItem(f"{lon:.6f}"))
        self.tbl.setItem(r, 4, QTableWidgetItem(f"{lat:.6f}"))
        self.tbl.blockSignals(False)

    def _on_changed(self, item):
        if item.column() == 0:
            self.sig_selection_changed.emit(self.get_selected_callsigns())

    def request_heatmap(self):
        rows = list({i.row() for i in self.tbl.selectedItems()})
        if not rows: return
        gws = self.get_gws()
        r = rows[0]
        if r < len(gws):
            self.sig_heatmap_request.emit(gws[r])

    def import_csv(self):
        path,_ = QFileDialog.getOpenFileName(self,"GW CSV","","CSV (*.csv)")
        if not path: return
        with open(path, newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                try:
                    self.add_gw(GWEntry(
                        callsign=row.get('callsign','GW'),
                        lon=float(row.get('lon',row.get('longitude',0))),
                        lat=float(row.get('lat',row.get('latitude',0))),
                        pt_dbm=float(row.get('pt_dbm',14)),
                        gt_dbi=float(row.get('gt_dbi',2.15)),
                        lt_db=float(row.get('lt_db',0)),
                        hb_m=float(row.get('hb_m',15)),
                    ))
                except Exception: continue

    def export_csv(self):
        path,_ = QFileDialog.getSaveFileName(self,"GW CSV","gw.csv","CSV (*.csv)")
        if not path: return
        with open(path,'w',newline='',encoding='utf-8-sig') as f:
            w=csv.writer(f)
            w.writerow(['callsign','lon','lat','pt_dbm','gt_dbi','lt_db','hb_m'])
            for g in self.get_gws():
                w.writerow([g.callsign,g.lon,g.lat,g.pt_dbm,g.gt_dbi,g.lt_db,g.hb_m])
