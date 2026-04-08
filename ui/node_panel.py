# ui/node_panel.py  — Node 관리 패널
import csv
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QFileDialog, QLabel,
)
from PyQt5.QtCore import Qt
from core.coverage import NodeEntry

COLS  = ['Callsign','경도','위도','Gr(dBi)','Lr(dB)','높이(m)','최소수신(dBm)']
STYLE = ("QTableWidget{background:#1c1f26;color:#e0e4ef;"
         "gridline-color:#2a2f3b;}"
         "QHeaderView::section{background:#252930;color:#a0a8be;"
         "border:none;padding:4px;}")
BTN   = ("QPushButton{background:#1a2a1a;color:#7ae87a;"
         "border:1px solid #2a5a2a;border-radius:4px;"
         "padding:3px 8px;font-size:11px;}"
         "QPushButton:hover{background:#255a25;}")


class NodePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4,4,4,4); lay.setSpacing(4)
        lbl = QLabel("Node(단말) 목록")
        lbl.setStyleSheet("color:#a0a8be;font-size:12px;font-weight:bold;")
        lay.addWidget(lbl)

        self.tbl = QTableWidget(0, len(COLS))
        self.tbl.setHorizontalHeaderLabels(COLS)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setEditTriggers(QAbstractItemView.DoubleClicked)
        self.tbl.setStyleSheet(STYLE)
        lay.addWidget(self.tbl)

        btn_l = QHBoxLayout()
        for txt, fn in [("+ 추가", self.add_default),
                         ("- 삭제", self.del_selected),
                         ("CSV 가져오기", self.import_csv),
                         ("CSV 내보내기", self.export_csv)]:
            b = QPushButton(txt); b.setStyleSheet(BTN)
            b.clicked.connect(fn); btn_l.addWidget(b)
        lay.addLayout(btn_l)

    def add_node(self, nd: NodeEntry):
        r = self.tbl.rowCount(); self.tbl.insertRow(r)
        for c,v in enumerate([nd.callsign, f"{nd.lon:.6f}", f"{nd.lat:.6f}",
                               f"{nd.gr_dbi:.2f}", f"{nd.lr_db:.2f}",
                               f"{nd.hm_m:.1f}", f"{nd.min_rx_dbm:.1f}"]):
            self.tbl.setItem(r, c, QTableWidgetItem(str(v)))

    def add_default(self):
        n = self.tbl.rowCount()+1
        self.add_node(NodeEntry(callsign=f"Node{n}",lon=127.10,lat=37.40))

    def del_selected(self):
        rows = sorted({i.row() for i in self.tbl.selectedItems()},reverse=True)
        for r in rows: self.tbl.removeRow(r)

    def get_nodes(self) -> list[NodeEntry]:
        out=[]
        for r in range(self.tbl.rowCount()):
            try:
                out.append(NodeEntry(
                    callsign=self.tbl.item(r,0).text(),
                    lon=float(self.tbl.item(r,1).text()),
                    lat=float(self.tbl.item(r,2).text()),
                    gr_dbi=float(self.tbl.item(r,3).text()),
                    lr_db=float(self.tbl.item(r,4).text()),
                    hm_m=float(self.tbl.item(r,5).text()),
                    min_rx_dbm=float(self.tbl.item(r,6).text()),
                ))
            except Exception: continue
        return out

    def set_coord(self, lon: float, lat: float):
        rows = list({i.row() for i in self.tbl.selectedItems()})
        if not rows: return
        r=rows[0]
        self.tbl.setItem(r,1,QTableWidgetItem(f"{lon:.6f}"))
        self.tbl.setItem(r,2,QTableWidgetItem(f"{lat:.6f}"))

    def import_csv(self):
        path,_=QFileDialog.getOpenFileName(self,"Node CSV","","CSV (*.csv)")
        if not path: return
        with open(path,newline='',encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                try:
                    self.add_node(NodeEntry(
                        callsign=row.get('callsign','Node'),
                        lon=float(row.get('lon',row.get('longitude',0))),
                        lat=float(row.get('lat',row.get('latitude',0))),
                        gr_dbi=float(row.get('gr_dbi',2.15)),
                        lr_db=float(row.get('lr_db',0)),
                        hm_m=float(row.get('hm_m',1.5)),
                        min_rx_dbm=float(row.get('min_rx_dbm',-126.6)),
                    ))
                except Exception: continue

    def export_csv(self):
        path,_=QFileDialog.getSaveFileName(self,"Node CSV","node.csv","CSV (*.csv)")
        if not path: return
        with open(path,'w',newline='',encoding='utf-8-sig') as f:
            w=csv.writer(f)
            w.writerow(['callsign','lon','lat','gr_dbi','lr_db','hm_m','min_rx_dbm'])
            for n in self.get_nodes():
                w.writerow([n.callsign,n.lon,n.lat,n.gr_dbi,n.lr_db,n.hm_m,n.min_rx_dbm])
