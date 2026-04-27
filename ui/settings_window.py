# ui/settings_window.py — 전역 설정 창
from __future__ import annotations
import json, os
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QDoubleSpinBox, QSpinBox, QPushButton, QLabel,
    QGroupBox, QComboBox, QTabWidget, QWidget,
    QCheckBox, QSlider,
)
from PyQt5.QtCore import Qt, pyqtSignal
from ui.dialogs import DARK, PANEL, TEXT, MUTED, BORDER, STYLE_DLG

# 설정 저장 경로
SETTINGS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "settings.json")

# 기본값
DEFAULT_SETTINGS = {
    # 전파 파라미터
    "fc_mhz"       : 915.0,
    "env"          : 2,
    "n_samples"    : 100,
    "diff_order"   : 2,
    # GW 기본값
    "gw_pt_dbm"    : 14.0,
    "gw_gt_dbi"    : 2.15,
    "gw_lt_db"     : 0.0,
    "gw_hb_m"      : 15.0,
    # Node 기본값
    "nd_gr_dbi"    : 2.15,
    "nd_lr_db"     : 0.0,
    "nd_hm_m"      : 1.5,
    "nd_min_rx"    : -126.6,
    # 히트맵
    "heatmap_step" : 0.0015,
    "heatmap_diff" : False,   # True=Deygout 포함, False=Song's만 (빠름)
    # 지도
    "map_tile"     : "CartoDB Voyager",
    # 커버리지 분석
    "cov_n_samples": 100,
}

ENV_LABELS = {1: "Dense Urban", 2: "Urban", 3: "Suburban", 4: "Open"}
MAP_TILES  = [
    "CartoDB Voyager",
    "CartoDB DarkMatter",
    "OpenStreetMap",
    "CartoDB Positron",
    "Stamen Terrain",
]

STYLE = STYLE_DLG + f"""
QTabWidget::pane {{
    border:1px solid {BORDER}; border-radius:6px;
    background:{DARK}; margin-top:-1px;
}}
QTabBar::tab {{
    background:{PANEL}; color:{MUTED};
    border:1px solid {BORDER}; border-bottom:none;
    border-radius:4px 4px 0 0;
    padding:6px 16px; margin-right:2px; font-size:11px;
}}
QTabBar::tab:selected {{ background:{DARK}; color:{TEXT}; }}
QTabBar::tab:hover    {{ color:{TEXT}; }}
QComboBox {{
    background:{PANEL}; color:{TEXT};
    border:1px solid {BORDER}; border-radius:4px;
    padding:4px 8px; min-height:26px;
}}
QComboBox QAbstractItemView {{
    background:{PANEL}; color:{TEXT};
    selection-background-color:#253a5a;
}}
QCheckBox {{ color:{TEXT}; spacing:6px; }}
QCheckBox::indicator {{
    width:16px; height:16px;
    border:1px solid {BORDER}; border-radius:3px;
    background:{PANEL};
}}
QCheckBox::indicator:checked {{
    background:#1d4a1d; border-color:#2a6a2a;
}}
"""


def _dspin(lo, hi, val, dec=1, suf="", step=0.1):
    s = QDoubleSpinBox()
    s.setRange(lo, hi); s.setValue(val)
    s.setDecimals(dec); s.setSuffix(suf)
    s.setSingleStep(step)
    return s

def _ispin(lo, hi, val):
    s = QSpinBox()
    s.setRange(lo, hi); s.setValue(val)
    return s


def load_settings() -> dict:
    """settings.json 로드. 없으면 기본값 반환."""
    try:
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, encoding='utf-8') as f:
                data = json.load(f)
            # 새로 추가된 기본값 병합
            merged = dict(DEFAULT_SETTINGS)
            merged.update(data)
            return merged
    except Exception:
        pass
    return dict(DEFAULT_SETTINGS)


def save_settings(settings: dict):
    """settings.json 저장."""
    try:
        with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[설정] 저장 실패: {e}")


class SettingsWindow(QDialog):
    """전역 설정 창."""
    sig_settings_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("설정")
        self.setStyleSheet(STYLE)
        self.resize(480, 560)
        self.setWindowFlag(Qt.Window)
        self._settings = load_settings()
        self._build()
        self._load_values()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        tabs = QTabWidget()
        lay.addWidget(tabs)

        # ── 탭 1: 전파 파라미터 ──────────────────────────────
        t1 = QWidget(); fl1 = QFormLayout(t1); fl1.setSpacing(10)

        self.sp_fc      = _dspin(400, 2000, 915.0, 1, " MHz", 1.0)
        self.cb_env     = QComboBox()
        for k, v in ENV_LABELS.items():
            self.cb_env.addItem(f"{k} - {v}", k)
        self.sp_nsamp   = _ispin(20, 500, 100)
        self.sp_dorder  = _ispin(1, 3, 2)

        fl1.addRow("반송 주파수", self.sp_fc)
        fl1.addRow("전파 환경", self.cb_env)
        fl1.addRow("DEM 샘플 수", self.sp_nsamp)
        fl1.addRow("Deygout 재귀 깊이", self.sp_dorder)

        note = QLabel(
            "· DEM 샘플 수: 높을수록 정확하나 느림 (권장: 100)\n"
            "· Deygout 깊이: 1=빠름/단순, 2=보통(기본), 3=느림/정밀")
        note.setStyleSheet(f"color:{MUTED};font-size:10px;")
        fl1.addRow("", note)
        tabs.addTab(t1, "📡 전파")

        # ── 탭 2: GW 기본값 ──────────────────────────────────
        t2 = QWidget(); fl2 = QFormLayout(t2); fl2.setSpacing(10)

        self.sp_gw_pt = _dspin(-30, 50,  14.0, 1, " dBm")
        self.sp_gw_gt = _dspin(0,   30,   2.15, 2, " dBi")
        self.sp_gw_lt = _dspin(0,   20,   0.0,  2, " dB")
        self.sp_gw_hb = _dspin(1,   200,  15.0, 1, " m")

        fl2.addRow("송신 출력 Pt",   self.sp_gw_pt)
        fl2.addRow("안테나 이득 Gt", self.sp_gw_gt)
        fl2.addRow("케이블 손실 Lt", self.sp_gw_lt)
        fl2.addRow("안테나 높이 hb", self.sp_gw_hb)

        note2 = QLabel("새로 추가되는 GW에 적용되는 기본값입니다.")
        note2.setStyleSheet(f"color:{MUTED};font-size:10px;")
        fl2.addRow("", note2)
        tabs.addTab(t2, "📡 GW")

        # ── 탭 3: Node 기본값 ────────────────────────────────
        t3 = QWidget(); fl3 = QFormLayout(t3); fl3.setSpacing(10)

        self.sp_nd_gr  = _dspin(0,   30,   2.15, 2, " dBi")
        self.sp_nd_lr  = _dspin(0,   20,   0.0,  2, " dB")
        self.sp_nd_hm  = _dspin(0.1, 50,   1.5,  1, " m")
        self.sp_nd_rxm = _dspin(-160, -50, -126.6, 1, " dBm")

        fl3.addRow("수신 이득 Gr",    self.sp_nd_gr)
        fl3.addRow("수신 손실 Lr",    self.sp_nd_lr)
        fl3.addRow("안테나 높이 hm",  self.sp_nd_hm)
        fl3.addRow("최소 수신 레벨",  self.sp_nd_rxm)

        note3 = QLabel("랜덤 배치 및 새로 추가되는 Node에 적용됩니다.")
        note3.setStyleSheet(f"color:{MUTED};font-size:10px;")
        fl3.addRow("", note3)
        tabs.addTab(t3, "📶 Node")

        # ── 탭 4: 히트맵 & 지도 ─────────────────────────────
        t4 = QWidget(); fl4 = QFormLayout(t4); fl4.setSpacing(10)

        self.sp_hm_step  = _dspin(0.0005, 0.005, 0.0015, 4, "°", 0.0005)
        self.chk_hm_diff = QCheckBox("Deygout 회절 포함 (정확하나 느림)")
        self.cb_tile     = QComboBox()
        for t in MAP_TILES:
            self.cb_tile.addItem(t)

        fl4.addRow("히트맵 격자 간격", self.sp_hm_step)
        fl4.addRow("", self.chk_hm_diff)
        fl4.addRow("지도 배경", self.cb_tile)

        step_note = QLabel(
            "· 0.0010° ≈ 111m/격자 (정밀, 느림)\n"
            "· 0.0015° ≈ 167m/격자 (기본, 빠름)\n"
            "· 0.0020° ≈ 222m/격자 (빠름, 거침)")
        step_note.setStyleSheet(f"color:{MUTED};font-size:10px;")
        fl4.addRow("", step_note)
        tabs.addTab(t4, "🗺 히트맵/지도")

        # ── 하단 버튼 ────────────────────────────────────────
        btn_row = QHBoxLayout()

        btn_reset = QPushButton("기본값 복원")
        btn_reset.setStyleSheet(
            f"QPushButton{{background:{PANEL};color:{MUTED};"
            f"border:1px solid {BORDER};border-radius:5px;"
            f"padding:6px 14px;font-size:11px;}}"
            f"QPushButton:hover{{color:{TEXT};}}")
        btn_reset.clicked.connect(self._reset)

        btn_cancel = QPushButton("취소")
        btn_cancel.setProperty("role", "cancel")
        btn_cancel.clicked.connect(self.reject)

        btn_ok = QPushButton("적용")
        btn_ok.setProperty("role", "ok")
        btn_ok.clicked.connect(self._apply)

        btn_row.addWidget(btn_reset)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_ok)
        lay.addLayout(btn_row)

    def _load_values(self):
        s = self._settings
        self.sp_fc.setValue(s.get("fc_mhz", 915.0))
        env_val = s.get("env", 2)
        for i in range(self.cb_env.count()):
            if self.cb_env.itemData(i) == env_val:
                self.cb_env.setCurrentIndex(i)
                break
        self.sp_nsamp.setValue(s.get("n_samples", 100))
        self.sp_dorder.setValue(s.get("diff_order", 2))

        self.sp_gw_pt.setValue(s.get("gw_pt_dbm", 14.0))
        self.sp_gw_gt.setValue(s.get("gw_gt_dbi", 2.15))
        self.sp_gw_lt.setValue(s.get("gw_lt_db",  0.0))
        self.sp_gw_hb.setValue(s.get("gw_hb_m",   15.0))

        self.sp_nd_gr.setValue(s.get("nd_gr_dbi", 2.15))
        self.sp_nd_lr.setValue(s.get("nd_lr_db",  0.0))
        self.sp_nd_hm.setValue(s.get("nd_hm_m",   1.5))
        self.sp_nd_rxm.setValue(s.get("nd_min_rx", -126.6))

        self.sp_hm_step.setValue(s.get("heatmap_step", 0.0015))
        self.chk_hm_diff.setChecked(s.get("heatmap_diff", False))
        tile = s.get("map_tile", "CartoDB Voyager")
        idx  = self.cb_tile.findText(tile)
        if idx >= 0:
            self.cb_tile.setCurrentIndex(idx)

    def _collect(self) -> dict:
        return {
            "fc_mhz"       : self.sp_fc.value(),
            "env"          : self.cb_env.currentData(),
            "n_samples"    : self.sp_nsamp.value(),
            "diff_order"   : self.sp_dorder.value(),
            "gw_pt_dbm"    : self.sp_gw_pt.value(),
            "gw_gt_dbi"    : self.sp_gw_gt.value(),
            "gw_lt_db"     : self.sp_gw_lt.value(),
            "gw_hb_m"      : self.sp_gw_hb.value(),
            "nd_gr_dbi"    : self.sp_nd_gr.value(),
            "nd_lr_db"     : self.sp_nd_lr.value(),
            "nd_hm_m"      : self.sp_nd_hm.value(),
            "nd_min_rx"    : self.sp_nd_rxm.value(),
            "heatmap_step" : self.sp_hm_step.value(),
            "heatmap_diff" : self.chk_hm_diff.isChecked(),
            "map_tile"     : self.cb_tile.currentText(),
        }

    def _apply(self):
        self._settings = self._collect()
        save_settings(self._settings)
        self.sig_settings_changed.emit(self._settings)
        self.accept()

    def _reset(self):
        self._settings = dict(DEFAULT_SETTINGS)
        self._load_values()

    def get_settings(self) -> dict:
        return dict(self._settings)
