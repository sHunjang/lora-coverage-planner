# ui/perf_monitor_window.py — 성능 모니터링 창
from __future__ import annotations
import os
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QGridLayout,
)
from PyQt5.QtCore import Qt, QTimer
from ui.dialogs import DARK, PANEL, TEXT, MUTED, BORDER, STYLE_DLG

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

STYLE = STYLE_DLG + f"""
QGroupBox {{
    color:{MUTED}; border:1px solid {BORDER}; border-radius:8px;
    margin-top:10px; padding-top:14px; font-size:11px;
}}
QGroupBox::title {{ subcontrol-origin:margin; left:10px; }}
"""

BAR_BG = "#252930"


class _Bar(QLabel):
    """간단한 가로 막대 그래프 (퍼센트 표시)."""
    def __init__(self, color="#4f8ef7"):
        super().__init__()
        self._pct = 0.0
        self._color = color
        self.setFixedHeight(14)
        self._update_style()

    def set_pct(self, pct: float, color: str | None = None):
        self._pct = max(0.0, min(100.0, pct))
        if color:
            self._color = color
        self._update_style()

    def _update_style(self):
        self.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {self._color}, stop:{max(self._pct/100,0.001):.3f} {self._color}, "
            f"stop:{min(self._pct/100+0.001,1.0):.3f} {BAR_BG}, stop:1 {BAR_BG});"
            f"border-radius:6px; border:1px solid {BORDER};")


def _val(text, size=20, bold=True, color=None):
    lbl = QLabel(text)
    c = color or TEXT
    w = "bold" if bold else "normal"
    lbl.setStyleSheet(f"color:{c}; font-size:{size}px; font-weight:{w};")
    return lbl


class PerfMonitorWindow(QDialog):
    """현재 프로세스의 CPU·메모리 사용량을 주기적으로 표시."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("성능 모니터링")
        self.setStyleSheet(STYLE)
        self.resize(420, 360)
        self.setWindowFlag(Qt.Window)

        self._proc = None
        if _HAS_PSUTIL:
            try:
                self._proc = psutil.Process(os.getpid())
                self._proc.cpu_percent(None)  # 초기 호출 (기준값 워밍업)
            except Exception:
                self._proc = None

        self._build()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(1000)
        self._refresh()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(12)

        if not _HAS_PSUTIL:
            warn = QLabel(
                "psutil 패키지가 설치되어 있지 않아 성능 정보를 표시할 수 없습니다.\n"
                "pip install psutil 로 설치 후 다시 실행하세요.")
            warn.setStyleSheet("color:#FF8C00; font-size:12px;")
            warn.setWordWrap(True)
            lay.addWidget(warn)
            lay.addStretch()
            return

        # ── CPU ──
        grp_cpu = QGroupBox("CPU 사용률 (이 프로그램)")
        gl1 = QGridLayout(grp_cpu); gl1.setSpacing(6)
        self.lbl_cpu_val = _val("0.0 %", 26)
        self.bar_cpu = _Bar("#4f8ef7")
        gl1.addWidget(self.lbl_cpu_val, 0, 0)
        gl1.addWidget(self.bar_cpu, 1, 0)
        lay.addWidget(grp_cpu)

        # ── 메모리 ──
        grp_mem = QGroupBox("메모리 사용량 (이 프로그램)")
        gl2 = QGridLayout(grp_mem); gl2.setSpacing(6)
        self.lbl_mem_val = _val("0 MB", 26)
        self.lbl_mem_pct = QLabel("")
        self.lbl_mem_pct.setStyleSheet(f"color:{MUTED}; font-size:11px;")
        self.bar_mem = _Bar("#7ae87a")
        gl2.addWidget(self.lbl_mem_val, 0, 0)
        gl2.addWidget(self.lbl_mem_pct, 0, 1)
        gl2.addWidget(self.bar_mem, 1, 0, 1, 2)
        lay.addWidget(grp_mem)

        # ── 시스템 전체 ──
        grp_sys = QGroupBox("시스템 전체")
        gl3 = QGridLayout(grp_sys); gl3.setSpacing(8)
        gl3.addWidget(QLabel("전체 CPU"), 0, 0)
        self.lbl_sys_cpu = _val("0.0 %", 14, color=MUTED)
        gl3.addWidget(self.lbl_sys_cpu, 0, 1)
        gl3.addWidget(QLabel("전체 메모리"), 1, 0)
        self.lbl_sys_mem = _val("0 / 0 GB", 14, color=MUTED)
        gl3.addWidget(self.lbl_sys_mem, 1, 1)
        gl3.addWidget(QLabel("스레드 수"), 2, 0)
        self.lbl_threads = _val("0", 14, color=MUTED)
        gl3.addWidget(self.lbl_threads, 2, 1)
        lay.addWidget(grp_sys)

        hint = QLabel("1초마다 자동 갱신됩니다.")
        hint.setStyleSheet(f"color:{MUTED}; font-size:10px;")
        lay.addWidget(hint)
        lay.addStretch()

    def _refresh(self):
        if not _HAS_PSUTIL or self._proc is None:
            return
        try:
            cpu_pct = self._proc.cpu_percent(None)
            mem_info = self._proc.memory_info()
            mem_mb = mem_info.rss / (1024 * 1024)
            sys_mem = psutil.virtual_memory()
            mem_sys_pct = (mem_info.rss / sys_mem.total) * 100
            sys_cpu = psutil.cpu_percent(None)
            n_threads = self._proc.num_threads()

            self.lbl_cpu_val.setText(f"{cpu_pct:.1f} %")
            self.bar_cpu.set_pct(min(cpu_pct, 100.0))

            self.lbl_mem_val.setText(f"{mem_mb:,.0f} MB")
            self.lbl_mem_pct.setText(f"(시스템의 {mem_sys_pct:.1f}%)")
            self.bar_mem.set_pct(mem_sys_pct)

            self.lbl_sys_cpu.setText(f"{sys_cpu:.1f} %")
            self.lbl_sys_mem.setText(
                f"{sys_mem.used/(1024**3):.1f} / {sys_mem.total/(1024**3):.1f} GB")
            self.lbl_threads.setText(str(n_threads))
        except Exception:
            pass

    def closeEvent(self, event):
        self._timer.stop()
        super().closeEvent(event)
