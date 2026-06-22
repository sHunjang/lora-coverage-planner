# ui/log_viewer_window.py — 로그 관리 창
from __future__ import annotations
import os, subprocess, sys
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QPlainTextEdit, QLabel, QComboBox, QCheckBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTextCursor, QColor, QTextCharFormat
from ui.dialogs import DARK, PANEL, TEXT, MUTED, BORDER, STYLE_DLG
from core.app_logger import get_manager

LEVEL_COLOR = {
    'INFO' : '#a0a8be',
    'WARN' : '#FFD700',
    'ERROR': '#FF6464',
}

STYLE = STYLE_DLG + f"""
QComboBox {{
    background:{PANEL}; color:{TEXT};
    border:1px solid {BORDER}; border-radius:4px;
    padding:4px 8px; min-height:24px;
}}
QPlainTextEdit {{
    background:#14161c; color:{TEXT};
    border:1px solid {BORDER}; border-radius:6px;
    font-family:Consolas,'Courier New',monospace; font-size:11px;
}}
"""


class LogViewerWindow(QDialog):
    """프로그램 실행 로그를 실시간으로 보여주는 창."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("로그 관리")
        self.setStyleSheet(STYLE)
        self.resize(820, 540)
        self.setWindowFlag(Qt.Window)

        self._mgr = get_manager()
        self._autoscroll = True
        self._level_filter = "전체"

        self._build()
        self._load_existing()

        if self._mgr.signal is not None:
            self._mgr.signal.new_line.connect(self._on_new_line)

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        top = QHBoxLayout()
        top.addWidget(QLabel("레벨:"))
        self.cb_level = QComboBox()
        self.cb_level.addItems(["전체", "INFO", "WARN", "ERROR"])
        self.cb_level.currentTextChanged.connect(self._on_filter_changed)
        top.addWidget(self.cb_level)

        self.chk_autoscroll = QCheckBox("자동 스크롤")
        self.chk_autoscroll.setChecked(True)
        self.chk_autoscroll.toggled.connect(
            lambda v: setattr(self, '_autoscroll', v))
        top.addWidget(self.chk_autoscroll)
        top.addStretch()

        btn_open_dir = QPushButton("📁 로그 폴더 열기")
        btn_open_dir.clicked.connect(self._open_log_dir)
        btn_clear = QPushButton("✕ 화면 지우기")
        btn_clear.clicked.connect(self._clear_view)
        for b in (btn_open_dir, btn_clear):
            b.setStyleSheet(
                f"QPushButton{{background:{PANEL};color:{TEXT};"
                f"border:1px solid {BORDER};border-radius:5px;"
                f"padding:5px 12px;font-size:11px;}}"
                f"QPushButton:hover{{border-color:#4f8ef7;}}")
            top.addWidget(b)
        lay.addLayout(top)

        self.txt = QPlainTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setLineWrapMode(QPlainTextEdit.NoWrap)
        lay.addWidget(self.txt)

        hint = QLabel(
            "콘솔 출력(print)을 실시간으로 표시합니다. "
            "화면 지우기는 보기에서만 지워지며, 로그 파일은 그대로 보관됩니다.")
        hint.setStyleSheet(f"color:{MUTED};font-size:10px;")
        lay.addWidget(hint)

    def _load_existing(self):
        for ts, level, line in self._mgr.get_lines():
            self._append_line(ts, level, line)

    def _on_new_line(self, ts, level, line):
        self._append_line(ts, level, line)

    def _append_line(self, ts, level, line):
        self._last_level = level
        if self._level_filter != "전체" and level != self._level_filter:
            return
        color = LEVEL_COLOR.get(level, TEXT)
        html = (f'<span style="color:#5a6075;">[{ts}]</span> '
                f'<span style="color:{color};">[{level}]</span> '
                f'<span style="color:{TEXT};">{_escape(line)}</span>')
        self.txt.appendHtml(html)
        if self._autoscroll:
            self.txt.moveCursor(QTextCursor.End)

    def _on_filter_changed(self, text):
        self._level_filter = text
        self.txt.clear()
        for ts, level, line in self._mgr.get_lines():
            self._append_line(ts, level, line)

    def _clear_view(self):
        self.txt.clear()

    def _open_log_dir(self):
        path = self._mgr.log_dir()
        os.makedirs(path, exist_ok=True)
        try:
            if sys.platform == 'win32':
                os.startfile(path)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', path])
            else:
                subprocess.Popen(['xdg-open', path])
        except Exception as e:
            print(f"[WARN] 로그 폴더 열기 실패: {e}")


def _escape(s: str) -> str:
    return (s.replace('&', '&amp;').replace('<', '&lt;')
             .replace('>', '&gt;'))
