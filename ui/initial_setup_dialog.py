# ui/initial_setup_dialog.py — 최초 실행 시 초기 환경 설정
from __future__ import annotations
import os
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QLabel, QGroupBox, QFileDialog,
)
from PyQt5.QtCore import Qt
from ui.dialogs import DARK, PANEL, TEXT, MUTED, BORDER, STYLE_DLG
from core.app_config import load_config, save_config


class InitialSetupDialog(QDialog):
    """최초 실행 시 회사/사용자/작업폴더 설정."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("초기 환경 설정")
        self.setStyleSheet(STYLE_DLG)
        self.setFixedWidth(520)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self._cfg = load_config()
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)

        info = QLabel(
            "LoRaScape를 처음 실행합니다.\n"
            "회사 정보와 기본 작업 폴더를 설정하세요.")
        info.setStyleSheet(f"color:{MUTED};font-size:11px;")
        info.setWordWrap(True)
        lay.addWidget(info)

        # 회사/사용자 정보
        grp = QGroupBox("사용자 정보")
        fl  = QFormLayout(grp); fl.setSpacing(8)
        self.e_company = QLineEdit(self._cfg.get('company', ''))
        self.e_company.setPlaceholderText("회사명")
        self.e_user = QLineEdit(self._cfg.get('user', ''))
        self.e_user.setPlaceholderText("사용자명")
        fl.addRow("회사명", self.e_company)
        fl.addRow("사용자명", self.e_user)
        lay.addWidget(grp)

        # 작업 폴더
        grp2 = QGroupBox("기본 작업 폴더")
        h    = QHBoxLayout(grp2)
        self.e_workdir = QLineEdit(self._cfg.get('work_dir', ''))
        self.e_workdir.setReadOnly(True)
        btn_br = QPushButton("찾아보기")
        btn_br.setFixedWidth(90)
        btn_br.clicked.connect(self._browse_workdir)
        h.addWidget(self.e_workdir, 1)
        h.addWidget(btn_br)
        lay.addWidget(grp2)

        self.lbl_err = QLabel("")
        self.lbl_err.setStyleSheet("color:#e87a7a;font-size:10px;")
        self.lbl_err.setWordWrap(True)
        lay.addWidget(self.lbl_err)

        # 버튼
        bot = QHBoxLayout()
        btn_ok = QPushButton("✔  저장 후 시작")
        btn_ok.setProperty("role", "ok")
        btn_ok.clicked.connect(self._accept)
        bot.addStretch()
        bot.addWidget(btn_ok)
        lay.addLayout(bot)

    def _browse_workdir(self):
        d = QFileDialog.getExistingDirectory(
            self, "작업 폴더 선택",
            self.e_workdir.text() or os.path.expanduser("~"))
        if d:
            self.e_workdir.setText(d)

    def _accept(self):
        company = self.e_company.text().strip()
        user    = self.e_user.text().strip()
        workdir = self.e_workdir.text().strip()

        if not company or not user:
            self.lbl_err.setText("회사명과 사용자명을 입력하세요.")
            return
        if not workdir:
            self.lbl_err.setText("작업 폴더를 선택하세요.")
            return

        # 작업 폴더 생성
        try:
            os.makedirs(workdir, exist_ok=True)
        except Exception as e:
            self.lbl_err.setText(f"작업 폴더 생성 실패: {e}")
            return

        self._cfg['company']  = company
        self._cfg['user']     = user
        self._cfg['work_dir'] = workdir
        save_config(self._cfg)
        self.accept()