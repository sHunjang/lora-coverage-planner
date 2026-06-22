# ui/license_dialog.py — 라이선스 인증 다이얼로그
from __future__ import annotations
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QLabel, QGroupBox,
)
from PyQt5.QtCore import Qt
from ui.dialogs import DARK, PANEL, TEXT, MUTED, BORDER, STYLE_DLG
from core.license import verify_code, save_license
from core.app_config import load_config


class LicenseDialog(QDialog):
    """라이선스 인증 — 회사명/사용자명/인증코드 입력."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("라이선스 등록")
        self.setStyleSheet(STYLE_DLG)
        self.setFixedWidth(520)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        # 초기 환경 설정에서 입력한 회사/사용자 정보 가져오기
        self._cfg = load_config()
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)

        info = QLabel(
            "LoRaScape 라이선스 인증이 필요합니다.\n"
            "솔루윈스에서 발급받은 인증 코드를 입력하세요.")
        info.setStyleSheet(f"color:{MUTED};font-size:11px;")
        info.setWordWrap(True)
        lay.addWidget(info)

        grp = QGroupBox("라이선스 정보")
        fl  = QFormLayout(grp); fl.setSpacing(8)
        self.e_company = QLineEdit(self._cfg.get('company', ''))
        self.e_company.setPlaceholderText("회사명")
        self.e_user = QLineEdit(self._cfg.get('user', ''))
        self.e_user.setPlaceholderText("사용자명")
        self.e_code = QLineEdit("")
        self.e_code.setPlaceholderText("XXXXX-XXXXX-XXXXX-XXXXX-XXXX")
        fl.addRow("회사명", self.e_company)
        fl.addRow("사용자명", self.e_user)
        fl.addRow("인증 코드", self.e_code)
        lay.addWidget(grp)

        self.lbl_msg = QLabel("")
        self.lbl_msg.setStyleSheet("color:#e87a7a;font-size:10px;")
        self.lbl_msg.setWordWrap(True)
        lay.addWidget(self.lbl_msg)

        bot = QHBoxLayout()
        btn_cancel = QPushButton("취소")
        btn_cancel.setProperty("role", "cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_ok = QPushButton("✔  인증")
        btn_ok.setProperty("role", "ok")
        btn_ok.clicked.connect(self._verify)
        bot.addStretch()
        bot.addWidget(btn_cancel)
        bot.addWidget(btn_ok)
        lay.addLayout(bot)

    def _verify(self):
        company = self.e_company.text().strip()
        user    = self.e_user.text().strip()
        code    = self.e_code.text().strip()

        if not (company and user and code):
            self.lbl_msg.setText("모든 항목을 입력하세요.")
            return

        if verify_code(company, user, code):
            save_license(company, user, code)
            self.accept()
        else:
            self.lbl_msg.setText(
                "✗ 인증 코드가 올바르지 않습니다. "
                "회사명·사용자명·코드를 확인하세요.")
