# ui/splash_screen.py
from __future__ import annotations
import os
from PyQt5.QtWidgets import QWidget, QApplication, QPushButton, QLabel
from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QPainter, QColor, QFont, QPen, QLinearGradient, QPixmap, QPainterPath


def _draw_background(p: QPainter, w: int, h: int, assets_dir: str = ""):
    """배경 + 세계 지도 이미지 + 텍스트 그리기."""

    # ── 배경 ─────────────────────────────────────────────────
    bg = QLinearGradient(0, 0, 0, h)
    bg.setColorAt(0.0, QColor("#0d1117"))
    bg.setColorAt(1.0, QColor("#161b27"))
    p.setBrush(bg)
    p.setPen(Qt.NoPen)
    p.drawRect(0, 0, w, h)

    # 테두리
    p.setPen(QPen(QColor("#2a2f3b"), 2))
    p.setBrush(Qt.NoBrush)
    p.drawRoundedRect(1, 1, w-2, h-2, 16, 16)

    # ── 지도 패널 영역 ───────────────────────────────────────
    MAP_X, MAP_Y = 40, 100
    MAP_W, MAP_H = w - 80, h - 200

    # 패널 배경 (이미지 로드 실패 시 폴백)
    p.setBrush(QColor("#0a1628"))
    p.setPen(QPen(QColor("#1e3050"), 1))
    p.drawRoundedRect(MAP_X, MAP_Y, MAP_W, MAP_H, 8, 8)

    # ── 세계 지도 이미지 로드 ────────────────────────────────
    _base = os.path.dirname(os.path.abspath(__file__))
    img_candidates = [
        os.path.join(assets_dir, 'world_map.png'),
        os.path.join(_base, '..', 'assets',   'world_map.png'),
        os.path.join(_base, 'world_map.png'),
        os.path.join(_base, '..', 'data', 'world_map.png'),
    ]
    world_map_px = None
    for path in img_candidates:
        if os.path.exists(path):
            world_map_px = QPixmap(path)
            break

    if world_map_px and not world_map_px.isNull():
        scaled = world_map_px.scaled(
            MAP_W, MAP_H,
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation
        )
        crop_x = (scaled.width()  - MAP_W) // 2
        crop_y = (scaled.height() - MAP_H) // 2
        cropped = scaled.copy(crop_x, crop_y, MAP_W, MAP_H)

        p.save()
        clip_path = QPainterPath()
        clip_path.addRoundedRect(MAP_X, MAP_Y, MAP_W, MAP_H, 8, 8)
        p.setClipPath(clip_path)
        p.drawPixmap(MAP_X, MAP_Y, cropped)
        p.restore()

        # 패널 테두리 다시 그리기
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor("#2a4a6a"), 1))
        p.drawRoundedRect(MAP_X, MAP_Y, MAP_W, MAP_H, 8, 8)

    # ── GW 포인트 오버레이 ───────────────────────────────────
    gw_points = [
        (0.55, 0.25), (0.62, 0.30), (0.20, 0.18),
        (0.72, 0.55), (0.45, 0.48), (0.15, 0.52),
    ]
    for rx, ry in gw_points:
        cx = MAP_X + int(MAP_W * rx)
        cy = MAP_Y + int(MAP_H * ry)
        for radius, alpha in [(20, 18), (13, 45), (6, 110)]:
            c = QColor("#4f8ef7")
            c.setAlpha(alpha)
            p.setBrush(c)
            p.setPen(Qt.NoPen)
            p.drawEllipse(cx-radius, cy-radius, radius*2, radius*2)
        p.setBrush(QColor("#7ab8e8"))
        p.setPen(Qt.NoPen)
        p.drawEllipse(cx-3, cy-3, 6, 6)

    # 연결선
    pts_px = [(MAP_X + int(MAP_W*rx), MAP_Y + int(MAP_H*ry))
              for rx, ry in gw_points]
    p.setPen(QPen(QColor(79, 142, 247, 35), 1))
    for i in range(len(pts_px)):
        for j in range(i+1, len(pts_px)):
            p.drawLine(pts_px[i][0], pts_px[i][1],
                       pts_px[j][0], pts_px[j][1])

    # ── 제목 텍스트 ──────────────────────────────────────────
    p.setRenderHint(QPainter.TextAntialiasing)

    # SmartCity LoRaWAN Network Simulator
    p.setPen(QColor("#a0b4cc"))
    p.setFont(QFont("Segoe UI", 17, QFont.Normal))
    p.drawText(0, 18, w, 34, Qt.AlignHCenter,
               "SmartCity LoRaWAN Network Simulator")

    # LoRaScape
    p.setFont(QFont("Segoe UI", 24, QFont.Bold))
    grad_text = QLinearGradient(w//2-120, 52, w//2+120, 78)
    grad_text.setColorAt(0.0, QColor("#4f8ef7"))
    grad_text.setColorAt(1.0, QColor("#7ab8e8"))
    p.setPen(QPen(grad_text, 0))
    p.drawText(0, 52, w, 42, Qt.AlignHCenter, "LoRaScape")

    # ── 버전 — 왼쪽 상단 ────────────────────────────────────
    p.setFont(QFont("Segoe UI", 8))
    p.setPen(QColor("#3a4060"))
    p.drawText(14, 10, 80, 14, Qt.AlignLeft, "v1.0.0")

    # ── SOLUWINS 로고 — 오른쪽 하단 ─────────────────────────
    LOGO_Y = h - 42
    icon_x = w - 160
    icon_y = LOGO_Y + 4
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("#1a6fc4"))
    p.drawRoundedRect(icon_x, icon_y, 14, 14, 2, 2)
    p.setBrush(QColor("#4f8ef7"))
    p.drawRoundedRect(icon_x+16, icon_y+6, 10, 10, 2, 2)
    p.setBrush(QColor("#FFD700"))
    p.drawRoundedRect(icon_x+16, icon_y, 10, 5, 1, 1)
    p.setFont(QFont("Segoe UI", 13, QFont.Bold))
    p.setPen(QColor("#e0e4ef"))
    p.drawText(icon_x+30, LOGO_Y, 120, 22,
               Qt.AlignLeft | Qt.AlignVCenter, "SOLUWINS")


class SplashScreen(QWidget):
    sig_start = pyqtSignal()

    def __init__(self, assests_dir: str = ""):
        super().__init__()
        self._assets_dir = assests_dir
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(900, 560)

        screen = QApplication.primaryScreen().availableGeometry()
        self.move(
            (screen.width()  - self.width())  // 2,
            (screen.height() - self.height()) // 2,
        )

        self._build()

        # 페이드인
        self.setWindowOpacity(0.0)
        self._anim = QPropertyAnimation(self, b"windowOpacity")
        self._anim.setDuration(600)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.InOutQuad)
        self._anim.start()

    def _build(self):
        W, H   = 900, 560
        MAP_Y  = 100
        MAP_H  = H - 200

        # START 버튼
        self.btn_start = QPushButton("▶  START", self)
        self.btn_start.setFixedSize(200, 48)
        self.btn_start.move((W - 200) // 2, MAP_Y + MAP_H + 16)
        self.btn_start.setStyleSheet("""
            QPushButton {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a4fa0, stop:1 #2a7a5a);
                color: #e0e4ef;
                border: 1px solid #4f8ef7;
                border-radius: 24px;
                font-size: 15px;
                font-weight: bold;
                font-family: 'Segoe UI';
                letter-spacing: 2px;
            }
            QPushButton:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2a6fd0, stop:1 #3aaa7a);
                border-color: #7ab8e8;
            }
            QPushButton:pressed {
                background: #1a3060;
            }
        """)
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.clicked.connect(self._on_start)

        # 안내 텍스트
        self.lbl_hint = QLabel("Click START to launch the simulator", self)
        self.lbl_hint.setFixedWidth(W)
        self.lbl_hint.move(0, MAP_Y + MAP_H + 72)
        self.lbl_hint.setAlignment(Qt.AlignHCenter)
        self.lbl_hint.setStyleSheet(
            "color: #3a4565; font-size: 10px; font-family: 'Segoe UI';"
            "background: transparent;")

    def _on_start(self):
        self.btn_start.setEnabled(False)
        self.btn_start.setText("Loading...")
        self._anim_out = QPropertyAnimation(self, b"windowOpacity")
        self._anim_out.setDuration(400)
        self._anim_out.setStartValue(1.0)
        self._anim_out.setEndValue(0.0)
        self._anim_out.setEasingCurve(QEasingCurve.InOutQuad)
        self._anim_out.finished.connect(self._finish)
        self._anim_out.start()

    def _finish(self):
        self.sig_start.emit()
        self.close()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        _draw_background(p, self.width(), self.height(), self._assets_dir)
        p.end()