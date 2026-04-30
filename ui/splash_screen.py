# ui/splash_screen.py
from __future__ import annotations
from PyQt5.QtWidgets import QWidget, QApplication, QPushButton, QLabel
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve, QRect
from PyQt5.QtGui import QPainter, QColor, QFont, QPen, QLinearGradient, QPixmap, QBrush


def _draw_background(p: QPainter, w: int, h: int):
    """배경 + 지도 패널 + GW 포인트 그리기."""
    # 배경
    bg = QLinearGradient(0, 0, 0, h)
    bg.setColorAt(0.0, QColor("#0d1117"))
    bg.setColorAt(1.0, QColor("#161b27"))
    p.setBrush(bg)
    p.setPen(Qt.NoPen)
    p.drawRect(0, 0, w, h)

    # 테두리
    pen = QPen(QColor("#2a2f3b"))
    pen.setWidth(2)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawRoundedRect(1, 1, w-2, h-2, 16, 16)

    # 지도 패널
    MAP_X, MAP_Y = 40, 110
    MAP_W, MAP_H = w - 80, h - 230

    map_bg = QLinearGradient(MAP_X, MAP_Y, MAP_X, MAP_Y + MAP_H)
    map_bg.setColorAt(0.0, QColor("#1a2035"))
    map_bg.setColorAt(1.0, QColor("#131929"))
    p.setBrush(map_bg)
    p.setPen(QPen(QColor("#2a3550"), 1))
    p.drawRoundedRect(MAP_X, MAP_Y, MAP_W, MAP_H, 10, 10)

    # 격자선
    p.setPen(QPen(QColor("#1e2840"), 1))
    for i in range(1, 9):
        x = MAP_X + MAP_W * i // 9
        p.drawLine(x, MAP_Y, x, MAP_Y + MAP_H)
    for i in range(1, 5):
        y = MAP_Y + MAP_H * i // 5
        p.drawLine(MAP_X, y, MAP_X + MAP_W, y)

    # 대륙 실루엣
    p.setPen(Qt.NoPen)
    continents = [
        (0.42, 0.15, 0.10, 0.55),  # 유럽/아프리카
        (0.52, 0.10, 0.22, 0.45),  # 아시아
        (0.10, 0.10, 0.18, 0.40),  # 북미
        (0.18, 0.50, 0.10, 0.35),  # 남미
        (0.70, 0.55, 0.10, 0.25),  # 호주
    ]
    for rx, ry, rw, rh in continents:
        p.setBrush(QColor("#1e3a2a"))
        p.drawEllipse(
            MAP_X + int(MAP_W * rx), MAP_Y + int(MAP_H * ry),
            int(MAP_W * rw), int(MAP_H * rh))

    # GW 포인트 + 신호 원
    gw_points = [
        (0.55, 0.25), (0.62, 0.30), (0.58, 0.45),
        (0.20, 0.20), (0.72, 0.60), (0.45, 0.50),
    ]
    pts_px = []
    for rx, ry in gw_points:
        cx = MAP_X + int(MAP_W * rx)
        cy = MAP_Y + int(MAP_H * ry)
        pts_px.append((cx, cy))
        for radius, alpha in [(22, 25), (14, 55), (7, 110)]:
            c = QColor("#4f8ef7"); c.setAlpha(alpha)
            p.setBrush(c); p.setPen(Qt.NoPen)
            p.drawEllipse(cx-radius, cy-radius, radius*2, radius*2)
        p.setBrush(QColor("#7ab8e8"))
        p.drawEllipse(cx-3, cy-3, 6, 6)

    # 연결선
    p.setPen(QPen(QColor(79, 142, 247, 50), 1))
    for i in range(len(pts_px)):
        for j in range(i+1, len(pts_px)):
            p.drawLine(pts_px[i][0], pts_px[i][1],
                       pts_px[j][0], pts_px[j][1])

    # 제목
    p.setPen(QColor("#7a8099"))
    p.setFont(QFont("Segoe UI", 15))
    p.drawText(0, 20, w, 28, Qt.AlignHCenter,
               "SmartCity LoRaWAN Network Simulator")

    # LoRaScape
    p.setFont(QFont("Segoe UI", 34, QFont.Bold))
    grad_text = QLinearGradient(w//2-150, 50, w//2+150, 85)
    grad_text.setColorAt(0.0, QColor("#4f8ef7"))
    grad_text.setColorAt(1.0, QColor("#7ab8e8"))
    p.setPen(QPen(grad_text, 0))
    p.drawText(0, 48, w, 52, Qt.AlignHCenter, "LoRaScape")

    # 버전 — 왼쪽 상단
    p.setFont(QFont("Segoe UI", 8))
    p.setPen(QColor("#3a4060"))
    p.drawText(14, 10, 80, 14, Qt.AlignLeft, "v1.0.0")

    # SOLUWINS 로고 — 오른쪽 하단
    LOGO_Y = h - 42
    icon_x = w - 160   # 오른쪽 정렬
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
    sig_start = pyqtSignal()   # START 버튼 클릭 시 메인창 열기 신호

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(900, 560)

        # 화면 중앙 배치
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(
            (screen.width()  - self.width())  // 2,
            (screen.height() - self.height()) // 2,
        )

        self._build()

        # 페이드인 애니메이션
        self.setWindowOpacity(0.0)
        self._anim = QPropertyAnimation(self, b"windowOpacity")
        self._anim.setDuration(600)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.InOutQuad)
        self._anim.start()

    def _build(self):
        W, H = 900, 560
        MAP_Y = 110; MAP_H = H - 230

        # START 버튼
        self.btn_start = QPushButton("▶  START", self)
        self.btn_start.setFixedSize(200, 48)
        self.btn_start.move((W - 200) // 2, MAP_Y + MAP_H + 20)
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
        self.lbl_hint.move(0, MAP_Y + MAP_H + 76)
        self.lbl_hint.setAlignment(Qt.AlignHCenter)
        self.lbl_hint.setStyleSheet(
            "color: #3a4565; font-size: 10px; font-family: 'Segoe UI';"
            "background: transparent;")

    def _on_start(self):
        """START 클릭 → 페이드아웃 후 메인창 실행."""
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
        _draw_background(p, self.width(), self.height())
        p.end()