# ui/manual_window.py
# 사용자 매뉴얼 PDF 자동 생성 창
# PyQt5 화면 캡처 + reportlab PDF 생성

from __future__ import annotations
import os
import io
import platform
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QProgressBar, QMessageBox, QGroupBox,
    QCheckBox, QWidget,
)
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QThread
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication
from ui.dialogs import DARK, PANEL, TEXT, MUTED, BORDER, STYLE_DLG

BTN = ("QPushButton{background:#1c2a3a;color:#7ab8e8;"
       "border:1px solid #2a4a6a;border-radius:5px;"
       "padding:7px 18px;font-size:12px;}"
       "QPushButton:hover{background:#254d78;}"
       "QPushButton:disabled{color:#3a5a6a;border-color:#1a2a3a;}")
BTN_GEN = ("QPushButton{background:#1d4a1d;color:#7ae87a;"
           "border:1px solid #2a6a2a;border-radius:5px;"
           "padding:8px 22px;font-size:13px;font-weight:bold;}"
           "QPushButton:hover{background:#256a25;}"
           "QPushButton:disabled{color:#3a5a3a;}")


def _get_korean_font():
    """운영체제별 한글 폰트 경로 반환."""
    if platform.system() == 'Windows':
        for path in ['C:/Windows/Fonts/malgun.ttf',
                     'C:/Windows/Fonts/gulim.ttc']:
            if os.path.exists(path):
                return path
    elif platform.system() == 'Darwin':
        path = '/System/Library/Fonts/AppleGothic.ttf'
        if os.path.exists(path):
            return path
    return None


class ManualWorker(QObject):
    """PDF 생성 워커."""
    sig_progress = pyqtSignal(int, str)
    sig_done     = pyqtSignal(str)
    sig_err      = pyqtSignal(str)

    def __init__(self, path, screenshots, result, gws, nodes, settings):
        super().__init__()
        self.path        = path
        self.screenshots = screenshots
        self.result      = result
        self.gws         = gws
        self.nodes       = nodes
        self.settings    = settings

    def run(self):
        try:
            self._build_pdf()
            self.sig_done.emit(self.path)
        except Exception:
            import traceback
            self.sig_err.emit(traceback.format_exc())

    def _build_pdf(self):
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer,
            Table, TableStyle, PageBreak, Image as RLImage,
            HRFlowable,
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        # ── 한글 폰트 등록 ────────────────────────────────
        font_name = 'Helvetica'
        font_path = _get_korean_font()
        if font_path:
            try:
                pdfmetrics.registerFont(TTFont('KoreanFont', font_path))
                font_name = 'KoreanFont'
            except Exception:
                pass

        # ── 스타일 정의 ───────────────────────────────────
        styles = getSampleStyleSheet()

        def _sty(name, parent='Normal', **kw):
            return ParagraphStyle(name, parent=styles[parent],
                                  fontName=font_name, **kw)

        sty_title   = _sty('T',  'Title',   fontSize=20, spaceAfter=6,
                            textColor=colors.HexColor('#3062b7'))
        sty_h1      = _sty('H1', 'Heading1', fontSize=14, spaceAfter=4,
                            spaceBefore=12,
                            textColor=colors.HexColor('#1d2871'))
        sty_h2      = _sty('H2', 'Heading2', fontSize=11, spaceAfter=3,
                            spaceBefore=8,
                            textColor=colors.HexColor('#000000'))
        sty_body    = _sty('B',  fontSize=9,  spaceAfter=3, leading=14)
        sty_bullet  = _sty('BL', fontSize=9,  spaceAfter=2,
                            leftIndent=12, leading=13)
        sty_caption = _sty('C',  fontSize=8,  spaceAfter=4,
                            textColor=colors.HexColor('#7a8099'),
                            alignment=1)
        sty_note    = _sty('N',  fontSize=8,  spaceAfter=3,
                            textColor=colors.HexColor('#D1A700'),
                            leftIndent=8)

        doc = SimpleDocTemplate(
            self.path, pagesize=A4,
            leftMargin=18*mm, rightMargin=18*mm,
            topMargin=18*mm, bottomMargin=18*mm,
        )
        story = []
        W = A4[0] - 36*mm

        def _hr():
            story.append(HRFlowable(
                width='100%', thickness=0.5,
                color=colors.HexColor('#2a2f3b'),
                spaceAfter=4))

        def _img(key, caption='', max_w=None, max_h=80*mm):
            px = self.screenshots.get(key)
            if px is None:
                story.append(Paragraph(f'[{key} 화면 없음]', sty_caption))
                return
            from PyQt5.QtCore import QBuffer, QByteArray, QIODevice
            byte_array = QByteArray()
            buffer = QBuffer(byte_array)
            buffer.open(QIODevice.WriteOnly)
            px.save(buffer, 'PNG')
            buffer.close()
            img_bytes = io.BytesIO(bytes(byte_array))
            mw  = max_w or W
            img = RLImage(img_bytes, width=mw, height=max_h,
                          kind='proportional')
            story.append(img)
            if caption:
                story.append(Paragraph(caption, sty_caption))
            story.append(Spacer(1, 3*mm))

        def _tbl(data, col_widths=None):
            t = Table(data, colWidths=col_widths)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0),
                 colors.HexColor('#1E2130')),
                ('TEXTCOLOR',  (0,0), (-1,0),
                 colors.HexColor('#7AB8E8')),
                ('FONTNAME',   (0,0), (-1,-1), font_name),
                ('FONTSIZE',   (0,0), (-1,-1), 8),
                ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
                ('GRID',       (0,0), (-1,-1), 0.4,
                 colors.HexColor('#2A2F3B')),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [
                    colors.HexColor('#181B22'),
                    colors.HexColor('#1E2130'),
                ]),
                ('TEXTCOLOR', (0,1), (-1,-1),
                 colors.HexColor('#E0E4EF')),
            ]))
            story.append(t)
            story.append(Spacer(1, 3*mm))

        # ══════════════════════════════════════════════════
        # 표지
        # ══════════════════════════════════════════════════
        self.sig_progress.emit(3, '표지 작성 중...')
        story.append(Spacer(1, 30*mm))
        story.append(Paragraph(
            '스마트시티 AIoT 네트워크 설계를 위한<br/>망 설계 도구의 사용자 Manual',
            sty_title))
        story.append(Spacer(1, 6*mm))
        story.append(Paragraph(
            'SmartCity LoRaWAN Network Simulator — LoRaScape', sty_h2))
        story.append(Paragraph('SOLUWINS', sty_body))
        story.append(Spacer(1, 8*mm))
        _hr()
        story.append(Spacer(1, 4*mm))

        if self.result:
            pct       = self.result.coverage_pct
            n_cov     = self.result.n_covered
            n_tot     = self.result.n_total
            n_gws     = len(self.gws)
            cell_rate = getattr(self.result, 'cell_success_rate', 0.0)
            avg_pdr   = getattr(self.result, 'avg_pdr',           100.0)
            avg_snr   = getattr(self.result, 'avg_snr',           0.0)
            summary_data = [
                ['항목', '값'],
                ['전체 Node 수',   f'{n_tot}개'],
                ['커버 Node 수',   f'{n_cov}개'],
                ['전체 커버리지', f'{pct:.1f}%'],
                ['활성 GW 수',    f'{n_gws}개'],
                ['셀 통신 성공율', f'{cell_rate:.1f}%'],
                ['평균 PDR',      f'{avg_pdr:.1f}%'],
                ['평균 SNR',      f'{avg_snr:.1f} dB'],
            ]
            story.append(Paragraph('■ 현재 분석 결과 요약', sty_h2))
            _tbl(summary_data, [80*mm, 60*mm])

        story.append(PageBreak())

        # ══════════════════════════════════════════════════
        # 1장. 개요
        # ══════════════════════════════════════════════════
        self.sig_progress.emit(8, '1장 개요 작성 중...')
        story.append(Paragraph('1. 개요 (Introduction)', sty_h1))
        _hr()

        story.append(Paragraph('1.1 목적', sty_h2))
        story.append(Paragraph(
            '본 시뮬레이터는 대도시 도심지 내 각종 편의 시설, 하천 및 공원 등 스마트시티 환경에서 '
            '서로 다른 서비스 특성을 가진 초대규모 IoT에 대한 안정적 연결과 다양한 네트워크 수용을 위한 '
            'AIoT 네트워크 설계용 시뮬레이터입니다.', sty_body))
        for txt in [
            'AIoT GW의 무선 커버리지 확보 및 간섭 최소화',
            '다양한 단말의 안정적 연결을 위한 게이트웨이(GW) 최적 배치 방안 확보',
            'SNR/트래픽 기반 통신 품질(QoS) 정량 분석',
            '다양한 시나리오 비교를 통한 설계 의사결정 지원',
        ]:
            story.append(Paragraph(f'▪ {txt}', sty_bullet))
        story.append(Paragraph(
            '특히 스마트시티 내 LoRa 기반 LPWAN 환경에서 발생하는 긴 거리 전파, '
            '지형 및 건물 등의 장애물에 의한 신호 감쇠 등 도시 환경 음영지역 문제를 '
            '정량적으로 분석하여 설계 의사결정을 지원한다.', sty_body))

        story.append(Spacer(1, 3*mm))
        story.append(Paragraph('1.2 적용 범위', sty_h2))
        areas = [
            ('대도시 도심지',
             '고층 건물 밀집 지역, 전파 반사 및 차폐가 심한 환경, 음영지역(Dead Zone) 분석 필수'),
            ('하천 및 공원 지역',
             '개활지 기반 장거리 통신 환경, 저밀도 단말 분포, 홍수/재난 감시 IoT 적용'),
            ('주거 및 상업 지역',
             '중간 밀도 IoT 환경, 다양한 서비스 혼재'),
        ]
        for title, desc in areas:
            story.append(Paragraph(f'▪ {title}', sty_h2))
            story.append(Paragraph(desc, sty_bullet))

        story.append(Spacer(1, 3*mm))
        story.append(Paragraph('1.3 주요 기능', sty_h2))
        funcs = [
            ('RF 링크 버짓 기반 커버리지 계산',
             'Pr = EIRP - PL + Gr - Lr - Indoor_Loss'),
            ('간섭 및 SF별 커버리지 분석',
             'SF7~SF12 수신 감도 기준 커버리지 경계 계산'),
            ('SNR 기반 통신 성공율 분석',
             '셀 전체/경계 통신 성공율, 평균 SNR/마진 계산'),
            ('트래픽 용량 분석 (ALOHA 기반)',
             'GW별 채널 부하율, 패킷 성공률(PDR) 계산'),
            ('AI 기반 GW 위치 최적화',
             'K-means 클러스터링 + 유전 알고리즘(GA) 최소 GW 탐색'),
            ('시나리오 비교 분석',
             '분석 결과 히스토리 저장, 2개 시나리오 수치 비교'),
            ('다중 레이어 지도 시각화',
             '격자 히트맵, 수신전력 분포, 등고선, SF 레이어'),
            ('PDF/Excel 리포트 자동 생성',
             '커버리지 요약, GW/Node 상세, SF 분포 포함'),
        ]
        func_data = [['기능', '설명']] + [[f, d] for f, d in funcs]
        _tbl(func_data, [65*mm, 105*mm])

        story.append(PageBreak())

        # ══════════════════════════════════════════════════
        # 2장. 시스템 구성
        # ══════════════════════════════════════════════════
        self.sig_progress.emit(14, '2장 시스템 구성 작성 중...')
        story.append(Paragraph('2. 시스템 구성 (System Architecture)', sty_h1))
        _hr()

        story.append(Paragraph('2.1 전체 구조', sty_h2))
        story.append(Paragraph(
            '본 시스템은 4개의 핵심 모듈로 구성됩니다:', sty_body))

        # ── 모듈 개요 ─────────────────────────────────────
        modules = [
            ('① 입력 데이터 처리 모듈',
             'GIS 기반 지도/DEM/DSM 데이터 입력, 단말/GW 데이터 처리, '
             '데이터 정합성 검증, CSV 가져오기/내보내기'),
            ('② 전파 및 간섭 분석 엔진',
             'COST-231 Hata / SmartCity AIoT 모델 적용, RSSI/SNR 계산, '
             '커버리지 맵 생성, 중첩도 분석, SNR 기반 통신 성공율, '
             'ALOHA 기반 트래픽 용량 분석'),
            ('③ AI 최적화 엔진',
             'K-means 기반 초기 배치, 유전 알고리즘(GA) 최적화, '
             'Path Loss 제약 조건 반영, 최소 GW 수 산출'),
            ('④ 결과 시각화 및 리포트 모듈',
             'GIS 지도 기반 다중 레이어 시각화, RSSI/SNR/SF/중첩도 그래프, '
             '시나리오 비교 창, PDF/Excel 자동 생성'),
        ]
        for name, desc in modules:
            story.append(Paragraph(name, sty_h2))
            story.append(Paragraph(desc, sty_bullet))
            story.append(Spacer(1, 2*mm))

        # ── 모듈별 상세 기능 및 출력 ────────────────────
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph('■ 모듈별 주요 기능 및 출력 데이터', sty_h2))

        module_details = [
            (
                '① 입력 데이터 처리 모듈',
                [
                    '공간 데이터 처리 — GIS 지도(Shapefile), DEM/DSM 데이터 변환, 성남시 경계 마스크',
                    '단말 데이터 처리 — 위치 좌표(위도/경도), 안테나 이득, 수신 감도, 실내 투과 손실',
                    'GW 후보 생성 — 사용자 지정 위치, 클러스터링 기반 자동 후보 생성',
                    '데이터 정합성 검증 — 좌표 오류 검출, 중복 단말 제거, 비정상 파라미터 필터링',
                    'CSV 가져오기/내보내기 — GW/Node 목록 일괄 처리',
                ],
                [
                    '정제된 단말 리스트',
                    'GW 후보 위치 집합',
                    '전파 시뮬레이션용 환경 모델 (지형/건물 정보)',
                ]
            ),
            (
                '② 전파 및 간섭 분석 엔진',
                [
                    '전파 손실 모델링 — COST-231 Hata 모델 / SmartCity AIoT 모델 (환경 자동 분류 포함)',
                    '수신 신호 계산 — RSSI(수신전력), SNR(신호 대 잡음비), SNR 마진',
                    '커버리지 분석 — Node별 커버 여부 판단, Coverage Map 생성 (Node × GW 병렬 계산)',
                    'SF별 Link 분석 — SF7~SF12 기준 커버리지 경계, ADR SF 자동 결정',
                    '중첩도 분석 — GW별 커버리지 중첩 영역, 매크로 다이버시티 이득 계산',
                    'SNR 기반 통신 성공율 — 셀 전체/경계 성공율 계산',
                    'ALOHA 기반 트래픽 용량 — GW 채널 부하율 및 패킷 성공률(PDR) 계산',
                ],
                [
                    '커버리지 결과 (Node별 Pr, SF, SNR, 마진)',
                    '중첩도 지도 및 통신 성공율',
                    '트래픽 부하율 및 패킷 성공률(PDR)',
                ]
            ),
            (
                '③ AI 최적화 엔진',
                [
                    '초기 배치 생성 — K-means 클러스터링 + Greedy Set Cover (연결 수 기반)',
                    '제약 조건 반영 — Path Loss 한계, RSSI 기준, GW 최소 커버 수',
                    '반복 최적화 — GA 세대/인구 기반 수렴 탐색, 지역 최적해 탈출',
                    'ILP(정수 선형 계획) 시도 — K-means 모드에서 최적 집합 탐색',
                    '소규모 GW 제거 — min_cover 미만 GW 자동 제거 (Step 9)',
                ],
                [
                    '최적 GW 위치 좌표',
                    '최소 GW 개수',
                    '커버리지 및 트래픽 분석 결과',
                ]
            ),
            (
                '④ 결과 시각화 및 리포트 모듈',
                [
                    '지도 기반 다중 레이어 시각화 — 격자 히트맵, 수신전력 분포, 등고선, SF, 중첩도',
                    'RSSI/SNR/SF/중첩도 분포 그래프 (4탭, matplotlib)',
                    '시나리오 비교 창 — 히스토리 10개, 22개 지표 색상 구분 비교 테이블',
                    'PDF 리포트 자동 생성 — 한글 폰트, 커버리지 요약, GW/Node 상세',
                    'Excel 리포트 자동 생성 — 커버리지 요약/GW/Node/SF 분포 4개 시트',
                    '사용자 매뉴얼 PDF 자동 생성 — 현재 화면 스크린샷 포함',
                ],
                [
                    '시각화 화면 (지도 + 다중 레이어)',
                    '분석 리포트 (PDF/Excel)',
                    '시나리오 비교 테이블',
                ]
            ),
        ]

        for mod_name, features, outputs in module_details:
            story.append(Paragraph(mod_name, sty_h2))
            story.append(Paragraph('■ 주요 기능', sty_body))
            for f in features:
                story.append(Paragraph(f'  - {f}', sty_bullet))
            story.append(Paragraph('■ 출력 데이터', sty_body))
            for o in outputs:
                story.append(Paragraph(f'  - {o}', sty_bullet))
            story.append(Spacer(1, 2*mm))

        story.append(Spacer(1, 3*mm))
        story.append(Paragraph('2.2 주요 구성 요소', sty_h2))
        comp_data = [
            ['구성 요소', '설명'],
            ['Device Model',       '단말 위치, 출력, SF, 수신 임계치, 실내 투과 손실 등'],
            ['Gateway Model',      '수신 감도, 안테나 높이, 송신 출력, 채널 수'],
            ['Propagation Model',  'COST-231 Hata / SmartCity AIoT 모델 (환경 자동 분류 포함)'],
            ['Optimization Engine','K-means 클러스터링 + Greedy Set Cover + 유전 알고리즘(GA)'],
            ['Traffic Analyzer',   'Pure ALOHA 기반 GW 채널 부하율 및 PDR 계산'],
            ['Scenario Comparator','최대 10개 분석 결과 히스토리, 2개 시나리오 비교 테이블'],
        ]
        _tbl(comp_data, [55*mm, 115*mm])

        story.append(PageBreak())

        # ══════════════════════════════════════════════════
        # 3장. 설치 및 실행
        # ══════════════════════════════════════════════════
        self.sig_progress.emit(20, '3장 설치/실행 작성 중...')
        story.append(Paragraph('3. 설치 및 실행 (Installation)', sty_h1))
        _hr()

        story.append(Paragraph('3.1 시스템 요구사항', sty_h2))
        req_data = [
            ['항목', '최소 사양', '권장 사양'],
            ['CPU',    '8 Core 이상',  '16 Core 이상'],
            ['RAM',    '16 GB 이상',   '32 GB 이상'],
            ['저장공간','20 GB 이상',  '50 GB 이상'],
            ['OS',     'Windows 10 64-bit', 'Windows 11 64-bit'],
            ['Python', '3.9 이상',     '3.11 이상 (Miniconda 권장)'],
        ]
        _tbl(req_data, [40*mm, 60*mm, 70*mm])

        story.append(Paragraph('3.2 필수 라이브러리', sty_h2))
        libs = [
            'PyQt5, PyQtWebEngine — GUI 및 웹 기반 지도 렌더링',
            'folium — 인터랙티브 지도 시각화',
            'rasterio, geopandas, shapely — GIS/DEM 데이터 처리',
            'numpy, scipy, sklearn — 수치 계산 및 K-means 최적화',
            'matplotlib — RSSI/SNR/SF 분포 그래프',
            'reportlab, openpyxl — PDF/Excel 리포트 자동 생성',
            'pyproj — 좌표계 변환 (WGS84 ↔ EPSG:3857)',
        ]
        for lib in libs:
            story.append(Paragraph(f'▪ {lib}', sty_bullet))

        story.append(Spacer(1, 3*mm))
        story.append(Paragraph('3.3 설치 절차', sty_h2))
        for step in [
            'Miniconda 설치 및 conda 환경 생성 (conda create -n map python=3.11)',
            'pip install -r requirements.txt 로 의존 라이브러리 일괄 설치',
            'data/ 폴더에 Shapefile 및 DEM 파일 배치',
        ]:
            story.append(Paragraph(f'▪ {step}', sty_bullet))

        story.append(Spacer(1, 3*mm))
        story.append(Paragraph('3.4 실행 방법', sty_h2))
        story.append(Paragraph(
            'Miniconda 환경에서 아래 명령어로 실행합니다:', sty_body))
        story.append(Paragraph(
            'conda activate map &amp;&amp; python main.py', sty_note))
        story.append(Paragraph(
            '앱 시작 시 GW/Node 세션이 자동 복원됩니다 (session.json).', sty_body))
        _img('main', '그림 3.1 메인 화면 — 지도 뷰 + 분석 결과 패널 + 툴바')

        story.append(Paragraph('3.5 툴바 버튼 안내', sty_h2))
        tb_data = [
            ['버튼', '기능'],
            ['GW 목록',      'GW 추가/삭제/편집, 커버리지/히트맵 분석 실행'],
            ['단말 목록',    'Node 추가/삭제/편집, CSV 가져오기/내보내기'],
            ['GW 최적 배치', 'K-means/GA 기반 GW 최적 위치 탐색 (2단계 실행)'],
            ['범례 설정',    '수신전력 색상 범례 레벨 편집'],
            ['그래프',       'RSSI/SNR/SF/중첩도 분포 그래프 창 (4탭)'],
            ['설정',         '전파 모델, GW/Node 기본값, SNR, 히트맵/지도 설정'],
            ['거리 측정',    '지도 클릭으로 거리/방위각 측정 (토글)'],
            ['결과 저장',    '분석 결과 JSON 형식으로 저장'],
            ['결과 불러오기','저장된 JSON 결과 복원'],
            ['리포트',       'PDF/Excel 리포트 생성'],
            ['결과 비교',    '분석 히스토리에서 2개 시나리오 22개 지표 비교'],
            ['매뉴얼',       '본 사용자 매뉴얼 PDF 자동 생성'],
        ]
        _tbl(tb_data, [40*mm, 130*mm])

        story.append(PageBreak())

        # ══════════════════════════════════════════════════
        # 4장. 데이터 입력
        # ══════════════════════════════════════════════════
        self.sig_progress.emit(28, '4장 데이터 입력 작성 중...')
        story.append(Paragraph('4. 데이터 입력 (Input Configuration)', sty_h1))
        _hr()

        story.append(Paragraph('4.1 지도 및 환경 데이터', sty_h2))
        story.append(Paragraph(
            '성남시 경계 Shapefile(EPSG:3857)과 DSM/DEM 데이터를 자동 로드합니다. '
            '지도 배경은 [설정] → [히트맵/지도 탭]에서 변경할 수 있습니다.', sty_body))
        for item in [
            'GIS 기반 지도 데이터 (Shapefile — 성남시 경계, EPSG:3857)',
            'DEM/DSM 데이터 (건물 포함 수치 표면 모델, 10m 해상도)',
            '지도 배경: CartoDB Voyager / DarkMatter / OpenStreetMap 등',
        ]:
            story.append(Paragraph(f'▪ {item}', sty_bullet))

        story.append(Spacer(1, 3*mm))
        story.append(Paragraph('4.2 단말(Node) 설정', sty_h2))
        story.append(Paragraph(
            '툴바의 [단말 목록] 버튼으로 단말 추가/삭제/편집합니다. '
            '지도 우클릭 → [이 위치에 단말기 추가]로 바로 추가할 수 있습니다.', sty_body))
        _img('node_list', '그림 4.1 단말 목록 창')
        nd_data = [
            ['파라미터', '설명', '기본값'],
            ['Callsign',      '단말 식별자',         'Node1, Node2...'],
            ['위도/경도',     '단말 위치 좌표',       '지도 클릭으로 설정'],
            ['Gr (dBi)',      '수신 안테나 이득',     '2.15 dBi'],
            ['Lr (dB)',       '수신 손실',            '0.0 dB'],
            ['hm (m)',        '안테나 높이',          '1.5 m'],
            ['최소수신(dBm)', '수신 감도 임계값',     '-126.6 dBm'],
            ['실내손실(dB)',  '실내 투과 손실',       '0.0 dB'],
        ]
        _tbl(nd_data, [40*mm, 75*mm, 55*mm])

        story.append(Paragraph('4.3 게이트웨이(GW) 설정', sty_h2))
        story.append(Paragraph(
            '툴바의 [GW 목록] 버튼으로 GW를 추가합니다. '
            '지도 우클릭 → [이 위치에 GW 추가]로 바로 추가 가능합니다. '
            'GW/Node 마커는 드래그로 위치를 변경할 수 있으며, '
            '이동 완료 시 커버리지가 자동 재분석됩니다.', sty_body))
        _img('gw_list', '그림 4.2 GW 목록 창')
        gw_data = [
            ['파라미터', '설명', '기본값'],
            ['Callsign', '게이트웨이 식별자',  'GW1, GW2...'],
            ['Pt (dBm)', '송신 출력',          '14.0 dBm'],
            ['Gt (dBi)', '안테나 이득',        '2.15 dBi'],
            ['Lt (dB)',  '케이블 손실',        '0.0 dB'],
            ['hb (m)',   'GW 안테나 높이',     '15.0 m'],
        ]
        _tbl(gw_data, [40*mm, 75*mm, 55*mm])

        story.append(Paragraph('4.4 전파 모델 설정', sty_h2))
        story.append(Paragraph(
            '[설정] → [전파 탭]에서 전파 모델과 환경을 선택합니다.', sty_body))
        prop_data = [
            ['설정 항목', '옵션/범위'],
            ['전파 모델',  'SmartCity LoRaScape Model / COST-231 Hata Model'],
            ['전파 환경',  'Auto(DSM 자동분류) / Dense Urban / Urban / Suburban / Open'],
            ['반송 주파수','400~2000 MHz (기본: 915 MHz)'],
            ['DEM 샘플 수','20~500개 (기본: 100개, 높을수록 정확하나 느림)'],
        ]
        _tbl(prop_data, [55*mm, 115*mm])

        story.append(Paragraph('4.5 우클릭 컨텍스트 메뉴', sty_h2))
        story.append(Paragraph(
            '지도에서 우클릭하면 현재 위치 기준의 빠른 작업 메뉴가 표시됩니다.', sty_body))
        ctx_data = [
            ['메뉴 항목', '기능'],
            ['이 위치에 GW 추가',     '클릭 위치에 GW 즉시 생성 (설정값 파라미터 적용)'],
            ['이 위치에 단말기 추가', '클릭 위치에 Node 즉시 생성'],
            ['커버리지 분석 실행',    '활성 GW 전체로 커버리지 분석 즉시 실행'],
            ['히트맵 계산',           '활성 GW 전체 히트맵 즉시 계산'],
            ['거리 측정 시작/종료',   '클릭 위치를 첫 점으로 거리 측정 시작'],
            ['측정 초기화',           '거리 측정 점 전체 삭제'],
            ['좌표 복사',             '위도/경도 또는 GeoJSON 형식으로 클립보드 복사'],
        ]
        _tbl(ctx_data, [65*mm, 105*mm])

        story.append(PageBreak())

        # ══════════════════════════════════════════════════
        # 5장. 커버리지 분석
        # ══════════════════════════════════════════════════
        self.sig_progress.emit(38, '5장 커버리지 분석 작성 중...')
        story.append(Paragraph('5. 커버리지 분석 (Coverage Analysis)', sty_h1))
        _hr()

        story.append(Paragraph('5.1 수신 전력 계산', sty_h2))
        story.append(Paragraph(
            '링크 버짓 기반으로 모든 Node × GW 조합의 수신전력을 병렬 계산합니다:', sty_body))
        story.append(Paragraph(
            'Pr = Pt + Gt - Lt - PL + Gr - Lr - Indoor_Loss  [dBm]', sty_note))
        formula_data = [
            ['파라미터', '설명'],
            ['Pt',          'GW 송신 출력 (dBm)'],
            ['Gt / Lt',     'GW 안테나 이득 / 케이블 손실 (dBi / dB)'],
            ['PL',          '경로 손실 — 전파 모델 기반 계산 (dB)'],
            ['Gr / Lr',     '단말 수신 이득 / 손실 (dBi / dB)'],
            ['Indoor_Loss', '실내 투과 손실 (dB)'],
        ]
        _tbl(formula_data, [30*mm, 140*mm])

        story.append(Paragraph('5.2 커버리지 맵', sty_h2))
        story.append(Paragraph(
            '[▶ 선택 커버리지] 버튼 클릭 시 히트맵과 커버리지 분석이 순차 실행됩니다. '
            '레이어 컨트롤에서 각 레이어를 독립적으로 켜고 끌 수 있습니다.', sty_body))
        _img('heatmap', '그림 5.1 커버리지 히트맵 및 수신전력 분포')
        layers = [
            ('전파 세기 (격자)',   'GW 주변 전파 강도를 격자 기반으로 시각화 (참고용)'),
            ('수신전력 분포',     '커버리지 분석 결과 기반 CircleMarker — Node 마커와 100% 일치'),
            ('중첩 커버 영역',    '2개 이상 GW에 커버되는 Node 표시 (보라색)'),
            ('음영 지역 (미커버)','미커버 Node 위치 (빨간색)'),
            ('등고선',            '수신전력 등고선 (-90 ~ -120 dBm, 색상 범례 연동)'),
            ('SF 레이어',         'SF7~SF12별 커버리지 경계선'),
        ]
        layer_data = [['레이어', '설명']] + [[n, d] for n, d in layers]
        _tbl(layer_data, [55*mm, 115*mm])

        story.append(Paragraph('5.3 통신 가능 영역', sty_h2))
        story.append(Paragraph(
            '단말의 최소 수신 감도(min_rx_dbm) 이상인 경우 커버로 판단합니다. '
            'RSSI Threshold 기준과 SNR 기준을 동시에 적용합니다.', sty_body))

        story.append(Paragraph('5.4 결과 지표', sty_h2))
        _img('result_panel', '그림 5.2 분석 결과 패널')
        metrics = [
            ('전체 커버리지 (%)',   '커버된 Node / 전체 Node × 100'),
            ('중첩 커버 (%)',       '2개 이상 GW에 커버되는 Node / 커버 Node × 100'),
            ('단독 커버 (%)',       '정확히 1개 GW에만 커버되는 Node / 전체 Node × 100'),
            ('음영 지역 (개)',      '미커버 Node 수'),
            ('다중 GW 연결 (개)',   '2개 이상 GW 수신 가능 Node 수'),
            ('셀 전체 성공율 (%)',  '커버 Node 중 SNR 마진 > 0 dB 비율'),
            ('셀 경계 성공율 (%)',  'SNR 마진 < 3 dB Node 중 SNR 마진 > 0 dB 비율'),
            ('평균 SNR (dB)',       '커버 Node 평균 신호 대 잡음비'),
            ('평균 SNR 마진 (dB)', '커버 Node 평균 SNR - SF별 임계값'),
            ('평균 ToA (ms)',       'ADR SF 기준 평균 패킷 공중 전송 시간'),
            ('평균 부하 (%)',       'GW 채널 용량 대비 평균 트래픽 사용률'),
            ('평균 PDR (%)',        'Pure ALOHA 기반 평균 패킷 성공률'),
        ]
        metric_data = [['지표', '설명']] + [[n, d] for n, d in metrics]
        _tbl(metric_data, [55*mm, 115*mm])

        story.append(Paragraph('5.5 ADR (Adaptive Data Rate)', sty_h2))
        story.append(Paragraph(
            '수신전력 기반으로 각 Node의 최적 SF를 자동 결정합니다. '
            '낮은 SF는 빠른 전송 속도, 높은 SF는 긴 전달 거리를 의미합니다.', sty_body))
        adr_data = [
            ['SF', '수신 감도 (dBm)', 'ToA (ms, 20B)', '특징'],
            ['SF7',  '-123.0', '61.7',   '고속, 단거리'],
            ['SF8',  '-126.0', '123.4',  ''],
            ['SF9',  '-129.0', '246.8',  ''],
            ['SF10', '-132.0', '493.5',  ''],
            ['SF11', '-134.5', '987.1',  ''],
            ['SF12', '-137.0', '1974.1', '저속, 장거리'],
        ]
        _tbl(adr_data, [20*mm, 42*mm, 42*mm, 66*mm])

        story.append(Paragraph('5.6 매크로 다이버시티', sty_h2))
        story.append(Paragraph(
            '복수 GW에서 동시에 수신 가능한 Node는 신호를 선형 합산하여 '
            '실효 수신전력이 향상됩니다.', sty_body))
        story.append(Paragraph(
            'Pr_macro = 10 * log10(Σ 10^(Pr_i/10))  [dBm]', sty_note))

        story.append(PageBreak())

        # ══════════════════════════════════════════════════
        # 6장. 간섭 분석
        # ══════════════════════════════════════════════════
        self.sig_progress.emit(50, '6장 간섭 분석 작성 중...')
        story.append(Paragraph('6. 간섭 분석 (Interference Analysis)', sty_h1))
        _hr()

        story.append(Paragraph('6.1 LoRa 특성 반영', sty_h2))
        story.append(Paragraph(
            'LoRa의 직교 Spreading Factor(SF) 특성을 반영하여 '
            'SNR 기반 통신 성공율을 계산합니다.', sty_body))

        story.append(Paragraph('6.2 SNR 기반 통신 성공율', sty_h2))
        story.append(Paragraph(
            'LoRa SF별 SNR 임계값과 수신 신호의 SNR 마진으로 통신 성공 여부를 판단합니다.',
            sty_body))
        story.append(Paragraph(
            'SNR = Pr - (열잡음 + 잡음지수)  [dB]', sty_note))
        story.append(Paragraph(
            '열잡음 = -174 + 10*log10(BW_Hz)  [dBm]  '
            '(125kHz 기준: -174 + 51.0 = -123.0 dBm)', sty_note))
        story.append(Paragraph(
            'SNR 마진 = SNR - SF별 SNR 임계값  [dB]', sty_note))

        sf_data = [
            ['SF', 'SNR 임계값 (dB)', '수신 감도 (dBm)', '통신 성공 조건'],
            ['SF7',  '-7.5',  '-123.0', 'SNR 마진 > 0 dB'],
            ['SF8',  '-10.0', '-126.0', 'SNR 마진 > 0 dB'],
            ['SF9',  '-12.5', '-129.0', 'SNR 마진 > 0 dB'],
            ['SF10', '-15.0', '-132.0', 'SNR 마진 > 0 dB'],
            ['SF11', '-17.5', '-134.5', 'SNR 마진 > 0 dB'],
            ['SF12', '-20.0', '-137.0', 'SNR 마진 > 0 dB'],
        ]
        _tbl(sf_data, [20*mm, 35*mm, 40*mm, 75*mm])

        story.append(Paragraph('■ 마진 설정', sty_h2))
        rate_data = [
            ['성공율 지표', '정의', '기준'],
            ['셀 전체 통신 성공율',
             '커버된 Node 중 SNR 마진 > 0 dB 비율', '95% 이상 권장'],
            ['셀 경계 통신 성공율',
             'SNR 마진 < 3 dB Node 중 SNR 마진 > 0 dB 비율', '80% 이상 권장'],
        ]
        _tbl(rate_data, [55*mm, 75*mm, 40*mm])
        _img('snr_setting', '그림 6.1 SNR/마진 설정 탭')

        story.append(Paragraph('6.3 트래픽 용량 분석 (ALOHA 기반)', sty_h2))
        story.append(Paragraph(
            'LoRa GW의 채널 용량과 단말 전송 부하를 분석하여 '
            'GW 과부하 여부와 패킷 성공률(PDR)을 계산합니다.', sty_body))
        story.append(Paragraph(
            '채널 용량 = 3,600,000 ms/h × 8채널 × 1% duty = 288,000 ms/h',
            sty_note))
        story.append(Paragraph(
            'G = GW 시간당 총 ToA / (3,600,000 × 8채널)', sty_note))
        story.append(Paragraph(
            'PDR = e^(-2G) × 100%  [Pure ALOHA]', sty_note))
        trf_data = [
            ['지표', '설명', '기준'],
            ['트래픽 부하 (%)', 'GW 채널 용량 대비 시간당 ToA 사용률', '80% 미만 권장'],
            ['PDR (%)',         'Pure ALOHA 기반 패킷 성공률',          '90% 이상 권장'],
            ['과부하 GW',       '부하 ≥ 80% GW 수',                     '0개 권장'],
        ]
        _tbl(trf_data, [40*mm, 85*mm, 45*mm])

        story.append(PageBreak())

        # ══════════════════════════════════════════════════
        # 7장. GW 최적화
        # ══════════════════════════════════════════════════
        self.sig_progress.emit(62, '7장 GW 최적화 작성 중...')
        story.append(Paragraph('7. GW 최적화 (Gateway Optimization)', sty_h1))
        _hr()

        story.append(Paragraph('7.1 목표 정의', sty_h2))
        story.append(Paragraph(
            '최소한의 GW로 최대한 많은 단말을 안정적으로 연결하면서 '
            '간섭과 비용을 최소화하는 것을 목표로 합니다.', sty_body))
        for item in [
            '최소 GW 수 탐색 — Greedy + ILP/GA 최적화',
            '간섭 최소화 — Path Loss 제약 조건 반영',
            '트래픽 처리 용량 제한 반영',
        ]:
            story.append(Paragraph(f'▪ {item}', sty_bullet))

        story.append(Paragraph('7.2 알고리즘 선택', sty_h2))
        algo_data = [
            ['알고리즘', '특징', '권장 상황'],
            ['K-means 클러스터링',
             '빠르고 안정적\nGreedy + K-means 조합\nILP 최적화 시도',
             'Node 수 500개 이하\n빠른 결과 필요 시'],
            ['유전 알고리즘 (GA)',
             '정밀 최적화\n세대/인구 수 파라미터 조절\n더 적은 GW 탐색 가능',
             'Node 수 많을 때\n최소 GW 수 탐색 필요 시'],
        ]
        _tbl(algo_data, [45*mm, 70*mm, 55*mm])

        story.append(Paragraph('7.3 최적화 단계 (공통)', sty_h2))
        steps = [
            ('Step 1', 'station↔station 연결 수 기반 GW 후보 우선순위 정렬'),
            ('Step 2', 'GW↔station 커버 집합 계산 (병렬 처리, 거리 필터 적용)'),
            ('Step 3', 'Greedy Set Cover — 연결 수 많은 순으로 GW 초기 배치'),
            ('Step 4', 'K-means 클러스터링 (k = Greedy GW 수)'),
            ('Step 5', 'GW 위치 확정 — 클러스터 무게중심 → 최근접 station'),
            ('Step 6', '커버리지 검증'),
            ('Step 7', '미커버 Node → GW 강제 추가'),
            ('Step 8', 'ILP 또는 GA 최적화 — GW 수 최소화'),
            ('Step 9', '소규모 GW 제거 (min_cover 기준 미만 GW 제거)'),
        ]
        step_data = [['단계', '설명']] + [[s, d] for s, d in steps]
        _tbl(step_data, [25*mm, 145*mm])

        story.append(Paragraph('7.4 제약 조건', sty_h2))
        const_data = [
            ['제약 조건', '값', '설명'],
            ['Path Loss 한계', 'PL ≤ PL_limit',    'EIRP/수신감도 기반 자동 계산'],
            ['수신 감도 기준', 'RSSI ≥ min_rx_dbm', '단말별 최소 수신 감도'],
            ['GW 최소 커버 수', '기본 3개 이상',     '설정 창에서 변경 가능'],
            ['GA 세대 수',     '기본 50세대',        '많을수록 정밀하나 느림'],
            ['GA 인구 수',     '기본 30개체',        '많을수록 탐색 범위 넓음'],
        ]
        _tbl(const_data, [45*mm, 55*mm, 70*mm])
        _img('optimize', '그림 7.1 GW 최적 배치 창')

        story.append(Paragraph('7.5 결과', sty_h2))
        story.append(Paragraph(
            '최적화 완료 후 지도에 OPT-GW 마커로 결과가 표시되며, '
            '커버리지 분석과 트래픽 용량 분석이 자동으로 실행됩니다. '
            '결과는 히스토리에 자동 저장되어 시나리오 비교에 활용할 수 있습니다.', sty_body))
        for item in [
            '최적 GW 위치 좌표 — 지도에 OPT-GW1, OPT-GW2... 마커로 표시',
            'GW 개수 — 최소화된 GW 수',
            '커버리지 개선율 — 기존 배치 대비 결과 비교 (히스토리 활용)',
        ]:
            story.append(Paragraph(f'▪ {item}', sty_bullet))
        _img('optimize_result', '그림 7.2 GW 최적 배치 결과')

        story.append(PageBreak())

        # ══════════════════════════════════════════════════
        # 8장. 시각화
        # ══════════════════════════════════════════════════
        self.sig_progress.emit(74, '8장 시각화 작성 중...')
        story.append(Paragraph('8. 시각화 (Visualization)', sty_h1))
        _hr()

        story.append(Paragraph('8.1 지도 기반 UI', sty_h2))
        story.append(Paragraph(
            '레이어 컨트롤(우측 상단)에서 각 레이어를 독립적으로 켜고 끌 수 있습니다. '
            'GW/Node 마커는 드래그로 이동하며, 이동 완료 시 커버리지가 자동 재분석됩니다.',
            sty_body))
        for item in [
            '단말(Node), GW 위치 마커 표시',
            '커버리지 레이어 (격자 히트맵, 수신전력 분포, 등고선, SF)',
            '중첩 커버 영역 / 음영 지역 레이어',
        ]:
            story.append(Paragraph(f'▪ {item}', sty_bullet))
        _img('map_full', '그림 8.1 지도 기반 UI 전체 화면')

        story.append(Paragraph('8.2 그래프', sty_h2))
        story.append(Paragraph(
            '툴바 [그래프] 버튼으로 4탭 분석 그래프 창을 엽니다. '
            '[이미지 저장] 버튼으로 PNG 저장 가능합니다.', sty_body))
        graph_items = [
            ('RSSI 분포',  '커버/미커버 Node 수신전력 히스토그램, SF별 감도 기준선, 평균선'),
            ('중첩도 분포','GW별 담당 Node 수 막대, 수신 GW 수별 Node 분포'),
            ('SF 분포',    'ADR SF 파이 차트, SF별 커버리지 비율 가로 막대'),
            ('SNR 분포',   'SNR 히스토그램, SNR 마진 성공/실패 분포 히스토그램'),
        ]
        graph_data = [['탭', '내용']] + [[n, d] for n, d in graph_items]
        _tbl(graph_data, [35*mm, 135*mm])
        _img('graph', '그림 8.2 분석 결과 그래프 창')

        story.append(Paragraph('8.3 결과 비교 창', sty_h2))
        story.append(Paragraph(
            '커버리지 분석을 실행할 때마다 결과가 히스토리에 자동 저장됩니다 (최대 10개). '
            '툴바 [결과 비교] 버튼으로 비교 창을 엽니다.', sty_body))
        compare_data = [
            ['기능', '설명'],
            ['자동 히스토리 저장', '분석 완료 시 GW/Node/결과 스냅샷 자동 저장'],
            ['이름 변경',          '시나리오별 의미 있는 이름 지정 가능'],
            ['2개 시나리오 비교',  '22개 지표를 색상 구분 테이블로 비교 (개선/악화)'],
            ['스냅샷 복원',        '선택한 히스토리 결과를 메인 창에 즉시 복원'],
        ]
        _tbl(compare_data, [55*mm, 115*mm])

        story.append(Paragraph('8.4 리포트 출력', sty_h2))
        story.append(Paragraph(
            '툴바 [리포트] 버튼으로 리포트 생성 창을 엽니다.', sty_body))
        report_items = [
            ('Excel (.xlsx)', '커버리지 요약 / GW 목록 / Node 목록 / SF 분포 4개 시트'),
            ('PDF (.pdf)',    '한글 폰트 지원, 커버리지 요약 / GW / Node 상세 테이블'),
            ('사용자 매뉴얼', '현재 분석 결과 및 스크린샷 포함 전체 매뉴얼 자동 생성'),
        ]
        report_data = [['출력 형식', '내용']] + [[n, d] for n, d in report_items]
        _tbl(report_data, [45*mm, 125*mm])
        _img('report', '그림 8.3 리포트 생성 창')

        story.append(PageBreak())

        # ══════════════════════════════════════════════════
        # 9장. 운영 및 유지보수
        # ══════════════════════════════════════════════════
        self.sig_progress.emit(84, '9장 운영/유지보수 작성 중...')
        story.append(Paragraph('9. 운영 및 유지보수', sty_h1))
        _hr()

        story.append(Paragraph('9.1 데이터 관리', sty_h2))
        for item in [
            'GIS 업데이트: data/ 폴더의 Shapefile 및 DEM 파일 교체 후 재실행',
            '단말 정보 갱신: 단말 목록 창의 CSV 가져오기/내보내기 기능 활용',
            '세션 자동 저장: 종료 시 GW/Node 위치가 session.json에 자동 저장/복원',
            '결과 저장/불러오기: JSON 형식으로 분석 결과 영구 보관 가능',
        ]:
            story.append(Paragraph(f'▪ {item}', sty_bullet))

        story.append(Paragraph('9.2 시스템 관리', sty_h2))
        for item in [
            '설정 저장: settings.json에 전역 설정 자동 저장 (전파 모델, GW/Node 기본값 등)',
            '결과 비교: 분석 히스토리 최대 10개 보관, 세션 종료 시 초기화',
            '히트맵 속도: 설정 → 격자 간격 0.0005° 이상 권장 (자동 조정 기능 내장)',
            '성능 모니터링: 콘솔 로그에서 커버리지 분석 진행률 및 결과 확인 가능',
        ]:
            story.append(Paragraph(f'▪ {item}', sty_bullet))

        story.append(Paragraph('9.3 장애 대응', sty_h2))
        trouble_data = [
            ['증상', '원인', '해결'],
            ['히트맵 계산 느림',
             'step 값이 너무 작음',
             '설정 → 격자 간격 0.0005° 이상으로 설정'],
            ['커버리지 분석 오류',
             '공간 데이터 로드 실패',
             'data/ 폴더의 SHP/DEM 파일 경로 확인'],
            ['그래프 한글 깨짐',
             'matplotlib 폰트 설정 오류',
             'Windows: Malgun Gothic 폰트 설치 확인'],
            ['GA 최적화 느림',
             '세대/인구 수 과다',
             '세대 수 30, 인구 수 20으로 축소'],
            ['히트맵이 Node와 불일치',
             '히트맵 h_station과 Node hm_m 차이',
             '설정 → Node 기본 높이(nd_hm_m) 확인'],
            ['PDR이 매우 낮음',
             'GW 과부하 (부하 ≥ 80%)',
             'GW 추가 또는 단말 전송 주기 늘리기'],
            ['데이터 검증 오류',
             '비정상 파라미터',
             '단말/GW 목록에서 좌표 및 파라미터 재확인'],
            ['재시뮬레이션 필요',
             '설정 변경 후 결과 미반영',
             '커버리지 분석 재실행 또는 히트맵 재계산'],
        ]
        _tbl(trouble_data, [40*mm, 55*mm, 75*mm])

        story.append(PageBreak())

        # ══════════════════════════════════════════════════
        # 10장. 부록
        # ══════════════════════════════════════════════════
        self.sig_progress.emit(92, '10장 부록 작성 중...')
        story.append(Paragraph('10. 부록', sty_h1))
        _hr()

        story.append(Paragraph('10.1 용어 정의', sty_h2))
        terms = [
            ('LoRa',               'Long Range — 장거리 저전력 무선 통신 기술'),
            ('LPWAN',              'Low Power Wide Area Network — 저전력 광역 통신망'),
            ('GW (Gateway)',       '게이트웨이 — LoRa 신호를 수신하는 기지국'),
            ('Node (단말)',        'LoRa 신호를 송신하는 IoT 단말'),
            ('SF (Spreading Factor)', '확산 지수 — SF7(고속/단거리) ~ SF12(저속/장거리)'),
            ('RSSI',               'Received Signal Strength Indicator — 수신 신호 세기 (dBm)'),
            ('SNR',                'Signal to Noise Ratio — 신호 대 잡음비 (dB)'),
            ('SNR 마진',           'SNR - SF별 임계값 (dB), 양수이면 통신 성공'),
            ('PL',                 'Path Loss — 경로 손실 (dB)'),
            ('EIRP',               'Effective Isotropic Radiated Power — 등가 등방 복사 전력 (dBm)'),
            ('ADR',                'Adaptive Data Rate — 수신전력 기반 자동 SF 조정'),
            ('ToA',                'Time on Air — 패킷 공중 전송 시간 (ms)'),
            ('PDR',                'Packet Delivery Ratio — 패킷 성공률 (%)'),
            ('ALOHA',              '랜덤 접근 통신 프로토콜 — PDR = e^(-2G)'),
            ('DEM',                'Digital Elevation Model — 수치 고도 모델'),
            ('DSM',                'Digital Surface Model — 건물 포함 수치 표면 모델'),
            ('K-means',            '클러스터링 기반 GW 초기 배치 알고리즘'),
            ('GA',                 'Genetic Algorithm — 유전 알고리즘, GW 수 최소화'),
            ('ILP',                'Integer Linear Programming — 정수 선형 계획법'),
            ('매크로 다이버시티',  '복수 GW 신호 합산으로 실효 수신전력 향상'),
        ]
        term_data = [['용어', '정의']] + [[t, d] for t, d in terms]
        _tbl(term_data, [52*mm, 118*mm])

        story.append(Spacer(1, 5*mm))
        story.append(Paragraph('10.2 전파 모델 수식', sty_h2))
        story.append(Paragraph('▪ SmartCity LoRaScape Model', sty_h2))
        story.append(Paragraph(
            'BPL = 39.25 + 35.15*log10(fc) - 19.21*log10(hb) + '
            '(42.5 - 5.2*log10(hb)) * log10(d)  [dB]', sty_note))
        story.append(Paragraph('▪ COST-231 Hata Model', sty_h2))
        story.append(Paragraph(
            'L = 46.3 + 33.9*log10(fc) - 13.82*log10(hb) - a(hm) + '
            '(44.9 - 6.55*log10(hb)) * log10(d) + Cm  [dB]', sty_note))

        story.append(Spacer(1, 5*mm))
        story.append(Paragraph('10.3 주요 개선 이력', sty_h2))
        history_data = [
            ['버전', '주요 변경 사항'],
            ['v1.0', '기본 커버리지 분석, GW 목록, Node 목록'],
            ['v1.1', 'K-means/GA GW 최적화, 히트맵 생성'],
            ['v1.2', 'SNR 기반 통신 성공율, ADR SF 분포, 매크로 다이버시티'],
            ['v1.3', 'RSSI/SNR/SF/중첩도 그래프 창, 한글 폰트 지원'],
            ['v1.4', '트래픽 용량 분석 (ALOHA PDR), 우클릭 컨텍스트 메뉴'],
            ['v1.5', '시나리오 비교 기능 (히스토리 10개), 병렬 커버리지 계산'],
            ['v1.6', '격자 히트맵 + 분석 결과 CircleMarker 이중 레이어'],
        ]
        _tbl(history_data, [20*mm, 150*mm])

        self.sig_progress.emit(98, 'PDF 저장 중...')
        doc.build(story)
        self.sig_progress.emit(100, '완료')


class ManualWindow(QDialog):
    """사용자 매뉴얼 PDF 자동 생성 창."""

    def __init__(self, main_window, result=None, gws=None,
                 nodes=None, settings=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("사용자 매뉴얼 PDF 생성")
        self.setStyleSheet(STYLE_DLG)
        self.resize(480, 560)
        self.setWindowFlag(Qt.Window)

        self._main    = main_window
        self._result  = result
        self._gws     = gws   or []
        self._nodes   = nodes or []
        self._settings= settings or {}
        self._thread  = None
        self._worker  = None
        self._shots   = {}

        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(10)

        info = QLabel(
            "현재 실행 중인 화면을 자동 캡처하여 PDF 매뉴얼을 생성합니다.\n"
            "생성 전 각 창을 미리 열어두면 더 좋은 스크린샷이 포함됩니다.")
        info.setStyleSheet(f"color:{MUTED};font-size:11px;")
        info.setWordWrap(True)
        lay.addWidget(info)

        grp = QGroupBox("포함할 화면 선택")
        grp.setStyleSheet(
            f"QGroupBox{{color:{MUTED};border:1px solid {BORDER};"
            f"border-radius:6px;margin-top:6px;padding-top:8px;}}"
            f"QGroupBox::title{{subcontrol-origin:margin;left:8px;}}")
        chk_lay = QVBoxLayout(grp); chk_lay.setSpacing(4)

        self._chks = {}
        sections = [
            ('main',            '메인 화면 전체'),
            ('node_list',       '단말 목록 창'),
            ('gw_list',         'GW 목록 창'),
            ('heatmap',         '커버리지 히트맵'),
            ('result_panel',    '분석 결과 패널'),
            ('optimize',        'GW 최적 배치 창'),
            ('optimize_result', 'GW 최적 배치 결과'),
            ('graph',           '그래프 창'),
            ('report',          '리포트 창'),
            ('snr_setting',     'SNR/마진 설정'),
        ]
        for key, label in sections:
            chk = QCheckBox(label)
            chk.setChecked(True)
            chk.setStyleSheet(f"color:{TEXT};font-size:11px;")
            chk_lay.addWidget(chk)
            self._chks[key] = chk
        lay.addWidget(grp)

        btn_capture = QPushButton("📸 화면 캡처")
        btn_capture.setStyleSheet(BTN)
        btn_capture.clicked.connect(self._capture_screens)
        lay.addWidget(btn_capture)

        self.lbl_capture = QLabel("캡처된 화면: 0개")
        self.lbl_capture.setStyleSheet(f"color:{MUTED};font-size:10px;")
        lay.addWidget(self.lbl_capture)

        self.prog = QProgressBar()
        self.prog.setRange(0, 100); self.prog.setValue(0)
        self.prog.setFixedHeight(14)
        self.prog.setStyleSheet(
            f"QProgressBar{{background:{DARK};border:1px solid {BORDER};"
            f"border-radius:6px;text-align:center;color:{TEXT};"
            f"font-size:10px;}}"
            f"QProgressBar::chunk{{background:#4f8ef7;border-radius:6px;}}")
        self.prog.setTextVisible(True)
        self.prog.setFormat("%p%")
        lay.addWidget(self.prog)

        self.lbl_status = QLabel("화면을 캡처한 후 PDF를 생성하세요.")
        self.lbl_status.setStyleSheet(f"color:{MUTED};font-size:10px;")
        self.lbl_status.setWordWrap(True)
        lay.addWidget(self.lbl_status)

        btn_row = QHBoxLayout()
        self.btn_gen = QPushButton("📄 PDF 매뉴얼 생성")
        self.btn_gen.setStyleSheet(BTN_GEN)
        self.btn_gen.clicked.connect(self._generate)
        btn_close = QPushButton("닫기")
        btn_close.setStyleSheet(BTN)
        btn_close.clicked.connect(self.close)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        btn_row.addWidget(self.btn_gen)
        lay.addLayout(btn_row)

    def _capture_screens(self):
        self._shots = {}
        captured = 0

        if self._chks['main'].isChecked():
            self._shots['main']     = self._grab_widget(self._main)
            self._shots['map_full'] = self._grab_widget(
                getattr(self._main, 'map_w', self._main))
            captured += 1

        if self._chks['result_panel'].isChecked():
            rp = getattr(self._main, 'result_panel', None)
            if rp:
                self._shots['result_panel'] = self._grab_widget(rp)
                captured += 1

        win_map = {
            'node_list': '_node_win',
            'gw_list'  : '_gw_win',
            'optimize' : '_opt_win',
        }
        for key, attr in win_map.items():
            if self._chks[key].isChecked():
                win = getattr(self._main, attr, None)
                if win and win.isVisible():
                    self._shots[key] = self._grab_widget(win)
                    captured += 1

        for key in ['heatmap', 'optimize_result']:
            if self._chks.get(key) and self._chks[key].isChecked():
                self._shots[key] = self._grab_widget(
                    getattr(self._main, 'map_w', self._main))
                captured += 1

        for key in ['graph', 'report', 'snr_setting']:
            if self._chks.get(key) and self._chks[key].isChecked():
                if key not in self._shots:
                    self._shots[key] = self._grab_widget(self._main)
                    captured += 1

        self.lbl_capture.setText(f"캡처된 화면: {captured}개")
        self.lbl_status.setText(
            f"{captured}개 화면 캡처 완료. PDF 생성 버튼을 클릭하세요.")

    @staticmethod
    def _grab_widget(widget) -> QPixmap:
        try:
            return widget.grab()
        except Exception:
            px = QPixmap(800, 400)
            px.fill()
            return px

    def _generate(self):
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "매뉴얼 PDF 저장",
            "LoRaScape_User_Manual.pdf", "PDF (*.pdf)")
        if not path:
            return

        self.btn_gen.setEnabled(False)
        self.prog.setValue(0)
        self.lbl_status.setText("PDF 생성 중...")

        if not self._shots:
            self._capture_screens()

        w = ManualWorker(
            path        = path,
            screenshots = self._shots,
            result      = self._result,
            gws         = self._gws,
            nodes       = self._nodes,
            settings    = self._settings,
        )
        t = QThread()
        self._thread = t
        self._worker = w
        w.moveToThread(t)
        t.started.connect(w.run)
        w.sig_progress.connect(self._on_progress)
        w.sig_done.connect(self._on_done)
        w.sig_err.connect(self._on_err)
        w.sig_done.connect(t.quit)
        w.sig_err.connect(t.quit)
        t.start()

    def _on_progress(self, pct, msg):
        self.prog.setValue(pct)
        self.lbl_status.setText(msg)

    def _on_done(self, path):
        self.prog.setValue(100)
        self.lbl_status.setText(f"완료: {os.path.basename(path)}")
        self.btn_gen.setEnabled(True)
        QMessageBox.information(self, "완료",
            f"매뉴얼 PDF 생성 완료!\n{path}")

    def _on_err(self, msg):
        self.prog.setValue(0)
        self.lbl_status.setText("오류 발생 — 콘솔 확인")
        self.btn_gen.setEnabled(True)
        print(f"[MANUAL ERROR]\n{msg}")
        QMessageBox.critical(self, "오류",
            f"PDF 생성 실패.\n{msg[:300]}")