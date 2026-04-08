# LoRaWAN Gateway Placement Optimizer

스마트시티 IoT 단말기 대상 LoRaWAN 게이트웨이 최적 배치 시뮬레이터

## 개요

성남시 스마트시티 IoT 단말기(수면 감지 센서, 온도 센서 등)에 대해  
LoRaWAN Gateway를 최적 위치에 배치하여 단말기와의 통신 커버리지를  
분석하고 데이터 수집 효율을 극대화하는 시뮬레이션 도구입니다.

## 주요 기능

- **GW/Node 관리** — 추가, 삭제, 파라미터 편집, CSV 가져오기/내보내기
- **지도 기반 UI** — 마우스 클릭/드래그로 GW·단말기 위치 지정
- **커버리지 분석** — DEM 기반 지형 회절 전파 모델 (Song's + Deygout)
- **히트맵 표시** — dBm 레벨별 색상, SF7~SF12 레이어 ON/OFF
- **지형 단면도** — GW↔Node 간 LOS, Fresnel 존 시각화
- **링크 버짓** — SF별 수신 마진, GW×Node 전체 매트릭스
- **GW 최적 배치** — Greedy Set Cover → K-means → ILP 자동 최적화
- **거리 분석** — GW↔Node 간 거리, 방위각 계산
- **연결 단말기 보기** — GW별 연결 단말기 상세 정보
- **연결 GW 보기** — 단말기별 연결 GW 상세 정보
- **Node 랜덤 배치** — 성남시 경계 내 균일 랜덤 배치
- **우클릭 추가** — 지도에서 우클릭으로 GW/단말기 즉시 추가

## 설치 방법

### 요구사항

- Python 3.9 이상
- Windows 10/11 (권장)

### 의존성 설치

```bash
pip install -r requirements.txt
```

### 실행

```bash
python main.py
```

## 프로젝트 구조
```bash
lora_coverage_planner/
├── main.py
├── core/
│   ├── coverage.py          # 커버리지 계산 엔진
│   ├── dem_loader.py        # DEM/SHP 로더
│   ├── propagation.py       # 전파 모델 (Song's + Deygout)
│   ├── gw_optimizer.py      # GW 최적 배치 알고리즘
│   └── link_matrix.py       # 링크 행렬 계산
├── ui/
│   ├── main_window.py       # 메인 윈도우
│   ├── map_widget.py        # Folium 지도 위젯
│   ├── gw_list_window.py    # GW 목록 창
│   ├── node_list_window.py  # Node 목록 창
│   ├── dialogs.py           # 파라미터 편집 다이얼로그
│   ├── profile_window.py    # 지형 단면도
│   ├── linkbudget_window.py # 링크 버짓
│   ├── gw_optimize_window.py# GW 최적 배치
│   ├── distance_window.py   # 거리 분석
│   ├── gw_node_detail_window.py  # GW별 연결 단말기
│   └── node_gw_detail_window.py  # 단말기별 연결 GW
└── data/
├── Outline_Seongnam_3857.shp  # 성남시 경계 (별도 배포)
└── dem_build_seongnam_3857-2.img  # DEM 고도 데이터 (별도 배포)
```

## 전파 모델

- **Song's Model** — 도심 환경 반영 경로 손실
- **Deygout 회절** — DEM 기반 지형 회절 손실 계산
- **주파수** — 915 MHz (LoRa)
- **SF7~SF12** — 수신 감도 -123 ~ -137 dBm

## 데이터 파일

DEM 및 SHP 파일은 용량 문제로 별도 제공됩니다.  
[다운로드 링크](#) <!-- 추후 업데이트 -->

파일을 `data/` 폴더에 넣고 실행하세요.

## 라이선스

MIT License

## 기여

Issues 및 Pull Request 환영합니다.