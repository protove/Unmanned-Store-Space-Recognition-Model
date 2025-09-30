# Unmanned Store Space Recognition Model

무인 매장의 공간 인식 및 객체 감지를 위한 딥러닝 모델입니다. Apple의 DepthPro와 Meta의 Segment Anything 2 (SAM2)를 활용하여 매장 내 공간 분석, 사람 거리 측정, 그리고 다양한 객체 감지 기능을 제공합니다.

## 주요 기능

- 🏪 **공간 감지**: 매장 내 디스플레이, 바닥, 기타 객체 영역 감지
- 👥 **사람 거리 측정**: 깊이 정보를 활용한 사람과의 거리 계산
- 📏 **깊이 맵 생성**: 단일 이미지로부터 정확한 깊이 정보 추출
- 🎯 **언어 기반 세그멘테이션**: 자연어 프롬프트를 통한 객체 감지
- 📹 **비디오 처리**: 동영상에서 프레임 추출 및 배치 처리

## 빠른 시작

### 설치

```bash
# 저장소 클론
git clone https://github.com/protove/Unmanned-Store-Space-Recognition-Model.git
cd Unmanned-Store-Space-Recognition-Model

# 의존성 설치
pip install -r requirements.txt

# SAM2 설치
./install_sam2.sh
```

### 사용 예시

```bash
# 공간 감지 실행
cd lang-segment-anything/final_tool
python space_detection.py

# 사람 거리 감지 실행
python person_distance_detection.py
```

## 자세한 설치 및 사용법

전체 설치 가이드와 사용법은 [README_SETUP.md](README_SETUP.md)를 참조하세요.

## 프로젝트 구조

```
├── depth_pro/                 # DepthPro 깊이 추정 모델
├── lang-segment-anything/     # LangSAM 세그멘테이션 모델
├── requirements.txt           # Python 의존성
├── install_sam2.sh           # SAM2 설치 스크립트
└── README_SETUP.md           # 상세 설치 가이드
```

## 기술 스택

- **DepthPro**: Apple의 단안 깊이 추정 모델
- **SAM2**: Meta의 Segment Anything 2
- **GroundingDINO**: IDEA Research의 객체 감지 모델
- **PyTorch**: 딥러닝 프레임워크
- **OpenCV**: 컴퓨터 비전 라이브러리

## 기여하기

이슈 리포트나 기능 제안은 GitHub Issues를 통해 해주세요.

## 라이선스

이 프로젝트는 여러 오픈소스 라이브러리를 기반으로 하며, 각각의 라이선스를 준수합니다.