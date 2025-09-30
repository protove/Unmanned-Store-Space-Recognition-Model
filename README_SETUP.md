# Unmanned Store Space Recognition Model - Setup Guide

이 프로젝트는 무인 매장의 공간 인식을 위한 모델로, DepthPro와 LangSAM을 활용합니다.

## 시스템 요구사항

- Python 3.9 이상
- CUDA 11.8 이상 (GPU 사용 시)
- Git

## 설치 방법

### 1. 기본 라이브러리 설치

```bash
# 필요한 Python 패키지 설치
pip install -r requirements.txt
```

### 2. SAM2 설치 (필수)

SAM2는 별도로 설치해야 합니다:

```bash
# 자동 설치 스크립트 실행
./install_sam2.sh

# 또는 수동 설치
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
cd ..
```

### 3. GPU 지원 (선택사항)

CUDA를 사용하는 경우:

```bash
# CUDA 버전에 맞는 PyTorch 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. 모델 체크포인트 다운로드

#### DepthPro 모델
```bash
# depth_pro 체크포인트 다운로드 (필요시)
mkdir -p depth_pro/ml-depth-pro/checkpoints
# DepthPro 모델 가중치를 checkpoints 폴더에 배치
```

#### SAM2 모델 체크포인트
SAM2 모델은 자동으로 다운로드되지만, 필요에 따라 수동으로 다운로드할 수 있습니다.

## 프로젝트 구조

```
Unmanned-Store-Space-Recognition-Model/
├── depth_pro/                    # DepthPro 관련 코드
│   ├── ml-depth-pro/             # DepthPro 모델 구현
│   ├── HoughLines.py             # 허프 변환 라인 검출
│   └── horizon_highlighted.py    # 수평선 강조 처리
├── lang-segment-anything/        # LangSAM 관련 코드
│   ├── lang_sam/                 # LangSAM 모델 구현
│   │   ├── models/               # GDINO, SAM 모델
│   │   └── utils.py              # 유틸리티 함수
│   └── final_tool/               # 최종 도구들
│       ├── space_detection.py    # 공간 감지
│       ├── person_distance_detection.py  # 사람 거리 감지
│       └── mono_test.py          # 단일 테스트
├── requirements.txt              # Python 패키지 의존성
├── install_sam2.sh              # SAM2 설치 스크립트
└── README_SETUP.md              # 이 파일
```

## 주요 기능별 사용법

### 1. 공간 감지 (Space Detection)
```bash
cd lang-segment-anything/final_tool
python space_detection.py
```

### 2. 사람 거리 감지 (Person Distance Detection)
```bash
cd lang-segment-anything/final_tool
python person_distance_detection.py
```

### 3. DepthPro 배치 처리
```bash
cd depth_pro/ml-depth-pro
python depth_pro_batch.py --input /path/to/images --output /path/to/output
```

### 4. 비디오 프레임 추출
```bash
cd lang-segment-anything/final_tool
python save_frame.py
```

### 5. 이미지 리사이징
```bash
cd lang-segment-anything/final_tool
python img_resize.py
```

## 환경 변수 설정

CUDA 메모리 관리를 위해 다음 환경 변수를 설정할 수 있습니다:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 트러블슈팅

### 일반적인 문제들

1. **SAM2 import 오류**
   - SAM2가 제대로 설치되었는지 확인
   - `pip list | grep sam2` 명령으로 설치 확인

2. **CUDA 메모리 부족**
   - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 환경 변수 설정
   - 배치 크기 줄이기

3. **모델 체크포인트 오류**
   - 모델 가중치 파일이 올바른 경로에 있는지 확인
   - 인터넷 연결 확인 (자동 다운로드용)

4. **PIL/Pillow 오류**
   - `pip install --upgrade pillow pillow-heif` 실행

5. **Transformers 버전 호환성**
   - `pip install --upgrade transformers` 실행

## 라이선스

이 프로젝트는 여러 오픈소스 프로젝트를 기반으로 합니다:
- DepthPro: Apple의 ML-Depth-Pro
- SAM2: Meta의 Segment Anything 2
- GroundingDINO: IDEA Research의 Grounding DINO

각 구성요소의 라이선스를 확인하여 사용하시기 바랍니다.
