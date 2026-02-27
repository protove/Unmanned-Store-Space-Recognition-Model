<div align="center">

# 🏪 무인 매장 공간 인식 모델
### Unmanned Store Space Recognition Model

**단일 RGB 이미지만으로 매장 내 진열 공간을 자동 인식하는 2단계 AI 파이프라인**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## 📌 프로젝트 개요

무인 매장에서는 **상품 보충 시점 파악**과 **공간 효율 최적화**가 핵심 과제입니다.
이 프로젝트는 **별도의 깊이 센서 없이** RGB 카메라 한 대만으로 매장 내 진열 공간을 자동 인식하고, 비어있는 공간과 진열대 구조를 정량화하는 AI 파이프라인을 구축했습니다.

### 핵심 문제 정의

| 기존 방식 | 본 프로젝트 |
|-----------|------------|
| 깊이 카메라(LiDAR, ToF) 필요 → 고비용 | 단안 RGB 카메라만 사용 |
| 룰 기반 이미지 처리 → 환경 변화에 취약 | 언어 기반 세그멘테이션 → 유연한 객체 인식 |
| 수동 공간 라벨링 필요 | 자연어 프롬프트로 즉시 적용 |

---

## 🎯 주요 기능

- **🔭 단안 깊이 추정** — Apple DepthPro로 2D 이미지에서 미터 단위 깊이 맵 생성
- **🗣️ 언어 기반 세그멘테이션** — `"shelf."`, `"display rack."` 등 자연어로 객체 영역 추출
- **📦 진열대 클러스터 분석** — 겹침 비율 기반 마스크 클러스터링으로 독립 진열 구역 식별
- **📏 거리별 공간 우선순위** — 깊이 정보로 가장 먼 구역을 자동 강조
- **🐳 완전 자동화 파이프라인** — Docker Compose 한 번으로 DepthPro → LangSAM 순서 실행

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT                                   │
│              test1.png, test2.png, ... testN.png                │
│                    (매장 내부 RGB 이미지)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │     Stage 1 : DepthPro       │  docker/Dockerfile.depthpro
              │   (Apple ML, ViT-L 기반)     │
              │                             │
              │  RGB → Metric Depth Map     │
              │  출력: *_depth.npy / .png   │
              └──────────────┬──────────────┘
                             │  depth_shared 볼륨
              ┌──────────────▼──────────────┐
              │     Stage 2 : LangSAM        │  docker/Dockerfile.langsam
              │  (GroundingDINO + SAM2)      │
              │                             │
              │  ① 바닥 마스크 생성          │
              │  ② 진열대 마스크 추출        │
              │  ③ 겹침 기반 클러스터링      │
              │  ④ 깊이 기반 우선순위 정렬   │
              └──────────────┬──────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                         OUTPUT                                  │
│  overlay/        — 원본 위에 마스크 시각화                       │
│  mask_only/      — 순수 마스크 레이어 (floor / display / top3)  │
│  json/           — 클러스터별 좌표·크기·깊이 메타데이터          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ 기술 스택

| 범주 | 기술 |
|------|------|
| **깊이 추정** | [Apple DepthPro](https://github.com/apple/ml-depth-pro) (ViT-Large, metric depth) |
| **객체 탐지** | [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) (language-guided detection) |
| **세그멘테이션** | [Meta SAM 2.1](https://github.com/facebookresearch/segment-anything-2) (hiera-small) |
| **프레임워크** | PyTorch 2.0+, torchvision |
| **영상 처리** | OpenCV, Pillow, NumPy |
| **컨테이너** | Docker Compose (NVIDIA GPU 지원) |
| **분석** | scikit-learn, scipy, matplotlib |

---

## 🚀 빠른 시작 — Docker 파이프라인 (권장)

### 사전 요구사항

- Docker + Docker Compose v2
- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) *(CPU 폴백 지원)*

### 1단계 — 저장소 클론

```bash
git clone https://github.com/protove/Unmanned-Store-Space-Recognition-Model.git
cd Unmanned-Store-Space-Recognition-Model
```

### 2단계 — 입력 이미지 배치

```bash
mkdir -p data/input
# 매장 이미지를 아래 이름으로 저장 (N = SPACE_IMAGE_COUNT 값)
cp your_image1.png data/input/test1.png
cp your_image2.png data/input/test2.png
cp your_image3.png data/input/test3.png
```

### 3단계 — 파이프라인 실행

```bash
chmod +x pipeline.sh
./pipeline.sh
```

`pipeline.sh`는 자동으로:
1. `data/checkpoints/depth_pro.pt` 존재 여부 확인 → 없으면 다운로드 안내 (약 2.5 GB)
2. Docker 이미지 빌드 (최초 1회, 이후 캐시 사용)
3. **DepthPro → LangSAM** 순서로 컨테이너 실행
4. 결과를 `data/output/` 에 저장

### 이미지 수 변경

```bash
SPACE_IMAGE_COUNT=5 ./pipeline.sh    # test1.png ~ test5.png 처리
```

---

## 📁 프로젝트 구조

```
Unmanned-Store-Space-Recognition-Model/
│
├── docker/
│   ├── Dockerfile.depthpro        # Stage 1 이미지 (CUDA 11.8 + DepthPro)
│   └── Dockerfile.langsam         # Stage 2 이미지 (SAM2 + GroundingDINO)
│
├── docker-compose.yml             # 파이프라인 오케스트레이션
├── pipeline.sh                    # 원클릭 실행 스크립트
│
├── depth_pro/
│   └── ml-depth-pro/
│       ├── src/depth_pro/         # Apple DepthPro 소스 패키지
│       └── depth_pro_batch.py     # 배치 처리 진입점 (CLI + 함수)
│
├── lang-segment-anything/
│   ├── lang_sam/                  # LangSAM 핵심 모듈
│   │   ├── lang_sam.py            # LangSAM 클래스 (GDINO + SAM2 통합)
│   │   └── utils.py
│   └── final_tool/
│       ├── config.py              # 중앙 설정 (임계값, 경로, 환경변수)
│       ├── space_detection.py     # 메인 파이프라인 스크립트
│       └── person_distance_detection_ver2.py
│
├── requirements.txt
└── .dockerignore
```

---

## ⚙️ 상세 동작 설명

### Stage 1 — 깊이 맵 생성 (DepthPro)

Apple의 DepthPro는 **단안(monocular) 이미지에서 FOV를 자동 추정**하며 절대 깊이(metric depth)를 픽셀 단위로 생성합니다.

```python
# depth_pro_batch.py 핵심 흐름
model, transform = create_model_and_transforms(config, device)
result = model.infer(transform(image).unsqueeze(0))
depth_map = result["depth"]          # [H, W] float32, 단위: 미터
focal_length = result["focallength_px"]
```

출력 파일:
- `{name}_depth.npy` — NumPy 배열 (Stage 2 입력)
- `{name}_depth.png` — viridis 컬러맵 시각화
- `{name}_meta.txt` — focal length, min/max/mean depth

### Stage 2 — 공간 인식 (LangSAM)

```
자연어 프롬프트 → GroundingDINO (바운딩 박스) → SAM2 (픽셀 마스크)
```

처리 단계:
1. **바닥 분리** — `"floor."`, `"ground."` 프롬프트로 바닥 마스크 추출 후 진열대 분석에서 제외
2. **진열대 마스크 생성** — 10개 프롬프트 (`"shelf."`, `"display rack."` 등)로 다중 마스크 수집
3. **필터링** — 크기 임계값(`5,000 px`), 공간 분산(`95,000`) 기준으로 노이즈 제거
4. **클러스터링** — IoU 기반 겹침 비율(`0.9`)로 동일 진열대 마스크 통합
5. **깊이 우선순위** — 각 클러스터 평균 깊이 계산 → 가장 먼 구역 강조 표시

---

## 📊 출력 예시

```
data/output/
├── overlay/
│   └── test1_overlay.png          ← 원본 위에 컬러 마스크 오버레이
├── mask_only/
│   ├── floor/   test1_floor.png   ← 바닥 영역 마스크
│   ├── display/ test1_display.png ← 진열대 전체 마스크
│   ├── top3_masks/                ← 상위 3개 클러스터 (색상 구분)
│   └── classified_masks/          ← 최종 분류 결과
└── json/
    └── test1_result.json          ← 클러스터별 {중심점, 면적, 평균깊이}
```

---

## 🔧 로컬 실행 (Docker 없이)

```bash
# 의존성 설치
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything-2.git
pip install -e depth_pro/ml-depth-pro/

# DepthPro 체크포인트 다운로드
wget https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt \
     -P depth_pro/ml-depth-pro/checkpoints/

# Stage 1 실행
python depth_pro/ml-depth-pro/depth_pro_batch.py \
    --input  lang-segment-anything/assets/space_data/ \
    --output lang-segment-anything/assets/space_data/depth_map/

# Stage 2 실행
cd lang-segment-anything/final_tool
python space_detection.py
```

---

## 📐 환경변수 레퍼런스

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `INPUT_IMAGE_DIR` | `assets/space_data/` | 원본 이미지 폴더 |
| `DEPTH_MAP_DIR` | `assets/space_data/depth_map/` | 깊이 맵 NPY 폴더 |
| `OUTPUT_DIR` | `output/final/` | 결과 저장 폴더 |
| `SPACE_IMAGE_COUNT` | `3` | 처리할 이미지 수 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | GPU 메모리 최적화 |
| `HF_HOME` | `/data/hf_cache` | HuggingFace 캐시 경로 |

---

## 💻 하드웨어 요구사항

| 구성 | 최소 | 권장 |
|------|------|------|
| GPU VRAM | 8 GB | 12 GB 이상 |
| RAM | 16 GB | 32 GB |
| 저장공간 | 15 GB | 30 GB (캐시 포함) |
| CUDA | 11.8 | 12.x |

> GPU 없는 환경에서도 CPU 폴백으로 실행 가능 (처리 시간 대폭 증가)

---

## 🤝 기여 방법

1. 이 저장소를 Fork합니다
2. 기능 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋합니다 (`git commit -m 'feat: Add amazing feature'`)
4. 브랜치에 Push합니다 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성합니다

---

## 📄 라이선스 및 참조 모델

이 프로젝트는 다음 오픈소스 모델을 활용합니다:

- **[Apple DepthPro](https://github.com/apple/ml-depth-pro)** — Apple Inc., [LICENSE](https://github.com/apple/ml-depth-pro/blob/main/LICENSE)
- **[Meta SAM 2](https://github.com/facebookresearch/segment-anything-2)** — Meta AI Research, Apache 2.0
- **[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)** — IDEA Research, Apache 2.0

---

<div align="center">

**⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!**

</div>
