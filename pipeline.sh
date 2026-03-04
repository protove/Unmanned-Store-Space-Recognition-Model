#!/usr/bin/env bash
# ============================================================================
# Unmanned Store Space Recognition – Full Pipeline Runner
#
# 사용법:
#   ./pipeline.sh [입력폴더] [출력폴더] [체크포인트폴더] [이미지수]
#
# 예시:
#   ./pipeline.sh                          # 기본값 사용
#   ./pipeline.sh ./data/input ./data/output ./data/checkpoints 3
#
# 전제조건:
#   1. Docker + Docker Compose v2 설치
#   2. NVIDIA Container Toolkit 설치 (GPU 사용 시)
#   3. data/checkpoints/depth_pro.pt 파일 위치
#      (없을 경우 이 스크립트가 자동으로 다운로드 안내)
#
# 입력 이미지 명명 규칙:
#   data/input/test1.png, test2.png, ... testN.png
# ============================================================================
set -euo pipefail

# ── 인자 파싱 (기본값 포함) ──────────────────────────────────────────────────
INPUT_DIR="${1:-$(pwd)/data/input}"
OUTPUT_DIR="${2:-$(pwd)/data/output}"
CHECKPOINT_DIR="${3:-$(pwd)/data/checkpoints}"
export SPACE_IMAGE_COUNT="${4:-3}"

CHECKPOINT_FILE="$CHECKPOINT_DIR/depth_pro.pt"
DEPTH_PRO_URL="https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"

# ── 색상 출력 헬퍼 ───────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Unmanned Store Space Recognition Pipeline"
echo "═══════════════════════════════════════════════════"
info  "입력 폴더    : $INPUT_DIR"
info  "출력 폴더    : $OUTPUT_DIR"
info  "체크포인트   : $CHECKPOINT_FILE"
info  "이미지 수    : $SPACE_IMAGE_COUNT (test1.png … test${SPACE_IMAGE_COUNT}.png)"
echo ""

# ── 디렉터리 생성 ────────────────────────────────────────────────────────────
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR" "$CHECKPOINT_DIR" \
         "$(pwd)/data/hf_cache" "$(pwd)/data/torch_cache"

# ── DepthPro 체크포인트 확인 ─────────────────────────────────────────────────
if [ ! -f "$CHECKPOINT_FILE" ]; then
    warn "depth_pro.pt 파일이 없습니다."
    echo ""
    echo "  자동 다운로드 하시겠습니까? (약 2.5 GB)"
    read -r -p "  [y/N] > " REPLY
    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
        info "다운로드 중: $DEPTH_PRO_URL"
        wget --progress=bar:force:noscroll -O "$CHECKPOINT_FILE" "$DEPTH_PRO_URL"
        success "다운로드 완료: $CHECKPOINT_FILE"
    else
        error "체크포인트 없이 실행할 수 없습니다."
        echo "  수동 다운로드:"
        echo "    wget $DEPTH_PRO_URL -P $CHECKPOINT_DIR"
        exit 1
    fi
else
    success "체크포인트 확인: $CHECKPOINT_FILE"
fi

# ── 입력 이미지 확인 ─────────────────────────────────────────────────────────
IMG_COUNT=$(find "$INPUT_DIR" -maxdepth 1 \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
if [ "$IMG_COUNT" -eq 0 ]; then
    error "입력 폴더에 이미지가 없습니다: $INPUT_DIR"
    echo "  이미지를 다음 이름으로 저장하세요: test1.png, test2.png, ..."
    exit 1
fi
success "입력 이미지 $IMG_COUNT 개 발견"

# ── Docker Compose 실행 ──────────────────────────────────────────────────────
echo ""
info "Docker 이미지 빌드 및 파이프라인 시작..."
echo ""

# data/ 폴더를 현재 디렉터리 기준으로 마운트하기 위해 심볼릭 링크 생성 (필요 시)
# docker-compose.yml은 ./data/input 등 상대경로를 사용함
# 사용자 지정 경로가 ./data/* 와 다른 경우 경고
COMPOSE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$INPUT_DIR" != "$COMPOSE_DIR/data/input" ]; then
    warn "주의: 사용자 지정 경로는 docker-compose.yml의 볼륨 마운트와 다를 수 있습니다."
    warn "      docker-compose.yml을 직접 편집하거나 data/ 하위 구조를 사용하세요."
fi

cd "$COMPOSE_DIR"

docker compose up \
    --build \
    --abort-on-container-exit \
    --exit-code-from langsam \
    2>&1

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    success "파이프라인 완료!"
    success "결과물 위치: $OUTPUT_DIR"
    echo ""
    echo "  출력 구조:"
    echo "    $OUTPUT_DIR/overlay/           – 오버레이 이미지"
    echo "    $OUTPUT_DIR/mask_only/         – 마스크 이미지"
    echo "    $OUTPUT_DIR/json/              – 마스크 메타데이터"
else
    error "파이프라인 실패 (exit code: $EXIT_CODE)"
    echo "  로그 확인: docker compose logs"
    exit $EXIT_CODE
fi
