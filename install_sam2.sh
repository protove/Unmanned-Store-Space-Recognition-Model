#!/bin/bash

# SAM2 설치 스크립트
# SAM2는 별도의 git 리포지토리에서 설치해야 합니다.

echo "Installing SAM2 (Segment Anything 2)..."

# SAM2 클론 및 설치
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
cd ..

# 설치된 SAM2 디렉토리를 python path에 추가하기 위한 심볼릭 링크 생성 (선택사항)
# ln -sf segment-anything-2/sam2 ./lang-segment-anything/

echo "SAM2 installation completed!"
echo "Note: Make sure to download the SAM2 model checkpoints if needed."
