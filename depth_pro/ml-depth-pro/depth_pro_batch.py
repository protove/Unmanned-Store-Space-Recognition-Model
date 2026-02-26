import logging
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import sys

# ─── Constants ────────────────────────────────────────────────────────────────
# Root of the ml-depth-pro package (this file's parent directory)
_THIS_DIR = Path(__file__).parent
_SRC_DIR = _THIS_DIR / "src"
DEFAULT_CHECKPOINT = _THIS_DIR / "checkpoints" / "depth_pro.pt"

# Default I/O paths (override via CLI arguments)
DEFAULT_INPUT_DIR = _THIS_DIR.parent.parent / "lang-segment-anything" / "output" / "total_s1_fps3"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "depth_maps"

FILE_TYPES = ['.jpg', '.jpeg', '.png', '.bmp']

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# depth_pro 모듈을 정확히 찾을 수 있도록 경로 추가
sys.path.append(str(_SRC_DIR))

# 모듈 및 클래스 임포트
from depth_pro.depth_pro import create_model_and_transforms, DepthProConfig


def process_folder_with_depth_pro(
    input_folder: Path,
    output_folder: Path,
    device=None,
    file_types=None,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
):
    """
    입력 폴더의 모든 이미지에 대해 Depth Pro 모델을 실행하고 결과를 출력 폴더에 저장합니다.

    Args:
        input_folder (Path): 처리할 이미지가 있는 입력 폴더 경로
        output_folder (Path): 깊이 맵을 저장할 출력 폴더 경로
        device (torch.device, optional): 사용할 장치 (None이면 자동 선택)
        file_types (list, optional): 처리할 이미지 파일 확장자 리스트
        checkpoint_path (Path): DepthPro 체크포인트 파일 경로
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    checkpoint_path = Path(checkpoint_path)

    # 기본 파일 타입 설정
    if file_types is None:
        file_types = FILE_TYPES

    # GPU 사용 가능 여부 확인 및 장치 설정
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"사용 장치: {device}")

    # 출력 폴더 생성
    output_folder.mkdir(parents=True, exist_ok=True)

    # 체크포인트 경로 확인
    if not checkpoint_path.exists():
        logger.error(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        return

    # DepthPro 설정 및 모델 로드
    config = DepthProConfig(
        patch_encoder_preset="dinov2l16_384",
        image_encoder_preset="dinov2l16_384",
        checkpoint_uri=str(checkpoint_path),
        decoder_features=256,
        use_fov_head=True,
        fov_encoder_preset="dinov2l16_384",
    )

    try:
        logger.info("모델 로드 중...")
        model, transform = create_model_and_transforms(config=config, device=device)
        model.eval()
        logger.info("모델 로드 완료!")
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        return

    # 입력 폴더에서 이미지 파일 찾기
    image_files = []
    for file_type in file_types:
        image_files.extend(input_folder.glob(f'*{file_type}'))
        image_files.extend(input_folder.glob(f'*{file_type.upper()}'))

    if not image_files:
        logger.warning(f"입력 폴더 '{input_folder}'에 이미지 파일이 없습니다.")
        return

    logger.info(f"총 {len(image_files)}개의 이미지 파일을 처리합니다.")

    # 각 이미지 파일 처리
    for image_path in tqdm(image_files, desc="깊이 맵 생성 중"):
        image_path = Path(image_path)
        try:
            name = image_path.stem

            # 이미지 로드 및 변환
            img = Image.open(image_path).convert('RGB')
            input_tensor = transform(img).unsqueeze(0).to(device)

            # 깊이 추론
            with torch.no_grad():
                result = model.infer(input_tensor)
                depth_map = result["depth"].cpu().numpy()
                focal_length = result["focallength_px"].item()

            # 깊이 맵 시각화 및 저장
            depth_output_path = output_folder / f"{name}_depth.png"
            plt.figure(figsize=(8, 8))
            plt.imshow(depth_map, cmap='viridis')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(depth_output_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            # 원시 깊이 데이터 저장 (NumPy 배열)
            np.save(output_folder / f"{name}_depth.npy", depth_map)

            # 메타데이터 저장
            meta_output_path = output_folder / f"{name}_meta.txt"
            with open(meta_output_path, 'w') as f:
                f.write(f"Focal Length (px): {focal_length}\n")
                f.write(f"Min Depth (m): {np.min(depth_map):.4f}\n")
                f.write(f"Max Depth (m): {np.max(depth_map):.4f}\n")
                f.write(f"Mean Depth (m): {np.mean(depth_map):.4f}\n")

        except FileNotFoundError:
            logger.error(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        except Exception as e:
            logger.error(f"이미지 '{image_path.name}' 처리 중 오류 발생: {e}")

    logger.info(f"처리 완료! 결과가 '{output_folder}' 폴더에 저장되었습니다.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Depth Pro - 폴더 내 이미지의 깊이 맵 생성')
    parser.add_argument('--input', type=str, help='입력 이미지 폴더 경로')
    parser.add_argument('--output', type=str, help='출력 폴더 경로')
    parser.add_argument('--checkpoint', type=str, help='체크포인트 파일 경로')
    parser.add_argument('--gpu', action='store_true', help='GPU 사용 (사용 가능한 경우)')
    parser.add_argument('--cpu', action='store_true', help='CPU 사용 강제')

    args = parser.parse_args()

    # 인자가 없는 경우 기본 경로 사용
    _input = Path(args.input) if args.input else DEFAULT_INPUT_DIR
    _output = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR
    _checkpoint = Path(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINT

    # 장치 설정
    _device = None
    if args.cpu:
        _device = torch.device('cpu')
    elif args.gpu and torch.cuda.is_available():
        _device = torch.device('cuda')

    # 폴더 처리 실행
    process_folder_with_depth_pro(_input, _output, _device, checkpoint_path=_checkpoint)