import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ─── Constants / Thresholds ───────────────────────────────────────────────────
IMAGE_RANGE_START = 14       # First image index (inclusive)
IMAGE_RANGE_END = 19         # Last image index (exclusive)

CANNY_THRESHOLD_LOW = 50     # Canny edge lower threshold
CANNY_THRESHOLD_HIGH = 150   # Canny edge upper threshold

HOUGH_THRESHOLD = 100        # Accumulator threshold for HoughLinesP
HOUGH_MIN_LINE_LENGTH = 150  # Minimum line length in pixels
HOUGH_MAX_LINE_GAP = 10      # Maximum gap between line segments

LINE_COLOR = (0, 0, 255)     # BGR colour for detected lines
LINE_THICKNESS = 2

# ─── Paths ────────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).parent
PROJECT_ROOT = _THIS_DIR.parent

EDGES_DIR = _THIS_DIR / "edges"
MASK_DIR = PROJECT_ROOT / "lang-segment-anything" / "output" / "mask_only" / "combined_masks"
HUGHLINE_DIR = _THIS_DIR / "hughline"
LANGSAM_DEPTH_DIR = _THIS_DIR / "langsam_depth"

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def process_image(image_index: int) -> None:
    """단일 이미지에 대해 허프 변환을 수행하고 결과를 저장합니다."""
    edge_path = EDGES_DIR / f"test{image_index}_edges.jpg"
    mask_path = MASK_DIR / f"combined_test{image_index}.png"

    # 이미지 로드
    if not edge_path.exists():
        logger.error(f"Edge image not found: {edge_path}")
        return
    if not mask_path.exists():
        logger.error(f"Mask image not found: {mask_path}")
        return

    image = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE)
    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        logger.error(f"Failed to load edge image: {edge_path}")
        return
    if mask_image is None:
        logger.error(f"Failed to load mask image: {mask_path}")
        return

    # 캐니 엣지 검출
    edges = cv2.Canny(image, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)

    # 허프 변환 (직선 검출)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )

    # 결과 이미지 준비
    image_colored = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    white_lines_image = np.zeros_like(image_colored)
    lang_white_lines_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)

    # 흰색 픽셀을 지나는 선만 그리기
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_mask = np.zeros_like(mask_image, dtype=np.uint8)
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 1)
            if np.any(cv2.bitwise_and(mask_image, line_mask) == 255):
                cv2.line(image_colored, (x1, y1), (x2, y2), LINE_COLOR, LINE_THICKNESS)
                cv2.line(white_lines_image, (x1, y1), (x2, y2), LINE_COLOR, LINE_THICKNESS)
                cv2.line(lang_white_lines_image, (x1, y1), (x2, y2), LINE_COLOR, LINE_THICKNESS)
    else:
        logger.warning(f"image {image_index}: No Hough lines detected.")

    # 결과 저장
    HUGHLINE_DIR.mkdir(parents=True, exist_ok=True)
    LANGSAM_DEPTH_DIR.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(HUGHLINE_DIR / f"test{image_index}_hughline.jpg"), image_colored)
    cv2.imwrite(str(LANGSAM_DEPTH_DIR / f"white_test{image_index}_hughline.jpg"), white_lines_image)
    cv2.imwrite(str(LANGSAM_DEPTH_DIR / f"lang_white_test{image_index}_hughline.jpg"), lang_white_lines_image)
    logger.info(f"image {image_index}: results saved.")

    # 결과 시각화
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
    plt.subplot(1, 4, 2), plt.imshow(image_colored), plt.title(f'All Lines Longer Than {HOUGH_MIN_LINE_LENGTH} px')
    plt.subplot(1, 4, 3), plt.imshow(mask_image), plt.title('Mask Image')
    plt.subplot(1, 4, 4), plt.imshow(white_lines_image), plt.title('White Pixel Lines Only')
    plt.show()


if __name__ == "__main__":
    for i in range(IMAGE_RANGE_START, IMAGE_RANGE_END):
        process_image(i)
