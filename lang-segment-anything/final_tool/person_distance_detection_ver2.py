import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import logging
# 현재 스크립트 위치의 상위 폴더를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from lang_sam import LangSAM
import csv
import pickle
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import json
import datetime

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent
_ASSETS_ROOT = _SCRIPT_DIR.parent.parent / "lang-segment-anything" / "assets" / "human_data"
_OUTPUT_ROOT = _SCRIPT_DIR.parent.parent / "lang-segment-anything" / "output" / "human"

TRAIN_FOLDER = _ASSETS_ROOT / "train"
DEPTH_FOLDER = _ASSETS_ROOT / "human_depth_map" / "train"
CSV_FOLDER = _ASSETS_ROOT / "csv"
MODEL_FOLDER = _OUTPUT_ROOT / "human_models"

OUTPUT_DIRS = [
    _OUTPUT_ROOT / "overlay",
    _OUTPUT_ROOT / "mask_only",
    _OUTPUT_ROOT / "model_plots",
    MODEL_FOLDER,
]

# ─── Text prompts for person segmentation ─────────────────────────────────────
PERSON_PROMPTS = ["person .", "people .", "human ."]


class HeightDepthModel:
    """Encapsulates training, prediction, and serialization of a height-depth linear model."""

    def __init__(self, model_index: int):
        self.model_index = model_index
        self._model: LinearRegression | None = None

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, heights: list[float], depths: list[float]) -> bool:
        """Train a linear regression model from height-depth pairs.

        Args:
            heights: List of mask heights in pixels.
            depths: List of corresponding mean depth values.

        Returns:
            True if training succeeded, False otherwise.
        """
        if len(heights) < 2 or len(depths) < 2 or len(heights) != len(depths):
            logger.error("모델 학습을 위한 데이터가 부족하거나 데이터 길이가 일치하지 않습니다.")
            return False

        X = np.array(heights).reshape(-1, 1)
        y = np.array(depths)

        self._model = LinearRegression()
        self._model.fit(X, y)

        a = self._model.coef_[0]
        b = self._model.intercept_
        logger.info(f"생성된 비례식 모델 (person{self.model_index}): depth = {a:.6f} * height + {b:.6f}")

        for height, actual_depth in zip(heights, depths):
            predicted = self._model.predict([[height]])[0]
            logger.info(f"높이: {height}, 예측 depth: {predicted:.4f} (실제: {actual_depth:.4f})")

        return True

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, height: float) -> float | None:
        """Predict depth for a given mask height.

        Args:
            height: Mask height in pixels.

        Returns:
            Predicted depth, or None if the model is not trained.
        """
        if self._model is None:
            logger.warning("Model is not trained yet.")
            return None
        return float(self._model.predict([[height]])[0])

    # ── Visualisation ─────────────────────────────────────────────────────────

    def plot(self, heights: list[float], depths: list[float], save_dir: Path | None = None) -> None:
        """Save a scatter + regression line plot."""
        if self._model is None:
            return

        plt.figure(figsize=(10, 6))
        plt.scatter(heights, depths, color='blue', label='실제 데이터')

        height_range = np.linspace(min(heights), max(heights), 100).reshape(-1, 1)
        depth_pred = self._model.predict(height_range)
        plt.plot(height_range, depth_pred, color='red', label='모델 예측')

        plt.xlabel('마스크 높이 (픽셀)')
        plt.ylabel('평균 깊이 (depth)')
        plt.title(f'Person{self.model_index} 높이-깊이 관계 모델')
        plt.legend()
        plt.grid(True)

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / f"person{self.model_index}_height_depth_model.png"
            plt.savefig(str(out_path))
            logger.info(f"모델 시각화 저장: {out_path}")
        plt.close()

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, model_folder: Path) -> None:
        """Save the model as both JSON (coefficients) and pickle.

        Args:
            model_folder: Directory where model files will be written.
        """
        if self._model is None:
            logger.warning("No trained model to save.")
            return

        model_folder = Path(model_folder)
        model_folder.mkdir(parents=True, exist_ok=True)

        coef = float(self._model.coef_[0])
        intercept = float(self._model.intercept_)

        model_data = {
            "coefficient": coef,
            "intercept": intercept,
            "formula": f"depth = {coef:.6f} * height + {intercept:.6f}",
            "model_name": f"height_depth_model_person{self.model_index}",
            "date_created": str(datetime.datetime.now()),
            "metadata": {
                "model_type": "LinearRegression",
                "source": f"person{self.model_index} training data",
            },
        }

        json_path = model_folder / f"height_depth_model_person{self.model_index}.json"
        with open(json_path, 'w') as f:
            json.dump(model_data, f, indent=4)
        logger.info(f"높이-깊이 비례식 모델이 JSON 형식으로 저장되었습니다: {json_path}")

        pkl_path = model_folder / f"height_depth_model_person{self.model_index}.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(self._model, f)
        logger.info(f"기존 pickle 형식으로도 저장되었습니다: {pkl_path}")

    @classmethod
    def load(cls, model_folder: Path, model_index: int) -> "HeightDepthModel":
        """Load a previously saved model from a pickle file.

        Args:
            model_folder: Directory containing the pickle file.
            model_index: Person index used when the model was saved.

        Returns:
            A HeightDepthModel instance with the loaded sklearn model.
        """
        pkl_path = Path(model_folder) / f"height_depth_model_person{model_index}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"Model file not found: {pkl_path}")
        instance = cls(model_index)
        with open(pkl_path, 'rb') as f:
            instance._model = pickle.load(f)
        logger.info(f"모델 로드 완료: {pkl_path}")
        return instance


# ─── Standalone helper functions ──────────────────────────────────────────────

def extract_mask_depth_info(mask: np.ndarray, depth_image: np.ndarray):
    """마스크 영역에 포함되는 depth 값을 추출하여 평균과 픽셀 수 계산"""
    if depth_image is None:
        logger.warning("No depth image available for calculation")
        return 0.0, 0

    masked_depth = depth_image[mask]
    pixel_count = len(masked_depth)

    if pixel_count == 0:
        logger.warning("No valid depth pixels in mask")
        return 0.0, 0

    avg_depth = float(np.mean(masked_depth))
    logger.info(f"추출된 마스크 - 픽셀 수: {pixel_count}, 평균 depth: {avg_depth:.4f}")
    return avg_depth, pixel_count


def get_mask_data_from_csv(model_index: int, image_index: int, mask_key: int = 2):
    """CSV 파일에서 특정 mask_key의 height와 avg_depth 값을 가져옴"""
    csv_path = CSV_FOLDER / f"person{model_index}_data.csv"

    if not csv_path.exists():
        logger.warning(f"CSV file not found: {csv_path}")
        return None, None

    try:
        with open(csv_path, 'r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if int(row['image_id']) == image_index and int(row['mask_key']) == mask_key:
                    min_y = int(row['min_y'])
                    max_y = int(row['max_y'])
                    height = max_y - min_y
                    avg_depth = float(row['avg_depth'])
                    logger.info(
                        f"CSV 파일에서 추출 - person{model_index}, image{image_index}, mask_key {mask_key}: "
                        f"높이 {height} (min_y: {min_y}, max_y: {max_y}), avg_depth {avg_depth}"
                    )
                    return height, avg_depth

        logger.warning(f"Image {image_index} with mask key {mask_key} not found in {csv_path}")
        return None, None

    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return None, None


def calculate_mask_height(mask: np.ndarray):
    """마스크의 높이(max_y - min_y) 계산"""
    if not np.any(mask):
        logger.warning("Empty mask")
        return 0, 0, 0

    y_indices = np.where(mask)[0]
    min_y = int(np.min(y_indices))
    max_y = int(np.max(y_indices))
    height = max_y - min_y

    logger.info(f"마스크 높이: {height} (min_y: {min_y}, max_y: {max_y})")
    return height, min_y, max_y


def process_person_images(model_index: int):
    """특정 person 폴더의 모든 이미지를 처리하여 높이-깊이 관계 데이터를 수집

    Args:
        model_index: 처리할 person 인덱스 (1, 2 등)

    Returns:
        Tuple of (combined_heights, combined_depths) lists.
    """
    heights = []
    depths = []
    reference_heights = []
    reference_depths = []

    sam_model = LangSAM()

    person_folder = TRAIN_FOLDER / f"person{model_index}"
    if not person_folder.exists():
        logger.error(f"Person folder not found: {person_folder}")
        return [], []

    image_files = [f for f in person_folder.iterdir() if f.suffix in ('.png', '.jpg')]

    for img_path in image_files:
        image_index = int(''.join(filter(str.isdigit, img_path.name)))
        logger.info(f"\n처리 중: person{model_index}, 이미지 {image_index} ({img_path.name})")

        output_image_path = _OUTPUT_ROOT / "overlay" / f"human_person{model_index}_{img_path.name}"
        mask_only_path = _OUTPUT_ROOT / "mask_only" / f"mask_only_person{model_index}_{img_path.name}"

        # 이미지 로드
        image_pil = Image.open(img_path).convert("RGB")

        # 오버레이 이미지 및 마스크 이미지 준비
        overlay_image = image_pil.copy()
        overlay = Image.new("RGBA", overlay_image.size, (255, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        mask_only_image = Image.new("RGB", image_pil.size, (0, 0, 0))
        mask_only_draw = ImageDraw.Draw(mask_only_image)

        # 각 프롬프트로 마스크 추출
        all_results = []
        for prompt in PERSON_PROMPTS:
            result = sam_model.predict([image_pil], [prompt])
            all_results.extend(result)
            logger.info(f"Processed prompt: {prompt}")

        # 모든 마스크 결합
        combined_person_mask = np.zeros((image_pil.height, image_pil.width), dtype=bool)

        for result in all_results:
            for mask in result["masks"]:
                mask_np = np.array(mask, dtype=bool)
                combined_person_mask = np.logical_or(combined_person_mask, mask_np)

                # 마스크 시각화 (vectorized using paste)
                _mask_img = Image.fromarray(mask_np.astype(np.uint8) * 255, mode="L")
                draw.bitmap((0, 0), Image.fromarray(mask_np.astype(np.uint8) * 255, mode="L"))
                overlay.paste(Image.new("RGBA", overlay.size, (255, 0, 0, 128)), mask=_mask_img)
                mask_only_image.paste(Image.new("RGB", mask_only_image.size, (255, 255, 255)), mask=_mask_img)

        # depth 이미지 로드
        depth_path_npz = DEPTH_FOLDER / f"person{model_index}" / f"train{image_index}.npz"
        depth_image = None

        if depth_path_npz.exists():
            try:
                depth_data = np.load(depth_path_npz)
                keys = list(depth_data.keys())
                if keys:
                    depth_image = depth_data[keys[0]]
                    logger.info(f"로드된 depth 이미지: {depth_path_npz} (키: {keys[0]})")
                    logger.info(f"Depth 범위: min={depth_image.min():.4f}, max={depth_image.max():.4f}, mean={depth_image.mean():.4f}")
                else:
                    logger.warning("NPZ 파일에 배열이 없습니다")
            except Exception as e:
                logger.error(f"Error loading depth npz: {e}")

                depth_path_npy = DEPTH_FOLDER / f"person{model_index}" / f"train{image_index}.npy"
                if depth_path_npy.exists():
                    try:
                        depth_image = np.load(depth_path_npy)
                        logger.info(f"NPY 파일에서 로드 성공: {depth_path_npy}")
                    except Exception as e2:
                        logger.error(f"Error loading depth npy: {e2}")
        else:
            logger.warning(f"Depth file not found: {depth_path_npz}")

        # 결합된 마스크에서 depth 정보와 높이 추출
        if depth_image is not None and np.any(combined_person_mask):
            current_depth, _ = extract_mask_depth_info(combined_person_mask, depth_image)
            current_height, min_y, max_y = calculate_mask_height(combined_person_mask)

            if current_height > 0 and current_depth > 0:
                heights.append(current_height)
                depths.append(current_depth)

            reference_height, reference_depth = get_mask_data_from_csv(model_index, image_index)
            if reference_height is not None and reference_depth is not None:
                reference_heights.append(reference_height)
                reference_depths.append(reference_depth)

            # 높이 표시선 그리기
            height_overlay = Image.new("RGBA", image_pil.size, (0, 0, 0, 0))
            height_draw = ImageDraw.Draw(height_overlay)
            height_draw.line([(0, min_y), (image_pil.width, min_y)], fill=(0, 255, 0, 255), width=2)
            height_draw.line([(0, max_y), (image_pil.width, max_y)], fill=(0, 255, 0, 255), width=2)

            try:
                from PIL import ImageFont
                font = ImageFont.load_default()
                height_draw.text((10, min_y - 30), f"Height: {current_height}", fill=(255, 255, 255, 255), font=font)
                height_draw.text((10, min_y - 10), f"Depth: {current_depth:.4f}", fill=(255, 255, 255, 255), font=font)
            except Exception as e:
                logger.warning(f"텍스트 표시 오류: {e}")

            overlay = Image.alpha_composite(overlay, height_overlay)

        # 결과 이미지 저장
        final_overlay_image = Image.alpha_composite(overlay_image.convert("RGBA"), overlay)
        final_overlay_image = final_overlay_image.convert("RGB")
        final_overlay_image.save(output_image_path)
        mask_only_image.save(mask_only_path)

    # 실제 데이터와 참조 데이터 통합
    combined_heights = heights + reference_heights
    combined_depths = depths + reference_depths

    logger.info(f"\nPerson{model_index} 데이터 수집 완료:")
    logger.info(f"실측 데이터: {len(heights)} 개")
    logger.info(f"참조 데이터: {len(reference_heights)} 개")
    logger.info(f"통합 데이터: {len(combined_heights)} 개")

    return combined_heights, combined_depths


def main():
    # 출력 디렉토리 생성
    for directory in OUTPUT_DIRS:
        Path(directory).mkdir(parents=True, exist_ok=True)

    # 각 person 폴더별로 처리
    for model_index in [1, 2]:
        logger.info(f"\n===== Person {model_index} 처리 시작 =====")

        heights, depths = process_person_images(model_index)

        if len(heights) >= 2:
            hdm = HeightDepthModel(model_index)
            success = hdm.fit(heights, depths)

            if success:
                hdm.plot(heights, depths, save_dir=_OUTPUT_ROOT / "model_plots")
                hdm.save(MODEL_FOLDER)
        else:
            logger.warning(f"Person{model_index}의 데이터가 부족하여 모델을 생성할 수 없습니다.")

    logger.info("\n모든 처리가 완료되었습니다!")


if __name__ == "__main__":
    main()
