import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
# 현재 스크립트 위치의 상위 폴더를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw
import numpy as np
from lang_sam import LangSAM
import csv
import pickle
from sklearn.linear_model import LinearRegression

# 1. 폴더 경로 설정
depth_folder = "./CapstoneD/Capstone_Final/lang-segment-anything/assets/human_data/human_depth_map/train"  # depth .npz 파일 폴더
csv_folder = "./CapstoneD/Capstone_Final/lang-segment-anything/assets/human_data/csv"  # csv 파일 폴더
model_folder = "./CapstoneD/Capstone_Final/lang-segment-anything/output/human_models"  # 모델 저장 폴더

# 모델 저장 폴더 생성
os.makedirs(model_folder, exist_ok=True)

# 출력 폴더 생성
output_dirs = [
    "./CapstoneD/Capstone_Final/lang-segment-anything/output/human/overlay",
    "./CapstoneD/Capstone_Final/lang-segment-anything/output/human/mask_only",
]

for directory in output_dirs:
    os.makedirs(directory, exist_ok=True)

# 2. 마스크 영역의 depth 정보 추출 함수
def extract_mask_depth_info(mask, depth_image):
    """
    마스크 영역에 포함되는 depth 값을 추출하여 평균과 픽셀 수 계산
    """
    if depth_image is None:
        print("Warning: No depth image available for calculation")
        return 0.0, 0
    
    # 마스크가 True인 부분의 depth 값만 추출
    masked_depth = depth_image[mask]
    
    # 픽셀 수
    pixel_count = len(masked_depth)
    
    # 평균 계산 (값이 없을 경우 0 반환)
    if pixel_count == 0:
        print("Warning: No valid depth pixels in mask")
        return 0.0, 0
    
    avg_depth = np.mean(masked_depth)
    print(f"추출된 마스크 - 픽셀 수: {pixel_count}, 평균 depth: {avg_depth:.4f}")
    
    return avg_depth, pixel_count

# 3. CSV 파일에서 특정 마스크 키의 데이터를 가져오는 함수
def get_mask_data_from_csv(image_index, mask_key=2):
    """
    CSV 파일에서 특정 mask_key의 height(max_y - min_y)와 avg_depth 값을 가져옴
    """
    csv_path = os.path.join(csv_folder, f"space1.csv")
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return None, None
    
    try:
        # CSV 파일 읽기
        with open(csv_path, 'r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if int(row['mask_key']) == mask_key:
                    min_y = int(row['min_y'])
                    max_y = int(row['max_y'])
                    height = max_y - min_y
                    avg_depth = float(row['avg_depth'])
                    print(f"CSV 파일에서 추출 - mask_key {mask_key}: 높이 {height} (min_y: {min_y}, max_y: {max_y}), avg_depth {avg_depth}")
                    return height, avg_depth
        
        print(f"Mask key {mask_key} not found in {csv_path}")
        return None, None
    
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return None, None

# 마스크의 높이를 계산하는 함수
def calculate_mask_height(mask):
    """
    마스크의 높이(max_y - min_y) 계산
    
    Args:
        mask: 부울 배열 형태의 마스크
        
    Returns:
        마스크의 높이와 min_y, max_y 좌표
    """
    if not np.any(mask):
        print("Warning: Empty mask")
        return 0, 0, 0
    
    # 마스크가 True인 픽셀의 y 좌표 찾기
    y_indices = np.where(mask)[0]
    
    min_y = np.min(y_indices)
    max_y = np.max(y_indices)
    height = max_y - min_y
    
    print(f"마스크 높이: {height} (min_y: {min_y}, max_y: {max_y})")
    
    return height, min_y, max_y

# 4. 비례식 모델을 생성하는 함수 (높이 기반)
def create_proportion_model(current_height, current_depth, reference_height, reference_depth):
    """
    현재 마스크와 참조 마스크 간의 높이-깊이 비례 관계를 학습하여 모델 생성
    
    Args:
        current_height: 현재 마스크의 높이
        current_depth: 현재 마스크의 평균 depth
        reference_height: 참조 마스크의 높이
        reference_depth: 참조 마스크의 평균 depth
        
    Returns:
        비례식을 표현하는 LinearRegression 모델
    """
    # 간단한 선형 모델: y = ax + b 형태로 표현
    # 여기서 x는 높이(height), y는 depth 값
    
    # 학습 데이터 준비
    X = np.array([[current_height], [reference_height]])
    y = np.array([current_depth, reference_depth])
    
    # 선형 회귀 모델 학습
    model = LinearRegression()
    model.fit(X, y)
    
    # 모델 계수 출력
    a = model.coef_[0]
    b = model.intercept_
    print(f"생성된 비례식 모델: depth = {a:.6f} * height + {b:.6f}")
    
    # 예측 결과 확인
    current_pred = model.predict([[current_height]])[0]
    reference_pred = model.predict([[reference_height]])[0]
    print(f"현재 마스크 예측: {current_pred:.4f} (실제: {current_depth:.4f})")
    print(f"참조 마스크 예측: {reference_pred:.4f} (실제: {reference_depth:.4f})")
    
    return model

# 5. 비례식 모델을 저장하는 함수 (이름 변경)
def save_proportion_model(model, image_index):
    """
    비례식 모델을 파일로 저장
    """
    model_path = os.path.join(model_folder, f"height_depth_model_test{image_index}.pkl")
    
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    
    print(f"높이-깊이 비례식 모델이 저장되었습니다: {model_path}")

# Initialize the model
model = LangSAM()
for i in range(1, 3):
    image_name = f"train{i}.png"
    input_image_path = f"./CapstoneD/Capstone_Final/lang-segment-anything/assets/human_data/train/{image_name}"
    output_image_path = f"./CapstoneD/Capstone_Final/lang-segment-anything/output/human/overlay/human_{image_name}"
    mask_only_path = f"./CapstoneD/Capstone_Final/lang-segment-anything/output/human/mask_only/mask_only_{image_name}"

    # Load the input image
    image_pil = Image.open(input_image_path).convert("RGB")

    # 여러 프롬프트 정의
    text_prompts = ["person .", "people .", "human ."]

    # Create an overlay image for all masks
    overlay_image = image_pil.copy()
    overlay = Image.new("RGBA", overlay_image.size, (255, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # 검정색 배경에 마스크만 표시할 이미지 생성
    mask_only_image = Image.new("RGB", image_pil.size, (0, 0, 0))
    mask_only_draw = ImageDraw.Draw(mask_only_image)

    # 각 프롬프트를 개별적으로 처리하여 결과를 결합
    all_results = []
    for prompt in text_prompts:
        result = model.predict([image_pil], [prompt])
        all_results.extend(result)
        
        print(f"Processed prompt: {prompt}")
        print(f"Result for this prompt: {result}")

    # 모든 마스크 결합 (전체 사람 마스크)
    combined_person_mask = np.zeros((image_pil.height, image_pil.width), dtype=bool)
    
    # 모든 결과에 대한 마스크를 순회하며 결합
    for result in all_results:
        for mask in result["masks"]:
            mask_np = np.array(mask, dtype=bool)
            combined_person_mask = np.logical_or(combined_person_mask, mask_np)
            
            for y in range(mask_np.shape[0]):
                for x in range(mask_np.shape[1]):
                    if mask_np[y, x]:
                        draw.point((x, y), fill=(255, 0, 0, 128))
                        mask_only_draw.point((x, y), fill=(255, 255, 255))

    # depth 이미지 로드
    depth_path_npz = os.path.join(depth_folder, f"train{i}.npz")
    depth_image = None

    if os.path.exists(depth_path_npz):
        try:
            depth_data = np.load(depth_path_npz)
            # 모든 키 목록 확인 및 출력
            keys = list(depth_data.keys())
            print(f"NPZ 파일 내 키 목록: {keys}")
            
            if len(keys) > 0:
                # 첫 번째 키 사용 (일반적으로 하나의 배열만 저장됨)
                key = keys[0]
                depth_image = depth_data[key]
                print(f"로드된 depth 이미지: {depth_path_npz} (키: {key})")
                print(f"Depth 범위: min={depth_image.min():.4f}, max={depth_image.max():.4f}, mean={depth_image.mean():.4f}")
            else:
                print("NPZ 파일에 배열이 없습니다")
        except Exception as e:
            print(f"Error loading depth npz: {str(e)}")
            
            # NPZ 로드 실패 시, NPY 파일도 시도
            depth_path_npy = os.path.join(depth_folder, f"train{i}.npy")
            if os.path.exists(depth_path_npy):
                try:
                    depth_image = np.load(depth_path_npy)
                    print(f"NPY 파일에서 로드 성공: {depth_path_npy}")
                    print(f"Depth 범위: min={depth_image.min():.4f}, max={depth_image.max():.4f}, mean={depth_image.mean():.4f}")
                except Exception as e2:
                    print(f"Error loading depth npy: {str(e2)}")
    else:
        print(f"Warning: Depth npz file not found: {depth_path_npz}")
    
    # 결합된 마스크에서 depth 정보와 높이 추출
    if depth_image is not None and np.any(combined_person_mask):
        # 마스크의 depth 추출 (평균 및 픽셀 수)
        current_depth, current_pixels = extract_mask_depth_info(combined_person_mask, depth_image)
        
        # 마스크의 높이 계산
        current_height, min_y, max_y = calculate_mask_height(combined_person_mask)
        
        # CSV 파일에서 참조 마스크 데이터 가져오기 (이제 높이와 깊이 정보)
        reference_height, reference_depth = get_mask_data_from_csv(i)
        
        if reference_height is not None and reference_depth is not None:
            # 높이-깊이 비례식 모델 생성
            proportion_model = create_proportion_model(
                current_height, current_depth,
                reference_height, reference_depth
            )
            
            # 모델 저장
            save_proportion_model(proportion_model, i)
            
            # 모델 사용 예시: 현재 높이로 depth 예측
            predicted_depth = proportion_model.predict([[current_height]])[0]
            print(f"현재 마스크(높이: {current_height})에 대한 예측 depth: {predicted_depth:.4f}")
            print(f"현재 마스크(높이: {current_height})의 실제 depth: {current_depth:.4f}")
    
    # Composite the overlay onto the original image
    final_overlay_image = Image.alpha_composite(overlay_image.convert("RGBA"), overlay)
    final_overlay_image = final_overlay_image.convert("RGB")  # JPEG 호환성을 위해 RGB로 변환
    final_overlay_image.save(output_image_path)
    
    # 마스크만 있는 이미지 저장
    mask_only_image.save(mask_only_path)

print("모든 처리가 완료되었습니다!")
