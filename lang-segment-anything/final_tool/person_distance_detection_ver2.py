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
import matplotlib.pyplot as plt
import json
import datetime

# 1. 폴더 경로 설정
train_folder = "./CapstoneD/Capstone_Final/lang-segment-anything/assets/human_data/train"  # 트레인 이미지 폴더
depth_folder = "./CapstoneD/Capstone_Final/lang-segment-anything/assets/human_data/human_depth_map/train"  # depth .npz 파일 폴더
csv_folder = "./CapstoneD/Capstone_Final/lang-segment-anything/assets/human_data/csv"  # csv 파일 폴더
model_folder = "./CapstoneD/Capstone_Final/lang-segment-anything/output/human_models"  # 모델 저장 폴더

# 모델 저장 폴더 생성
os.makedirs(model_folder, exist_ok=True)

# 출력 폴더 생성
output_dirs = [
    "./CapstoneD/Capstone_Final/lang-segment-anything/output/human/overlay",
    "./CapstoneD/Capstone_Final/lang-segment-anything/output/human/mask_only",
    "./CapstoneD/Capstone_Final/lang-segment-anything/output/human/model_plots",
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
def get_mask_data_from_csv(model_index, image_index, mask_key=2):
    """
    CSV 파일에서 특정 mask_key의 height(max_y - min_y)와 avg_depth 값을 가져옴
    """
    csv_path = os.path.join(csv_folder, f"person{model_index}_data.csv")  # 파일명 형식 수정
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return None, None
    
    try:
        # CSV 파일 읽기
        with open(csv_path, 'r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # image_id와 mask_key가 모두 일치하는 행을 찾음
                if int(row['image_id']) == image_index and int(row['mask_key']) == mask_key:
                    min_y = int(row['min_y'])
                    max_y = int(row['max_y'])
                    height = max_y - min_y
                    avg_depth = float(row['avg_depth'])
                    print(f"CSV 파일에서 추출 - person{model_index}, image{image_index}, mask_key {mask_key}: "
                          f"높이 {height} (min_y: {min_y}, max_y: {max_y}), avg_depth {avg_depth}")
                    return height, avg_depth
        
        print(f"Image {image_index} with mask key {mask_key} not found in {csv_path}")
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
def create_height_depth_model(heights, depths, model_index):
    """
    여러 데이터 포인트로부터 높이-깊이 관계를 학습하여 모델 생성
    
    Args:
        heights: 마스크 높이 리스트
        depths: 마스크 깊이 리스트
        model_index: 모델 인덱스 (person1, person2 등)
        
    Returns:
        비례식을 표현하는 LinearRegression 모델
    """
    if len(heights) < 2 or len(depths) < 2 or len(heights) != len(depths):
        print("Error: 모델 학습을 위한 데이터가 부족하거나 데이터 길이가 일치하지 않습니다.")
        return None
    
    # 학습 데이터 준비 - 2D 형태로 변환
    X = np.array(heights).reshape(-1, 1)
    y = np.array(depths)
    
    # 선형 회귀 모델 학습
    model = LinearRegression()
    model.fit(X, y)
    
    # 모델 계수 출력
    a = model.coef_[0]
    b = model.intercept_
    print(f"생성된 비례식 모델 (person{model_index}): depth = {a:.6f} * height + {b:.6f}")
    
    # 예측 결과 확인
    for i, (height, actual_depth) in enumerate(zip(heights, depths)):
        predicted_depth = model.predict([[height]])[0]
        print(f"데이터 {i+1} - 높이: {height}, 예측 depth: {predicted_depth:.4f} (실제: {actual_depth:.4f})")
    
    # 모델 시각화
    plt.figure(figsize=(10, 6))
    plt.scatter(heights, depths, color='blue', label='실제 데이터')
    
    # 모델 선 그리기
    height_range = np.linspace(min(heights), max(heights), 100).reshape(-1, 1)
    depth_pred = model.predict(height_range)
    plt.plot(height_range, depth_pred, color='red', label='모델 예측')
    
    plt.xlabel('마스크 높이 (픽셀)')
    plt.ylabel('평균 깊이 (depth)')
    plt.title(f'Person{model_index} 높이-깊이 관계 모델')
    plt.legend()
    plt.grid(True)
    
    # 모델 시각화 저장
    plt.savefig(f"./CapstoneD/Capstone_Final/lang-segment-anything/output/human/model_plots/person{model_index}_height_depth_model.png")
    plt.close()
    
    return model

# 5. 비례식 모델을 저장하는 함수 - JSON 형식으로 변경
def save_height_depth_model(model, model_index):
    """
    높이-깊이 비례식 모델을 JSON 파일로 저장
    """
    
    # 모델 계수와 절편을 딕셔너리로 저장
    model_data = {
        "coefficient": float(model.coef_[0]),  # 계수
        "intercept": float(model.intercept_),  # 절편
        "formula": f"depth = {float(model.coef_[0]):.6f} * height + {float(model.intercept_):.6f}",
        "model_name": f"height_depth_model_person{model_index}",
        "date_created": str(datetime.datetime.now()),
        "metadata": {
            "model_type": "LinearRegression",
            "source": f"person{model_index} training data"
        }
    }
    
    # JSON 파일로 저장
    model_path = os.path.join(model_folder, f"height_depth_model_person{model_index}.json")
    with open(model_path, 'w') as file:
        json.dump(model_data, file, indent=4)
    
    print(f"높이-깊이 비례식 모델이 JSON 형식으로 저장되었습니다: {model_path}")
    
    # 기존 pickle 형식도 유지하고 싶다면 추가
    pkl_path = os.path.join(model_folder, f"height_depth_model_person{model_index}.pkl")
    with open(pkl_path, 'wb') as file:
        pickle.dump(model, file)
    
    print(f"기존 pickle 형식으로도 저장되었습니다: {pkl_path}")

def process_person_images(model_index):
    """
    특정 person 폴더의 모든 이미지를 처리하여 높이-깊이 관계 데이터를 수집
    
    Args:
        model_index: 처리할 person 인덱스 (1, 2 등)
        
    Returns:
        높이 리스트, 깊이 리스트, 참조 높이 리스트, 참조 깊이 리스트
    """
    # 데이터를 저장할 리스트
    heights = []
    depths = []
    reference_heights = []
    reference_depths = []
    
    # LangSAM 모델 초기화
    sam_model = LangSAM()
    
    # 해당 person 폴더의 이미지 처리
    person_folder = os.path.join(train_folder, f"person{model_index}")
    
    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(person_folder) if f.endswith('.png') or f.endswith('.jpg')]
    
    for img_file in image_files:
        image_index = int(''.join(filter(str.isdigit, img_file)))  # 파일 이름에서 숫자 추출
        print(f"\n처리 중: person{model_index}, 이미지 {image_index} ({img_file})")
        
        input_image_path = os.path.join(person_folder, img_file)
        output_image_path = f"./CapstoneD/Capstone_Final/lang-segment-anything/output/human/overlay/human_person{model_index}_{img_file}"
        mask_only_path = f"./CapstoneD/Capstone_Final/lang-segment-anything/output/human/mask_only/mask_only_person{model_index}_{img_file}"

        # 이미지 로드
        image_pil = Image.open(input_image_path).convert("RGB")

        # 여러 프롬프트 정의
        text_prompts = ["person .", "people .", "human ."]

        # 오버레이 이미지 및 마스크 이미지 준비
        overlay_image = image_pil.copy()
        overlay = Image.new("RGBA", overlay_image.size, (255, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        mask_only_image = Image.new("RGB", image_pil.size, (0, 0, 0))
        mask_only_draw = ImageDraw.Draw(mask_only_image)

        # 각 프롬프트로 마스크 추출
        all_results = []
        for prompt in text_prompts:
            result = sam_model.predict([image_pil], [prompt])
            all_results.extend(result)
            print(f"Processed prompt: {prompt}")

        # 모든 마스크 결합
        combined_person_mask = np.zeros((image_pil.height, image_pil.width), dtype=bool)
        
        for result in all_results:
            for mask in result["masks"]:
                mask_np = np.array(mask, dtype=bool)
                combined_person_mask = np.logical_or(combined_person_mask, mask_np)
                
                # 마스크 시각화
                for y in range(mask_np.shape[0]):
                    for x in range(mask_np.shape[1]):
                        if mask_np[y, x]:
                            draw.point((x, y), fill=(255, 0, 0, 128))
                            mask_only_draw.point((x, y), fill=(255, 255, 255))

        # depth 이미지 로드
        depth_path_npz = os.path.join(depth_folder, f"person{model_index}", f"train{image_index}.npz")
        depth_image = None

        if os.path.exists(depth_path_npz):
            try:
                depth_data = np.load(depth_path_npz)
                keys = list(depth_data.keys())
                
                if len(keys) > 0:
                    key = keys[0]
                    depth_image = depth_data[key]
                    print(f"로드된 depth 이미지: {depth_path_npz} (키: {key})")
                    print(f"Depth 범위: min={depth_image.min():.4f}, max={depth_image.max():.4f}, mean={depth_image.mean():.4f}")
                else:
                    print("NPZ 파일에 배열이 없습니다")
            except Exception as e:
                print(f"Error loading depth npz: {str(e)}")
                
                # NPY 파일도 시도
                depth_path_npy = os.path.join(depth_folder, f"person{model_index}", f"train{image_index}.npy")
                if os.path.exists(depth_path_npy):
                    try:
                        depth_image = np.load(depth_path_npy)
                        print(f"NPY 파일에서 로드 성공: {depth_path_npy}")
                    except Exception as e2:
                        print(f"Error loading depth npy: {str(e2)}")
        else:
            print(f"Warning: Depth file not found: {depth_path_npz}")
        
        # 결합된 마스크에서 depth 정보와 높이 추출
        if depth_image is not None and np.any(combined_person_mask):
            # 마스크의 depth 값과 높이 계산
            current_depth, _ = extract_mask_depth_info(combined_person_mask, depth_image)
            current_height, min_y, max_y = calculate_mask_height(combined_person_mask)
            
            # 높이와 깊이 값 저장
            if current_height > 0 and current_depth > 0:
                heights.append(current_height)
                depths.append(current_depth)
            
            # CSV 파일에서 참조 데이터 가져오기
            reference_height, reference_depth = get_mask_data_from_csv(model_index, image_index)
            
            if reference_height is not None and reference_depth is not None:
                reference_heights.append(reference_height)
                reference_depths.append(reference_depth)
            
            # 높이 표시선 그리기
            height_overlay = Image.new("RGBA", image_pil.size, (0, 0, 0, 0))
            height_draw = ImageDraw.Draw(height_overlay)
            height_draw.line([(0, min_y), (image_pil.width, min_y)], fill=(0, 255, 0, 255), width=2)
            height_draw.line([(0, max_y), (image_pil.width, max_y)], fill=(0, 255, 0, 255), width=2)
            
            # 텍스트 표시
            font_size = 20
            try:
                from PIL import ImageFont
                font = ImageFont.load_default()
                height_draw.text((10, min_y - 30), f"Height: {current_height}", fill=(255, 255, 255, 255), font=font)
                height_draw.text((10, min_y - 10), f"Depth: {current_depth:.4f}", fill=(255, 255, 255, 255), font=font)
            except Exception as e:
                print(f"텍스트 표시 오류: {str(e)}")
            
            overlay = Image.alpha_composite(overlay, height_overlay)
        
        # 결과 이미지 저장
        final_overlay_image = Image.alpha_composite(overlay_image.convert("RGBA"), overlay)
        final_overlay_image = final_overlay_image.convert("RGB")
        final_overlay_image.save(output_image_path)
        
        mask_only_image.save(mask_only_path)
    
    # 모든 데이터를 통합 (실제 데이터와 참조 데이터 함께 사용)
    combined_heights = heights + reference_heights
    combined_depths = depths + reference_depths
    
    print(f"\nPerson{model_index} 데이터 수집 완료:")
    print(f"실측 데이터: {len(heights)} 개")
    print(f"참조 데이터: {len(reference_heights)} 개")
    print(f"통합 데이터: {len(combined_heights)} 개")
    
    return combined_heights, combined_depths

def main():
    # 각 person 폴더별로 처리
    for model_index in [1, 2]:
        print(f"\n===== Person {model_index} 처리 시작 =====")
        
        # 이미지 처리 및 데이터 수집
        heights, depths = process_person_images(model_index)
        
        if len(heights) >= 2:
            # 높이-깊이 관계 모델 생성
            height_depth_model = create_height_depth_model(heights, depths, model_index)
            
            # 모델 저장
            if height_depth_model:
                save_height_depth_model(height_depth_model, model_index)
        else:
            print(f"Person{model_index}의 데이터가 부족하여 모델을 생성할 수 없습니다.")
    
    print("\n모든 처리가 완료되었습니다!")

if __name__ == "__main__":
    main()
