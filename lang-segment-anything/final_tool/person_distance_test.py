import os
import sys
import json
# 현재 스크립트 위치의 상위 폴더를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from lang_sam import LangSAM
import matplotlib.pyplot as plt

# 경로 설정
model_folder = "./CapstoneD/Capstone_Final/lang-segment-anything/output/human_models"  # 저장된 모델 폴더
test_base_folder = "./CapstoneD/Capstone_Final/lang-segment-anything/assets/human_data/test"  # 테스트 이미지 기본 폴더
test_depth_folder = "./CapstoneD/Capstone_Final/lang-segment-anything/assets/human_data/human_depth_map/test"  # 테스트 이미지 depth 폴더
output_folder = "./CapstoneD/Capstone_Final/lang-segment-anything/output/human/test_results"  # 결과 저장 폴더

# 결과 저장 폴더 생성
os.makedirs(output_folder, exist_ok=True)

class HeightDepthModel:
    """
    JSON 파일에서 로드한 높이-깊이 변환 모델
    """
    def __init__(self, coefficient, intercept):
        self.coefficient = coefficient
        self.intercept = intercept
    
    def predict(self, X):
        """sklearn 모델과 유사한 인터페이스 제공"""
        if isinstance(X, list):
            if len(X) == 1 and isinstance(X[0], list) and len(X[0]) == 1:
                # [[height]] 형식일 경우 스칼라 값으로 반환
                return float(X[0][0] * self.coefficient + self.intercept)
            X = np.array(X)
        
        result = X * self.coefficient + self.intercept
        
        # 결과가 단일 값을 가진 배열이면 스칼라로 변환
        if isinstance(result, np.ndarray) and result.size == 1:
            return float(result.item())
        return result

def load_height_depth_model(model_index):
    """
    저장된 높이-깊이 비례식 모델을 JSON 파일에서 로드
    """
    model_path = os.path.join(model_folder, f"height_depth_model_person{model_index}.json")
    
    if not os.path.exists(model_path):
        print(f"모델 JSON 파일이 존재하지 않습니다: {model_path}")
        
        # 대체 방법으로 pkl 파일 확인
        pkl_path = os.path.join(model_folder, f"height_depth_model_person{model_index}.pkl")
        if os.path.exists(pkl_path):
            print(f"대신 pickle 파일을 시도합니다: {pkl_path}")
            try:
                import pickle
                with open(pkl_path, 'rb') as file:
                    model = pickle.load(file)
                
                # 모델 계수 출력
                a = model.coef_[0]
                b = model.intercept_
                print(f"로드된 높이-깊이 비례식 모델 (pickle): depth = {a:.6f} * height + {b:.6f}")
                return model
            except Exception as e:
                print(f"pickle 모델 로드 중 오류 발생: {str(e)}")
                return None
        return None
    
    try:
        # JSON 파일 로드
        with open(model_path, 'r') as file:
            model_data = json.load(file)
            
        # 필요한 계수 추출
        coefficient = model_data.get("coefficient")
        intercept = model_data.get("intercept")
        formula = model_data.get("formula", "")
        
        # HeightDepthModel 클래스의 인스턴스 생성
        model = HeightDepthModel(coefficient, intercept)
        
        # 모델 정보 출력
        print(f"로드된 높이-깊이 비례식 모델 (JSON): {formula}")
        print(f"계수: {coefficient:.6f}, 절편: {intercept:.6f}")
        
        # 모델 생성 날짜가 있으면 출력
        if "date_created" in model_data:
            print(f"모델 생성 날짜: {model_data['date_created']}")
        
        return model
    except Exception as e:
        print(f"JSON 모델 로드 중 오류 발생: {str(e)}")
        return None

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

def calculate_mask_height(mask):
    """
    마스크의 높이(max_y - min_y) 계산
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

def process_test_images(model_index, sam_model):
    """
    특정 모델 인덱스에 해당하는 테스트 이미지들을 처리
    """
    # 모델별 테스트 이미지 폴더
    test_folder = os.path.join(test_base_folder, f"person{model_index}")
    
    if not os.path.exists(test_folder):
        print(f"테스트 이미지 폴더를 찾을 수 없습니다: {test_folder}")
        return
    
    # JSON 파일에서 모델 로드
    height_depth_model = load_height_depth_model(model_index)
    if height_depth_model is None:
        print(f"모델 {model_index}를 로드할 수 없습니다. 건너뜁니다.")
        return
    
    # 테스트 이미지 처리 (각 모델별로 5장의 이미지: test1~test5.png)
    for i in range(1, 6):
        image_name = f"test{i}.png"
        input_image_path = os.path.join(test_folder, image_name)
        
        if not os.path.exists(input_image_path):
            print(f"테스트 이미지가 존재하지 않습니다: {input_image_path}")
            continue
        
        print(f"\n모델 {model_index}, 테스트 이미지 {i} 처리 중...\n")
        
        # 이미지 로드
        image_pil = Image.open(input_image_path).convert("RGB")
        
        # 출력 이미지 설정
        output_image = image_pil.copy()
        draw = ImageDraw.Draw(output_image)
        
        # 여러 프롬프트 정의 (훈련과 동일한 프롬프트 사용)
        text_prompts = ["person .", "people .", "human ."]
        
        # 각 프롬프트를 개별적으로 처리하여 결과를 결합
        all_results = []
        for prompt in text_prompts:
            result = sam_model.predict([image_pil], [prompt])
            all_results.extend(result)
            print(f"프롬프트 처리 완료: {prompt}")
        
        # 모든 마스크 결합 (전체 사람 마스크)
        combined_person_mask = np.zeros((image_pil.height, image_pil.width), dtype=bool)
        
        for result in all_results:
            for mask in result["masks"]:
                mask_np = np.array(mask, dtype=bool)
                combined_person_mask = np.logical_or(combined_person_mask, mask_np)
        
        # 마스크의 높이 계산
        mask_height, min_y, max_y = calculate_mask_height(combined_person_mask)
        
        # 실제 depth 정보 로드 (검증용)
        depth_image = None
        # depth 파일 경로 수정 - 각 모델의 person 폴더 안에 있는 depth 파일 접근
        depth_path_npz = os.path.join(test_depth_folder, f"person{model_index}", f"test{i}.npz")

        if os.path.exists(depth_path_npz):
            try:
                depth_data = np.load(depth_path_npz)
                keys = list(depth_data.keys())
                
                if len(keys) > 0:
                    key = keys[0]
                    depth_image = depth_data[key]
                    print(f"로드된 depth 이미지: {depth_path_npz} (키: {key})")
                    print(f"Depth 이미지 형태: {depth_image.shape}, 타입: {depth_image.dtype}")
                else:
                    print("NPZ 파일에 배열이 없습니다")
            except Exception as e:
                print(f"Error loading depth npz: {str(e)}")
                
                # NPZ 로드 실패 시, NPY 파일도 시도
                depth_path_npy = os.path.join(test_depth_folder, f"person{model_index}", f"test{i}.npy")
                if os.path.exists(depth_path_npy):
                    try:
                        depth_image = np.load(depth_path_npy)
                        print(f"NPY 파일에서 로드 성공: {depth_path_npy}")
                        print(f"Depth 이미지 형태: {depth_image.shape}, 타입: {depth_image.dtype}")
                    except Exception as e2:
                        print(f"Error loading depth npy: {str(e2)}")
        else:
            print(f"Warning: Depth file not found: {depth_path_npz}")
        
        # 실제 depth 계산 (있는 경우)
        actual_depth = "알 수 없음"
        if depth_image is not None and np.any(combined_person_mask):
            actual_depth_value, _ = extract_mask_depth_info(combined_person_mask, depth_image)
            actual_depth = f"{actual_depth_value:.4f}"
        
        # JSON 모델을 사용하여 마스크 높이로부터 depth 예측
        if mask_height > 0:
            # JSON에서 로드한 모델을 사용하여 예측
            predicted_depth = height_depth_model.predict([[mask_height]])
            
            # 배열이면 첫 번째 요소를 추출
            if isinstance(predicted_depth, np.ndarray):
                predicted_depth = float(predicted_depth[0])
            
            print(f"테스트 이미지 {i} - 마스크 높이: {mask_height}, 예측 depth: {predicted_depth:.4f}")
            
            # 정확도 계산 (실제 depth 값이 있는 경우)
            accuracy_info = ""
            if actual_depth != "알 수 없음":
                actual_depth_value = float(actual_depth)
                error = abs(predicted_depth - actual_depth_value)
                relative_error = (error / actual_depth_value) * 100 if actual_depth_value > 0 else float('inf')
                accuracy = max(0, 100 - relative_error)
                accuracy_info = f"\n정확도: {accuracy:.2f}%"
                print(f"오차: {error:.4f}, 상대 오차: {relative_error:.2f}%, 정확도: {accuracy:.2f}%")
            
            # 결과를 이미지에 표시
            try:
                # 폰트 설정 (기본 폰트 사용)
                font = ImageFont.load_default()
                
                # 텍스트 그리기
                text = f"model: {model_index}, num: {i}\nmask height: {mask_height}\npredicted depth: {predicted_depth:.4f}\nactual depth: {actual_depth}{accuracy_info}"
                draw.rectangle([10, 10, 350, 120], fill=(0, 0, 0, 180))
                draw.text((20, 20), text, fill=(255, 255, 255), font=font)
                
                # 마스크를 시각화 (반투명 오버레이로)
                mask_overlay = Image.new("RGBA", image_pil.size, (0, 0, 0, 0))
                mask_draw = ImageDraw.Draw(mask_overlay)
                
                for y in range(combined_person_mask.shape[0]):
                    for x in range(combined_person_mask.shape[1]):
                        if combined_person_mask[y, x]:
                            mask_draw.point((x, y), fill=(255, 0, 0, 128))
                
                # 마스크의 높이를 시각적으로 표시 (가로 선)
                mask_draw.line([(0, min_y), (image_pil.width, min_y)], fill=(0, 255, 0, 255), width=2)
                mask_draw.line([(0, max_y), (image_pil.width, max_y)], fill=(0, 255, 0, 255), width=2)
                
                # 이미지와 마스크 오버레이 합성
                output_image = Image.alpha_composite(output_image.convert("RGBA"), mask_overlay)
                output_image = output_image.convert("RGB")
                
                # 결과 저장 (모델 번호와 테스트 번호를 포함)
                output_path = os.path.join(output_folder, f"model{model_index}_test{i}_result.png")
                output_image.save(output_path)
                print(f"결과 이미지 저장: {output_path}")
                
            except Exception as e:
                print(f"이미지에 결과 표시 중 오류 발생: {str(e)}")
        else:
            print(f"테스트 이미지 {i}에서 사람 마스크를 찾을 수 없거나 높이가 0입니다.")

def main():
    # LangSAM 모델 초기화 (모든 테스트에 사용)
    sam_model = LangSAM()
    
    # 모델 1과 2에 대해 테스트 실행
    for model_index in [1, 2]:
        print(f"\n===== 모델 {model_index} 테스트 시작 =====\n")
        process_test_images(model_index, sam_model)
    
    print("\n모든 테스트가 완료되었습니다.")

if __name__ == "__main__":
    main()