import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
# 현재 스크립트 위치의 상위 폴더를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import cv2
from PIL import Image, ImageDraw
from lang_sam import LangSAM
from tqdm import tqdm
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

def process_images_with_depth(input_folder, output_folder, depth_maps_folder):
    """
    입력 폴더의 모든 이미지를 처리하여 마스크 추출, 영역 분할, 깊이 값 분석 후 결과를 JSON으로 저장합니다.
    
    Args:
        input_folder (str): 입력 이미지 폴더 경로
        output_folder (str): 결과 JSON 및 시각화 이미지 저장 폴더 경로
        depth_maps_folder (str): 깊이 맵(.npy 파일) 폴더 경로
    """
    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "overlays"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "visualization"), exist_ok=True)
    
    # 모델 초기화
    print("LangSAM 모델 초기화 중...")
    model = LangSAM()
    print("모델 초기화 완료")
    
    # 프롬프트 정의
    text_prompts = ["human ."]
    
    # 입력 폴더에서 모든 이미지 파일 찾기
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(input_folder) if os.path.splitext(f.lower())[1] in valid_extensions]
    
    print(f"총 {len(image_files)}개 이미지 파일을 처리합니다.")
    
    # 모든 이미지 파일에 대한 결과를 저장할 데이터 구조
    all_results = {}
    
    # 각 이미지 처리
    for image_filename in tqdm(image_files, desc="이미지 처리 중"):
        image_path = os.path.join(input_folder, image_filename)
        image_name = os.path.splitext(image_filename)[0]
        
        # 이미지 로드
        try:
            image_pil = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"이미지 '{image_filename}' 로드 중 오류 발생: {e}")
            continue
        
        # 이미지 너비와 높이
        width, height = image_pil.size
        
        # 해당 이미지에 대한 깊이 맵 파일 경로
        depth_map_path = os.path.join(depth_maps_folder, f"{image_name}_depth.npy")
        
        # 깊이 맵 로드 (npy 형식 기대)
        try:
            if os.path.exists(depth_map_path):
                depth_map = np.load(depth_map_path)
            else:
                print(f"경고: 깊이 맵 파일이 없습니다: {depth_map_path}")
                depth_map = None
        except Exception as e:
            print(f"깊이 맵 '{depth_map_path}' 로드 중 오류 발생: {e}")
            depth_map = None
        
        # 오버레이 및 마스크만 표시할 이미지 생성
        overlay_image = image_pil.copy()
        overlay = Image.new("RGBA", overlay_image.size, (255, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # 검정색 배경에 마스크만 표시할 이미지
        mask_only_image = Image.new("RGB", image_pil.size, (0, 0, 0))
        mask_only_draw = ImageDraw.Draw(mask_only_image)
        
        # 모든 프롬프트에서 마스크 결합
        combined_mask = np.zeros((height, width), dtype=bool)
        
        # 각 프롬프트 처리
        for prompt in text_prompts:
            # 마스크 예측
            result = model.predict([image_pil], [prompt])
            
            # 모든 마스크 결합
            for res in result:
                for mask in res["masks"]:
                    # 마스크를 boolean 배열로 변환
                    mask_np = np.array(mask, dtype=bool)
                    # 기존 마스크와 OR 연산으로 결합
                    combined_mask = np.logical_or(combined_mask, mask_np)
        
        # 연결된 컴포넌트 라벨링
        labeled_mask, num_features = ndimage.label(combined_mask)
        print(f"{image_filename}: {num_features}개의 연결된 영역 감지됨")
        
        # 라벨링된 영역 시각화 (각 영역별 다른 색상)
        label_viz = np.zeros((height, width, 3), dtype=np.uint8)
        colors = plt.cm.rainbow(np.linspace(0, 1, num_features + 1))[:, :3]  # +1은 배경 고려
        
        # 이미지의 결과 정보를 저장할 객체
        image_result = {
            "image_name": image_filename,
            "width": width,
            "height": height,
            "masks": []
        }
        
        # 각 라벨(영역)별 정보 추출 및 시각화
        for label_idx in range(1, num_features + 1):  # 1부터 시작 (0은 배경)
            mask = (labeled_mask == label_idx)
            
            # 영역의 바운딩 박스 계산
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue  # 빈 마스크는 건너뜀
                
            min_x, max_x = x_indices.min(), x_indices.max()
            min_y, max_y = y_indices.min(), y_indices.max()
            
            # 마스크 시각화
            color_rgb = tuple(int(c * 255) for c in colors[label_idx][:3])
            label_viz[mask] = color_rgb
            
            # 해당 마스크 영역의 깊이 값 평균 계산
            if depth_map is not None:
                try:
                    # 마스크 영역의 깊이 값 추출
                    mask_depth_values = depth_map[mask]
                    avg_depth = float(np.mean(mask_depth_values))
                    min_depth = float(np.min(mask_depth_values))
                    max_depth = float(np.max(mask_depth_values))
                except Exception as e:
                    print(f"깊이 값 계산 중 오류 발생: {e}")
                    avg_depth = 0
                    min_depth = 0
                    max_depth = 0
            else:
                avg_depth = 0
                min_depth = 0
                max_depth = 0
            
            # 마스크 정보 저장
            mask_info = {
                "mask_id": label_idx,
                "bbox": [int(min_x), int(min_y), int(max_x), int(max_y)],
                "area": int(np.sum(mask)),
                "avg_depth": avg_depth,
                "min_depth": min_depth,
                "max_depth": max_depth
            }
            
            image_result["masks"].append(mask_info)
            
            # 오버레이 및 마스크에 영역 표시
            for y in range(height):
                for x in range(width):
                    if mask[y, x]:
                        draw.point((x, y), fill=(color_rgb[0], color_rgb[1], color_rgb[2], 128))
                        mask_only_draw.point((x, y), fill=color_rgb)
        
        # 결과 저장
        all_results[image_filename] = image_result
        
        # 오버레이 이미지 저장
        final_overlay_image = Image.alpha_composite(overlay_image.convert("RGBA"), overlay)
        overlay_path = os.path.join(output_folder, "overlays", f"overlay_{image_filename}")
        final_overlay_image.convert("RGB").save(overlay_path)
        
        # 마스크만 있는 이미지 저장
        mask_path = os.path.join(output_folder, "masks", f"mask_{image_filename}")
        mask_only_image.save(mask_path)
        
        # 라벨링된 마스크 시각화 저장
        viz_path = os.path.join(output_folder, "visualization", f"labeled_mask_{image_filename}")
        plt.figure(figsize=(10, 10))
        plt.imshow(label_viz)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(viz_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    # 모든 결과를 JSON 파일로 저장
    json_path = os.path.join(output_folder, "mask_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    
    print(f"처리 완료! 결과가 '{json_path}'에 저장되었습니다.")

# 메인 함수 실행
if __name__ == "__main__":
    # 폴더 경로 설정
    input_folder = "/home/protove/vsc/CapstoneD/Capstone_Final/lang-segment-anything/output/total_s1_fps3"
    output_folder = "/home/protove/vsc/CapstoneD/Capstone_Final/lang-segment-anything/output/total_s1_human_depth_analysis"
    depth_maps_folder = "/home/protove/vsc/CapstoneD/Capstone_Final/lang-segment-anything/output/total_s1_fps3/depth_maps"
    
    # 이미지 처리 및 분석 실행
    process_images_with_depth(input_folder, output_folder, depth_maps_folder)
