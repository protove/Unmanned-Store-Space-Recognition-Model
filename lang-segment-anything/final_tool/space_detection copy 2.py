# 코드 맨 위에 추가
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
# 현재 스크립트 위치의 상위 폴더를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw
import numpy as np
from lang_sam import LangSAM
import random
import os
from scipy import ndimage
import cv2
import json
import datetime

# 상수 정의
MASK_SIZE_THRESHOLD = 5000  # 마스크 크기 임계값 (픽셀 수)
MIN_COMPONENT_SIZE = 150    # 최소 연결 요소 크기
VARIANCE_THRESHOLD = 95000  # 좌표 분산 임계값
DISTANCE_THRESHOLD = 30     # 아이스크림 디스플레이 병합을 위한 거리 임계값

# 경로 설정
BASE_DIR = "./CapstoneD/Capstone_Final/lang-segment-anything"
ASSET_DIR = f"{BASE_DIR}/assets/space_data"
OUTPUT_DIR = f"{BASE_DIR}/output/final"
DEPTH_DIR = f"{ASSET_DIR}/depth_map"

# 출력 디렉토리 구조
OUTPUT_DIRS = [
    f"{OUTPUT_DIR}/overlay",
    f"{OUTPUT_DIR}/overlay_remove",
    f"{OUTPUT_DIR}/mask_only/display",
    f"{OUTPUT_DIR}/mask_only/floor",
    f"{OUTPUT_DIR}/mask_only/filtered",
    f"{OUTPUT_DIR}/mask_only/non_overlap",
    f"{OUTPUT_DIR}/mask_only/top3_masks",
    f"{OUTPUT_DIR}/mask_only/classified_masks",
    f"{OUTPUT_DIR}/mask_only/contour_masks",
    f"{OUTPUT_DIR}/json"
]

def setup_directories():
    """필요한 모든 디렉토리 생성"""
    for directory in OUTPUT_DIRS:
        os.makedirs(directory, exist_ok=True)

def check_depth_folder():
    """Depth 폴더 내용 확인 (디버깅용)"""
    print("Checking depth folder contents:")
    if os.path.exists(DEPTH_DIR):
        depth_files = os.listdir(DEPTH_DIR)
        for f in depth_files[:5]:  # 처음 5개 파일만 출력
            print(f"- {f}")
        if len(depth_files) > 5:
            print(f"... and {len(depth_files) - 5} more files")
    else:
        print(f"Depth folder not found: {DEPTH_DIR}")

def compute_overlap_ratio(mask1, mask2):
    """두 마스크 간의 겹치는 비율을 계산"""
    intersection = np.logical_and(mask1, mask2)
    intersection_size = np.sum(intersection)
    
    # 더 작은 마스크를 기준으로 겹치는 비율 계산
    mask1_size = np.sum(mask1)
    mask2_size = np.sum(mask2)
    smaller_size = min(mask1_size, mask2_size)
    
    if smaller_size == 0:
        return 0.0
    
    return intersection_size / smaller_size

def cluster_overlapping_masks(mask_list, threshold=0.9):
    """지정된 임계값 이상으로 겹치는 마스크들을 클러스터링"""
    if not mask_list:
        return []
    
    # 클러스터 초기화
    clusters = []
    
    # 각 마스크에 대해 처리
    for mask_info in mask_list:
        mask = mask_info['mask']
        size = mask_info['size']
        color = mask_info['color']
        
        # 이 마스크가 어떤 클러스터에 속하는지 확인
        found_cluster = False
        
        for cluster in clusters:
            # 클러스터의 대표 마스크와 현재 마스크의 겹침 비율 계산
            overlap_ratio = compute_overlap_ratio(cluster['combined_mask'], mask)
            
            if overlap_ratio >= threshold:
                # 클러스터에 마스크 추가
                cluster['masks'].append(mask_info)
                # 클러스터의 합쳐진 마스크 업데이트 (논리적 OR)
                cluster['combined_mask'] = np.logical_or(cluster['combined_mask'], mask)
                # 클러스터 크기 업데이트
                cluster['total_size'] = np.sum(cluster['combined_mask'])
                found_cluster = True
                break
        
        if not found_cluster:
            # 새로운 클러스터 생성
            new_cluster = {
                'masks': [mask_info],
                'combined_mask': mask.copy(),
                'total_size': size,
                'color': color
            }
            clusters.append(new_cluster)
    
    # 클러스터를 크기에 따라 내림차순 정렬
    clusters.sort(key=lambda c: c['total_size'], reverse=True)
    
    return clusters

def calculate_avg_depth(mask, depth_image):
    """마스크 영역의 평균 depth 값을 계산"""
    if depth_image is None:
        print("Warning: No depth image available for calculation")
        return 0.0
    
    # 마스크가 True인 부분의 depth 값만 추출
    masked_depth = depth_image[mask]
    
    # 디버그 정보 출력
    if len(masked_depth) > 0:
        print(f"Mask depth values - min: {masked_depth.min():.4f}, max: {masked_depth.max():.4f}, mean: {np.mean(masked_depth):.4f}")
    
    # 평균 계산 (값이 없을 경우 0 반환)
    if len(masked_depth) == 0:
        print("Warning: No valid depth pixels in mask")
        return 0.0
    
    return np.mean(masked_depth)

def calculate_mask_centroid(mask):
    """마스크 영역의 중심 좌표(centroid)를 계산"""
    # 마스크에서 True인 픽셀의 인덱스 찾기
    y_indices, x_indices = np.where(mask)
    
    # 평균 계산 (값이 없을 경우 (0, 0) 반환)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return (0, 0)
    
    avg_x = np.mean(x_indices)
    avg_y = np.mean(y_indices)
    
    return (avg_x, avg_y)

def create_contour_mask(mask):
    """마스크 영역을 감싸는 단일 컨투어를 생성"""
    # 마스크를 8비트 이미지로 변환 (findContours 함수를 위해)
    mask_uint8 = mask.astype(np.uint8) * 255
    
    # 컨투어 찾기
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Warning: No contour found for mask")
        return mask.copy()
    
    # 모든 컨투어 점들을 하나의 배열로 합치기
    all_points = np.vstack([contour.reshape(-1, 2) for contour in contours])
    
    # 모든 점들을 포함하는 볼록 껍질(Convex Hull) 찾기
    if len(all_points) > 0:
        hull = cv2.convexHull(all_points)
        
        # 빈 이미지에 컨투어 그리기
        contour_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.fillPoly(contour_mask, [hull], 1)
        
        # 디버그 출력 추가
        pixels_in_contour = np.sum(contour_mask)
        print(f"Combined contour created with {pixels_in_contour} pixels, enclosing {len(contours)} individual contours")
        
        # 불리언 배열로 변환하여 반환
        return contour_mask.astype(np.bool_)
    else:
        print("Warning: No valid points found for convex hull")
        return mask.copy()

def save_mask_info_to_json(image_index, classified_masks, json_folder):
    """분류된 마스크 정보를 JSON 파일로 저장"""
    json_path = os.path.join(json_folder, f"mask_info_test{image_index}.json")
    
    # 마스크 설명 매핑
    mask_descriptions = {
        1: "Left side display",
        2: "Farthest display",
        3: "Right side display"
    }
    
    # JSON 데이터 구조 생성
    masks_data = []
    
    # 각 마스크 정보 저장
    for key, mask_info in classified_masks.items():
        # 마스크의 경계 좌표 계산
        mask = mask_info['mask']
        y_indices, x_indices = np.where(mask)
        
        # 경계 좌표 계산 (빈 마스크의 경우 기본값 사용)
        min_x = int(np.min(x_indices)) if len(x_indices) > 0 else 0
        max_x = int(np.max(x_indices)) if len(x_indices) > 0 else 0
        min_y = int(np.min(y_indices)) if len(y_indices) > 0 else 0
        max_y = int(np.max(y_indices)) if len(y_indices) > 0 else 0
        
        # 마스크 데이터 객체 생성
        mask_data = {
            'mask_key': int(key),
            'avg_depth': float(f"{mask_info['avg_depth']:.4f}"),
            'min_x': min_x,
            'max_x': max_x,
            'min_y': min_y,
            'max_y': max_y,
            'description': mask_descriptions.get(int(key), ""),
            'width': max_x - min_x,
            'height': max_y - min_y
        }
        
        masks_data.append(mask_data)
    
    # JSON 데이터 최종 구조
    json_data = {
        'image_index': image_index,
        'image_name': f"test{image_index}.png",
        'timestamp': str(datetime.datetime.now()),
        'masks': masks_data
    }
    
    # JSON 파일로 저장
    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(json_data, file, indent=4)
    
    print(f"마스크 정보가 JSON 파일에 저장되었습니다: {json_path}")
    
    # 모든 마스크 정보를 하나의 통합 JSON 파일에도 추가
    combined_json_path = os.path.join(json_folder, "all_masks_info.json")
    
    # 파일이 존재하는지 확인하고, 존재하면 읽어오기
    all_data = []
    if os.path.isfile(combined_json_path):
        try:
            with open(combined_json_path, 'r', encoding='utf-8') as file:
                all_data = json.load(file)
        except json.JSONDecodeError:
            print(f"통합 JSON 파일을 읽는 데 실패했습니다. 새 파일을 생성합니다.")
            all_data = []
    
    # 현재 이미지 데이터 추가
    all_data.append(json_data)
    
    # 업데이트된 데이터를 파일에 쓰기
    with open(combined_json_path, 'w', encoding='utf-8') as file:
        json.dump(all_data, file, indent=4)
    
    print(f"마스크 정보가 통합 JSON 파일에 추가되었습니다: {combined_json_path}")

def calculate_mask_variance(mask):
    """마스크의 좌표 분산을 계산"""
    # 마스크에서 True인 픽셀의 좌표 찾기
    y_indices, x_indices = np.where(mask)
    
    # 픽셀이 없는 경우 분산은 0
    if len(x_indices) == 0 or len(y_indices) == 0:
        return 0.0
    
    # x 좌표와 y 좌표의 분산 계산
    x_variance = np.var(x_indices)
    y_variance = np.var(y_indices)
    
    # x, y 분산의 합을 반환 (총 분산)
    total_variance = x_variance + y_variance
    
    print(f"마스크 좌표 분산: x_var={x_variance:.2f}, y_var={y_variance:.2f}, total={total_variance:.2f}")
    
    return total_variance

def remove_small_components(mask, min_size=100):
    """마스크에서 지정된 크기보다 작은 연결 요소(객체)를 제거"""
    # 마스크가 비어있으면 그대로 반환
    if not np.any(mask):
        return mask.copy()
    
    # 연결된 요소 분석
    labeled_mask, num_components = ndimage.label(mask)
    
    # 크기가 임계값 이상인 객체만 유지
    component_sizes = np.bincount(labeled_mask.ravel())
    component_sizes[0] = 0  # 배경(0) 크기는 무시
    
    # 작은 요소 제거 (배경을 포함하지 않으므로 1부터 시작)
    small_components = np.where(component_sizes < min_size)[0]
    mask_cleaned = np.isin(labeled_mask, small_components, invert=True)
    
    # 제거된 연결 요소 수와 픽셀 수 출력
    removed_components = len(small_components) if len(small_components) > 0 else 0
    original_pixels = np.sum(mask)
    cleaned_pixels = np.sum(mask_cleaned)
    removed_pixels = original_pixels - cleaned_pixels
    
    print(f"작은 연결 요소 제거: {removed_components}개 연결 요소, {removed_pixels} 픽셀 제거됨 ({removed_pixels/original_pixels*100:.1f}%)")
    
    return mask_cleaned

def calculate_min_distance(mask1, mask2):
    """두 마스크 간의 최소 거리 계산"""
    # 마스크에서 True인 픽셀의 좌표 찾기
    y1, x1 = np.where(mask1)
    y2, x2 = np.where(mask2)
    
    # 좌표가 없는 경우 무한대 거리 반환
    if len(x1) == 0 or len(x2) == 0:
        return float('inf')
    
    # 성능 최적화: 좌표를 배열로 변환
    points1 = np.column_stack((x1, y1))
    points2 = np.column_stack((x2, y2))
    
    # 최소 거리 계산을 위한 초기값
    min_dist = float('inf')
    
    # 계산 부하 최적화: 일정 수 이상 픽셀이 있는 경우 샘플링
    MAX_POINTS = 300  # 최대 비교할 점 수
    if len(points1) > MAX_POINTS:
        indices = np.random.choice(len(points1), MAX_POINTS, replace=False)
        points1 = points1[indices]
    if len(points2) > MAX_POINTS:
        indices = np.random.choice(len(points2), MAX_POINTS, replace=False)
        points2 = points2[indices]
    
    # 모든 점 쌍 사이의 거리 계산
    for p1 in points1:
        # 벡터화된 거리 계산으로 속도 향상
        dists = np.sqrt(np.sum((points2 - p1)**2, axis=1))
        curr_min = np.min(dists)
        if curr_min < min_dist:
            min_dist = curr_min
    
    return min_dist

def load_depth_image(image_index):
    """이미지 인덱스에 해당하는 depth 이미지 로드"""
    depth_filename_npz = f"test{image_index}.npz"
    depth_path_npz = os.path.join(DEPTH_DIR, depth_filename_npz)
    
    if os.path.exists(depth_path_npz):
        try:
            # npz 파일 로드
            depth_data = np.load(depth_path_npz)
            
            # npz 파일 구조 확인 (디버깅용)
            print(f"Depth npz keys: {list(depth_data.keys())}")
            
            # npz 파일에서 실제 depth 데이터 접근
            if 'arr_0' in depth_data:
                depth_image = depth_data['arr_0']
            elif 'depth' in depth_data:
                depth_image = depth_data['depth']
            else:
                # 첫 번째 배열을 사용
                key = list(depth_data.keys())[0]
                depth_image = depth_data[key]
                print(f"Using key '{key}' from npz file")
            
            print(f"Loaded depth npz: {depth_path_npz}")
            print(f"Depth range: min={depth_image.min():.4f}, max={depth_image.max():.4f}, mean={depth_image.mean():.4f}")
            
            return depth_image
        except Exception as e:
            print(f"Error loading depth npz: {e}")
            return None
    else:
        print(f"Warning: Depth npz file not found: {depth_path_npz}")
        return None

def create_floor_mask(model, image_index):
    """바닥 마스크 생성 함수"""
    image_name = f"test{image_index}.png"
    input_image_path = f"{ASSET_DIR}/{image_name}"
    
    # 이미지 로드
    image_pil = Image.open(input_image_path).convert("RGB")
    
    # 바닥 관련 프롬프트 정의
    floor_prompts = ["floor.", "ground.", "banner."]
    
    # 합쳐진 바닥 마스크를 저장할 배열
    combined_floor_mask = np.zeros((image_pil.height, image_pil.width), dtype=bool)
    
    # 각 프롬프트를 개별적으로 처리
    all_results = []
    for prompt in floor_prompts:
        result = model.predict([image_pil], [prompt])
        all_results.extend(result)
        print(f"Processed floor prompt: {prompt}")
    
    # 모든 바닥 마스크를 합침
    for result in all_results:
        for mask in result["masks"]:
            mask_np = np.array(mask, dtype=bool)
            
            # 마스크 크기 확인
            mask_size = np.sum(mask_np)
            if mask_size < MASK_SIZE_THRESHOLD:
                print(f"Filtered out small floor mask with size {mask_size} pixels")
                continue
                
            # 바닥 마스크 병합
            combined_floor_mask = combined_floor_mask | mask_np
    
    print(f"총 바닥 마스크 픽셀 수: {np.sum(combined_floor_mask)}")
    
    return combined_floor_mask, image_pil

def process_display_masks(model, image_index, floor_mask, image_pil, depth_image=None):
    """디스플레이 마스크 처리 함수"""
    # 디스플레이 관련 프롬프트
    display_prompts = [
        "stand.", "shelf.", "display rack.", "shelf stand.", 
        "merchandise stand.", "promotional rack.", 
        "Ice cream display freezer .", "Ice cream display freezers .", 
        "Ice cream freezer .", "glass-top freezer ."
    ]
    
    # 이미지 준비
    image_size = image_pil.size
    
    # 각종 이미지 객체 생성
    overlay_image = image_pil.copy()
    overlay = Image.new("RGBA", overlay_image.size, (255, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    mask_only_image = Image.new("RGB", image_pil.size, (0, 0, 0))
    mask_only_draw = ImageDraw.Draw(mask_only_image)
    
    filtered_mask_image = Image.new("RGB", image_pil.size, (0, 0, 0))
    filtered_mask_draw = ImageDraw.Draw(filtered_mask_image)
    
    non_overlap_image = Image.new("RGB", image_pil.size, (0, 0, 0))
    non_overlap_draw = ImageDraw.Draw(non_overlap_image)
    
    top3_image = Image.new("RGB", image_pil.size, (0, 0, 0))
    top3_draw = ImageDraw.Draw(top3_image)
    
    classified_image = Image.new("RGB", image_pil.size, (0, 0, 0))
    classified_draw = ImageDraw.Draw(classified_image)
    
    contour_image = Image.new("RGB", image_pil.size, (0, 0, 0))
    contour_draw = ImageDraw.Draw(contour_image)
    
    # 디스플레이 프롬프트 처리
    all_results = []
    for prompt in display_prompts:
        result = model.predict([image_pil], [prompt])
        all_results.extend(result)
        print(f"Processed display prompt: {prompt}")
    
    # 바닥과 겹치지 않는 마스크 추출
    non_overlap_masks = []
    
    # 마스크 처리
    for result in all_results:
        for mask in result["masks"]:
            mask_np = np.array(mask, dtype=bool)
            
            # 마스크 크기 확인
            mask_size = np.sum(mask_np)
            if mask_size < MASK_SIZE_THRESHOLD:
                print(f"Filtered out small display mask with size {mask_size} pixels")
                continue
            
            # 랜덤 색상 생성
            random_color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
                128
            )
            random_color_no_alpha = random_color[:3]
            
            # 일반 마스크 처리
            for y in range(mask_np.shape[0]):
                for x in range(mask_np.shape[1]):
                    if mask_np[y, x]:
                        draw.point((x, y), fill=random_color)
                        mask_only_draw.point((x, y), fill=random_color_no_alpha)
                        filtered_mask_draw.point((x, y), fill=random_color_no_alpha)
            
            # 바닥 마스크와 겹치지 않는 부분만 계산
            non_overlap_mask = np.logical_and(mask_np, np.logical_not(floor_mask))
            non_overlap_size = np.sum(non_overlap_mask)
            
            if non_overlap_size > 0:
                # 마스크 좌표 분산 계산
                coord_variance = calculate_mask_variance(non_overlap_mask)
                
                # 분산이 임계값 이하인 마스크만 처리
                if coord_variance <= VARIANCE_THRESHOLD:
                    # 마스크의 평균 depth 값 계산
                    avg_depth = calculate_avg_depth(non_overlap_mask, depth_image)
                    
                    # 마스크의 중심 좌표 계산
                    centroid_x, centroid_y = calculate_mask_centroid(non_overlap_mask)
                    
                    # 겹치지 않는 영역의 마스크와 색상 정보 및 특성 정보를 저장
                    non_overlap_masks.append({
                        'mask': non_overlap_mask,
                        'color': random_color_no_alpha,
                        'size': non_overlap_size,
                        'avg_depth': avg_depth,
                        'centroid_x': centroid_x,
                        'centroid_y': centroid_y,
                        'coord_variance': coord_variance
                    })
                    
                    # non_overlap 이미지에 마스크 적용
                    for y in range(mask_np.shape[0]):
                        for x in range(mask_np.shape[1]):
                            if non_overlap_mask[y, x]:
                                non_overlap_draw.point((x, y), fill=random_color_no_alpha)
                
                    print(f"마스크 포함: 크기 {non_overlap_size} 픽셀, 좌표 분산 {coord_variance:.2f}")
                else:
                    print(f"좌표 분산이 큰 마스크 제외: 크기 {non_overlap_size} 픽셀, 좌표 분산 {coord_variance:.2f}")
    
    print(f"총 {len(non_overlap_masks)}개의 마스크가 필터링 후 남음")
    
    return non_overlap_masks, (overlay_image, overlay, mask_only_image, filtered_mask_image, non_overlap_image, top3_image, classified_image, contour_image)

def process_clusters(non_overlap_masks, depth_image, top3_draw, classified_draw, contour_draw):
    """마스크 클러스터링 및 특성별 분류 처리"""
    # 클러스터링
    mask_clusters = cluster_overlapping_masks(non_overlap_masks, threshold=0.9)
    print(f"총 {len(non_overlap_masks)}개의 마스크가 {len(mask_clusters)}개의 클러스터로 통합됨")
    
    # 상위 5개 클러스터 선택
    top_clusters = mask_clusters[:5]
    print(f"상위 {len(top_clusters)}개 클러스터 선택")
    
    # 가장 멀리 있는 클러스터 식별
    farthest_cluster = None
    other_top_clusters = []
    
    if depth_image is not None and len(top_clusters) > 0:
        max_depth = -float('inf')
        max_depth_idx = -1
        
        # 각 클러스터의 depth 평균 계산
        for idx, cluster in enumerate(top_clusters):
            cluster_mask = cluster['combined_mask']
            avg_depth = calculate_avg_depth(cluster_mask, depth_image)
            
            print(f"클러스터 {idx+1} 평균 depth: {avg_depth:.4f}")
            
            # depth 값이 가장 큰 클러스터 찾기
            if avg_depth > max_depth:
                max_depth = avg_depth
                max_depth_idx = idx
        
        # 가장 멀리 있는 클러스터 분리
        if max_depth_idx >= 0:
            farthest_cluster = top_clusters[max_depth_idx]
            print(f"가장 멀리 있는 클러스터 식별: 평균 depth {max_depth:.4f}")
            
            # 가장 멀리 있는 클러스터를 제외한 나머지 클러스터 저장
            other_top_clusters = [cluster for idx, cluster in enumerate(top_clusters) if idx != max_depth_idx]
    else:
        other_top_clusters = top_clusters.copy()
        print("Depth 정보가 없거나 클러스터가 없어 가장 멀리 있는 클러스터를 식별할 수 없습니다.")
    
    return process_ice_cream_clusters(farthest_cluster, other_top_clusters, depth_image, top3_draw, classified_draw, contour_draw)

def process_ice_cream_clusters(farthest_cluster, other_top_clusters, depth_image, top3_draw, classified_draw, contour_draw):
    """아이스크림 디스플레이 클러스터 처리"""
    # 아이스크림 디스플레이 마스크 식별 및 병합
    print("남은 클러스터 중 아이스크림 디스플레이 마스크 식별 시작...")
    
    # 남은 클러스터 중 아이스크림 디스플레이 관련 클러스터 식별 (여기서는 단순히 구분만 함)
    ice_cream_clusters = other_top_clusters[:1]  # 예시: 첫 번째 클러스터를 아이스크림 관련으로 간주
    remaining_clusters = other_top_clusters[1:]  # 나머지 클러스터
    
    # 아이스크림 디스플레이 클러스터 병합
    merged_ice_cream_clusters = []
    if len(ice_cream_clusters) >= 2:
        print(f"{len(ice_cream_clusters)}개의 아이스크림 디스플레이 클러스터 발견됨, 최소 거리 기반 병합 시작...")
        
        # 모든 아이스크림 클러스터 쌍의 최소 거리 계산
        adjacency = {i: [] for i in range(len(ice_cream_clusters))}
        for i in range(len(ice_cream_clusters)):
            for j in range(i+1, len(ice_cream_clusters)):
                mask1 = ice_cream_clusters[i]['combined_mask']
                mask2 = ice_cream_clusters[j]['combined_mask']
                
                # 두 마스크의 최소 거리 계산
                min_dist = calculate_min_distance(mask1, mask2)
                print(f"아이스크림 클러스터 {i}와 {j} 사이의 최소 거리: {min_dist:.2f} 픽셀")
                
                # 거리가 임계값 이하면 연결
                if min_dist <= DISTANCE_THRESHOLD:
                    adjacency[i].append(j)
                    adjacency[j].append(i)
        
        # 연결된 그룹 찾기 (깊이 우선 탐색)
        visited = [False] * len(ice_cream_clusters)
        groups = []
        
        def dfs(node, group):
            visited[node] = True
            group.append(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    dfs(neighbor, group)
        
        # 모든 클러스터를 그룹으로 묶기
        for i in range(len(ice_cream_clusters)):
            if not visited[i]:
                group = []
                dfs(i, group)
                groups.append(group)
        
        print(f"총 {len(groups)}개의 아이스크림 클러스터 그룹 형성됨")
        
        # 각 그룹의 클러스터 병합
        for group_idx, group in enumerate(groups):
            if len(group) == 1:
                # 단일 클러스터는 그대로 추가
                merged_ice_cream_clusters.append(ice_cream_clusters[group[0]])
                continue
            
            # 그룹 내 모든 클러스터 병합
            combined_mask = np.zeros_like(ice_cream_clusters[0]['combined_mask'], dtype=bool)
            all_masks = []
            
            for idx in group:
                cluster = ice_cream_clusters[idx]
                combined_mask = np.logical_or(combined_mask, cluster['combined_mask'])
                all_masks.extend(cluster['masks'])
            
            # 병합된 클러스터 정보 생성
            merged_cluster = {
                'masks': all_masks,
                'combined_mask': combined_mask,
                'total_size': np.sum(combined_mask),
                'color': (255, 255, 0)  # 노란색
            }
            
            merged_ice_cream_clusters.append(merged_cluster)
            print(f"그룹 {group_idx+1}: {len(group)}개 클러스터 병합, 최종 크기 {merged_cluster['total_size']} 픽셀")
            
            # 시각화 - 병합된 아이스크림 클러스터 표시 (노란색)
            for y in range(combined_mask.shape[0]):
                for x in range(combined_mask.shape[1]):
                    if combined_mask[y, x]:
                        top3_draw.point((x, y), fill=(255, 255, 0))
    elif len(ice_cream_clusters) == 1:
        # 아이스크림 클러스터가 하나면 그대로 추가
        merged_ice_cream_clusters = ice_cream_clusters.copy()
        print("아이스크림 클러스터가 하나만 있어 병합 불필요")
    else:
        print("아이스크림 클러스터가 없습니다.")
    
    # 모든 클러스터 결합 (가장 멀리 있는 클러스터 + 남은 일반 클러스터 + 병합된 아이스크림 클러스터)
    final_clusters = []
    
    # 가장 멀리 있는 클러스터 추가 (있는 경우)
    if farthest_cluster is not None:
        final_clusters.append(farthest_cluster)
        print("가장 멀리 있는 클러스터를 최종 리스트에 추가")
    
    # 남은 일반 클러스터 추가
    final_clusters.extend(remaining_clusters)
    print(f"남은 {len(remaining_clusters)}개 일반 클러스터를 최종 리스트에 추가")
    
    # 병합된 아이스크림 클러스터 추가
    final_clusters.extend(merged_ice_cream_clusters)
    print(f"병합된 {len(merged_ice_cream_clusters)}개 아이스크림 클러스터를 최종 리스트에 추가")
    
    # 크기 기준으로 클러스터 재정렬
    final_clusters.sort(key=lambda c: c['total_size'], reverse=True)
    print(f"최종 클러스터 수: {len(final_clusters)}개 (크기순 재정렬)")
    
    return finalize_clusters(final_clusters, depth_image, top3_draw, classified_draw, contour_draw)

def finalize_clusters(top_clusters, depth_image, top3_draw, classified_draw, contour_draw):
    """클러스터 정제 및 최종 처리"""
    # 작은 연결 요소 제거
    print("작은 연결 요소 제거 시작...")
    cleaned_top_clusters = []
    
    for idx, cluster in enumerate(top_clusters):
        # 클러스터의 결합된 마스크에서 작은 연결 요소 제거
        original_mask = cluster['combined_mask']
        cleaned_mask = remove_small_components(original_mask, min_size=MIN_COMPONENT_SIZE)
        
        # 원본 마스크와 정제된 마스크의 크기 비교
        original_size = np.sum(original_mask)
        cleaned_size = np.sum(cleaned_mask)
        
        print(f"클러스터 {idx+1}: 원본 크기 {original_size} → 정제 후 크기 {cleaned_size} ({(cleaned_size/original_size*100):.1f}%)")
        
        # 정제된 마스크로 클러스터 업데이트
        cleaned_cluster = {
            'masks': cluster['masks'],
            'combined_mask': cleaned_mask,
            'total_size': cleaned_size,
            'color': cluster['color']
        }
        
        # 정제된 클러스터 저장
        cleaned_top_clusters.append(cleaned_cluster)
        
        # 선명한 색상 차이를 위해 클러스터별로 고유 색상 지정
        if idx == 0:
            color = (255, 0, 0)  # 빨강 (가장 큰 클러스터)
        elif idx == 1:
            color = (0, 255, 0)  # 초록
        elif idx == 2:
            color = (0, 0, 255)  # 파랑
        elif idx == 3:
            color = (255, 255, 0)  # 노랑 (4번째 클러스터용)
        elif idx == 4:
            color = (0, 255, 255)  # 청록색 (5번째 클러스터용)
        
        # 정제된 마스크 그리기
        for y in range(cleaned_mask.shape[0]):
            for x in range(cleaned_mask.shape[1]):
                if cleaned_mask[y, x]:
                    top3_draw.point((x, y), fill=color)
    
    # 정제된 클러스터로 top_clusters 업데이트
    top_clusters = cleaned_top_clusters
    print("작은 연결 요소 제거 완료")
    
    # 각 클러스터의 컨투어 생성 및 특성 분류
    contour_masks = []
    
    if len(top_clusters) > 0:
        for idx, cluster in enumerate(top_clusters):
            # 클러스터의 마스크에서 컨투어 생성
            contour_mask = create_contour_mask(cluster['combined_mask'])
            
            # 컨투어의 depth 값과 중심점 계산
            avg_depth = calculate_avg_depth(contour_mask, depth_image)
            centroid_x, centroid_y = calculate_mask_centroid(contour_mask)
            
            # 컨투어 정보 저장
            contour_masks.append({
                'mask': contour_mask,
                'size': np.sum(contour_mask),
                'avg_depth': avg_depth,
                'centroid_x': centroid_x,
                'centroid_y': centroid_y,
                'original_cluster': cluster
            })
            
            print(f"클러스터 {idx+1} 컨투어 - 크기: {np.sum(contour_mask)}, 평균 depth: {avg_depth}, 중심 X: {centroid_x}, 중심 Y: {centroid_y}")
    
    # 특성에 따른 마스크 분류 및 딕셔너리 생성
    classified_masks = {}
    
    try:
        # 1. 가장 멀리 있는 마스크 찾기 - 키 값 2로 저장, 빨간색 표시
        if depth_image is not None and contour_masks:
            farthest_contour = max(contour_masks, key=lambda c: c['avg_depth'])
            classified_masks[2] = farthest_contour
            
            # 빨간색(255, 0, 0)으로 표시
            for y in range(farthest_contour['mask'].shape[0]):
                for x in range(farthest_contour['mask'].shape[1]):
                    if farthest_contour['mask'][y, x]:
                        classified_draw.point((x, y), fill=(255, 0, 0))
                        contour_draw.point((x, y), fill=(255, 0, 0))
            
            print(f"가장 멀리 있는 컨투어: depth {farthest_contour['avg_depth']}, 크기 {farthest_contour['size']}")
    except Exception as e:
        print(f"Error identifying farthest contour: {str(e)}")
    
    try:
        # 2. x좌표 값이 가장 작은 마스크 찾기 - 키 값 1로 저장, 초록색 표시
        if contour_masks:
            left_contour = min(contour_masks, key=lambda c: c['centroid_x'])
            classified_masks[1] = left_contour
            
            # 초록색(0, 255, 0)으로 표시
            for y in range(left_contour['mask'].shape[0]):
                for x in range(left_contour['mask'].shape[1]):
                    if left_contour['mask'][y, x]:
                        classified_draw.point((x, y), fill=(0, 255, 0))
                        contour_draw.point((x, y), fill=(0, 255, 0))
            
            print(f"가장 왼쪽에 있는 컨투어: 중심 X {left_contour['centroid_x']}, 크기 {left_contour['size']}")
    except Exception as e:
        print(f"Error identifying leftmost contour: {str(e)}")
    
    try:
        # 3. x좌표 값이 가장 큰 마스크 찾기 - 키 값 3으로 저장, 파란색 표시
        if len(contour_masks) >= 2:
            right_contour = max(contour_masks, key=lambda c: c['centroid_x'])
            classified_masks[3] = right_contour
            
            # 파란색(0, 0, 255)으로 표시
            for y in range(right_contour['mask'].shape[0]):
                for x in range(right_contour['mask'].shape[1]):
                    if right_contour['mask'][y, x]:
                        classified_draw.point((x, y), fill=(0, 0, 255))
                        contour_draw.point((x, y), fill=(0, 0, 255))
            
            print(f"가장 오른쪽에 있는 컨투어: 중심 X {right_contour['centroid_x']}, 크기 {right_contour['size']}")
    except Exception as e:
        print(f"Error identifying rightmost contour: {str(e)}")
    
    # 분류된 마스크 정보 출력
    print("분류된 컨투어:")
    for key, value in classified_masks.items():
        print(f"키 {key}: 크기 {value['size']}, 평균 depth {value['avg_depth']}, 중심 X {value['centroid_x']}")
    
    return classified_masks

def save_floor_images(model, image_index):
    """바닥 마스크 이미지 저장 함수"""
    image_name = f"test{image_index}.png"
    input_image_path = f"{ASSET_DIR}/{image_name}"
    output_image_path = f"{OUTPUT_DIR}/overlay_remove/combined_overlay_remove_{image_name}"
    mask_only_remove_path = f"{OUTPUT_DIR}/mask_only/floor/mask_only_remove_{image_name}"
    filtered_mask_remove_path = f"{OUTPUT_DIR}/mask_only/filtered/filtered_mask_remove_{image_name}"
    
    # 이미지 로드
    image_pil = Image.open(input_image_path).convert("RGB")
    
    # 여러 프롬프트 정의
    text_prompts = ["floor .", "ground .", "banner .", "wall ."]
    
    # 오버레이 이미지 생성
    overlay_image = image_pil.copy()
    overlay = Image.new("RGBA", overlay_image.size, (0, 255, 0, 0))  # 초록색 오버레이
    draw = ImageDraw.Draw(overlay)
    
    # 검정색 배경에 마스크만 표시할 이미지 생성
    mask_only_image = Image.new("RGB", image_pil.size, (0, 0, 0))
    mask_only_draw = ImageDraw.Draw(mask_only_image)
    
    # 필터링된 마스크를 위한 이미지
    filtered_mask_image = Image.new("RGB", image_pil.size, (0, 0, 0))
    filtered_mask_draw = ImageDraw.Draw(filtered_mask_image)
    
    # 각 프롬프트를 개별적으로 처리
    all_results = []
    for prompt in text_prompts:
        result = model.predict([image_pil], [prompt])
        all_results.extend(result)
        print(f"Processed prompt: {prompt}")
    
    # 모든 결과에 대한 마스크를 순회
    for result in all_results:
        for mask in result["masks"]:
            mask_np = np.array(mask, dtype=bool)
            
            # 마스크의 크기(픽셀 수) 계산
            mask_size = np.sum(mask_np)
            print(f"Floor mask size: {mask_size} pixels")
            
            # 임계값 적용 - 작은 마스크 필터링
            if mask_size < MASK_SIZE_THRESHOLD:
                print(f"Filtered out small floor mask with size {mask_size} pixels")
                continue  # 이 마스크는 건너뛰기
                
            # 마스크를 이미지에 그리기
            for y in range(mask_np.shape[0]):
                for x in range(mask_np.shape[1]):
                    if mask_np[y, x]:
                        draw.point((x, y), fill=(0, 255, 0, 128))  # 반투명 초록색
                        mask_only_draw.point((x, y), fill=(255, 255, 255))  # 흰색
                        filtered_mask_draw.point((x, y), fill=(255, 255, 255))  # 흰색
    
    # 이미지 합성 및 저장
    final_overlay_image = Image.alpha_composite(overlay_image.convert("RGBA"), overlay)
    final_overlay_image = final_overlay_image.convert("RGB")
    final_overlay_image.save(output_image_path)
    mask_only_image.save(mask_only_remove_path)
    filtered_mask_image.save(filtered_mask_remove_path)

def save_display_images(image_index, image_pil, images, classified_masks):
    """디스플레이 마스크 이미지 저장 함수"""
    image_name = f"test{image_index}.png"
    overlay_image, overlay, mask_only_image, filtered_mask_image, non_overlap_image, top3_image, classified_image, contour_image = images
    
    # 파일 경로 설정
    output_image_path = f"{OUTPUT_DIR}/overlay/combined_overlay_{image_name}"
    mask_only_path = f"{OUTPUT_DIR}/mask_only/display/mask_only_{image_name}"
    filtered_mask_path = f"{OUTPUT_DIR}/mask_only/filtered/filtered_mask_{image_name}"
    non_overlap_path = f"{OUTPUT_DIR}/mask_only/non_overlap/non_overlap_{image_name}"
    top3_path = f"{OUTPUT_DIR}/mask_only/top3_masks/top3_{image_name}"
    classified_path = f"{OUTPUT_DIR}/mask_only/classified_masks/classified_{image_name}"
    contour_path = f"{OUTPUT_DIR}/mask_only/contour_masks/contour_{image_name}"
    
    # JSON 파일에 마스크 정보 저장
    json_folder = f"{OUTPUT_DIR}/json"
    os.makedirs(json_folder, exist_ok=True)
    if classified_masks:
        save_mask_info_to_json(image_index, classified_masks, json_folder)
    else:
        print(f"Image {image_index}: No classified masks to save to JSON")
    
    # 이미지 저장
    final_overlay_image = Image.alpha_composite(overlay_image.convert("RGBA"), overlay)
    final_overlay_image = final_overlay_image.convert("RGB")
    final_overlay_image.save(output_image_path)
    mask_only_image.save(mask_only_path)
    filtered_mask_image.save(filtered_mask_path)
    non_overlap_image.save(non_overlap_path)
    top3_image.save(top3_path)
    classified_image.save(classified_path)
    contour_image.save(contour_path)

def process_single_image(model, image_index, floor_masks_by_image):
    """단일 이미지 처리 전체 과정 함수"""
    print(f"\n===== 이미지 {image_index} 처리 시작 =====")
    
    # 1. Depth 이미지 로드
    depth_image = load_depth_image(image_index)
    
    # 2. 바닥 마스크 가져오기
    floor_mask = floor_masks_by_image.get(image_index)
    if floor_mask is None:
        print(f"Warning: No floor mask for image {image_index}")
        floor_mask = np.zeros((0, 0), dtype=bool)  # 빈 마스크
    
    image_name = f"test{image_index}.png"
    input_image_path = f"{ASSET_DIR}/{image_name}"
    image_pil = Image.open(input_image_path).convert("RGB")
    
    # depth 이미지 크기가 다르면 원본 이미지 크기에 맞게 조정
    if depth_image is not None and depth_image.shape[::-1] != image_pil.size:
        print(f"Resizing depth image from {depth_image.shape} to {image_pil.size[::-1]}")
        depth_image = cv2.resize(depth_image, image_pil.size, interpolation=cv2.INTER_NEAREST)
    
    # 3. 디스플레이 마스크 처리
    non_overlap_masks, images = process_display_masks(model, image_index, floor_mask, image_pil, depth_image)
    
    # 4. 클러스터 처리
    top3_draw = ImageDraw.Draw(images[5])  # top3_image
    classified_draw = ImageDraw.Draw(images[6])  # classified_image
    contour_draw = ImageDraw.Draw(images[7])  # contour_image
    classified_masks = process_clusters(non_overlap_masks, depth_image, top3_draw, classified_draw, contour_draw)
    
    # 5. 결과 이미지 저장
    save_display_images(image_index, image_pil, images, classified_masks)
    
    print(f"===== 이미지 {image_index} 처리 완료 =====\n")
    return classified_masks

def main():
    """메인 처리 함수"""
    # 1. 초기 설정
    setup_directories()
    check_depth_folder()
    
    # 2. 모델 초기화
    model = LangSAM()
    
    # 3. 이미지 인덱스 설정
    image_indices = list(range(1, 3))  # test1.png, test2.png
    
    # 4. 첫 번째 단계: 바닥 마스크 생성
    floor_masks_by_image = {}
    for i in image_indices:
        floor_mask, _ = create_floor_mask(model, i)
        floor_masks_by_image[i] = floor_mask
    
    # 5. 두 번째 단계: 각 이미지에 대해 디스플레이 마스크 처리
    results = {}
    for i in image_indices:
        results[i] = process_single_image(model, i, floor_masks_by_image)
    
    # 6. 바닥 이미지 저장
    for i in image_indices:
        save_floor_images(model, i)
    
    print("모든 처리가 완료되었습니다!")
    return results

if __name__ == "__main__":
    main()
