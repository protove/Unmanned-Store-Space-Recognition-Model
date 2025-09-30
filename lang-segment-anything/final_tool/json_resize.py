import json
import os
from tqdm import tqdm

def convert_coordinates(input_json_path, output_json_path, original_size, new_size):
    """
    JSON 파일 내의 좌표값을 원본 해상도에서 새 해상도로 비례하여 변환
    
    Args:
        input_json_path (str): 입력 JSON 파일 경로
        output_json_path (str): 출력 JSON 파일 경로
        original_size (tuple): 원본 이미지 크기 (width, height)
        new_size (tuple): 새 이미지 크기 (width, height)
    """
    # 원본 크기와 새 크기에서의 비율 계산
    width_ratio = new_size[0] / original_size[0]
    height_ratio = new_size[1] / original_size[1]
    
    print(f"너비 비율: {width_ratio:.4f}, 높이 비율: {height_ratio:.4f}")
    
    try:
        # JSON 파일 로드
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 마스크별 좌표값 변환
        if 'masks' in data:
            for mask in data['masks']:
                # 새 좌표값 계산 (original 값 저장하지 않음)
                mask['min_x'] = int(mask['min_x'] * width_ratio)
                mask['max_x'] = int(mask['max_x'] * width_ratio)
                mask['min_y'] = int(mask['min_y'] * height_ratio)
                mask['max_y'] = int(mask['max_y'] * height_ratio)
                
                # 너비와 높이 업데이트
                if 'width' in mask:
                    mask['width'] = mask['max_x'] - mask['min_x']
                if 'height' in mask:
                    mask['height'] = mask['max_y'] - mask['min_y']
        
        # 추가 메타데이터 기록
        data['coordinate_conversion'] = {
            'original_width': original_size[0],
            'original_height': original_size[1],
            'new_width': new_size[0],
            'new_height': new_size[1],
            'width_ratio': width_ratio,
            'height_ratio': height_ratio
        }
        
        # 변환된 JSON 저장
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        print(f"좌표값 변환 완료: {output_json_path}")
        
    except Exception as e:
        print(f"좌표값 변환 중 오류 발생: {e}")

def batch_convert_coordinates(input_dir, output_dir, original_size, new_size, file_pattern="*.json"):
    """
    디렉토리 내의 모든 JSON 파일의 좌표값을 변환
    
    Args:
        input_dir (str): 입력 JSON 파일들이 있는 디렉토리
        output_dir (str): 출력 JSON 파일들을 저장할 디렉토리
        original_size (tuple): 원본 이미지 크기 (width, height)
        new_size (tuple): 새 이미지 크기 (width, height)
        file_pattern (str): 처리할 JSON 파일 패턴
    """
    import glob
    
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 입력 디렉토리에서 JSON 파일들 찾기
    json_files = glob.glob(os.path.join(input_dir, file_pattern))
    
    if not json_files:
        print(f"입력 디렉토리 '{input_dir}'에 JSON 파일이 없습니다.")
        return
    
    print(f"총 {len(json_files)}개의 JSON 파일 변환 중...")
    
    # 각 JSON 파일 처리
    for json_path in tqdm(json_files):
        filename = os.path.basename(json_path)
        output_path = os.path.join(output_dir, filename)
        convert_coordinates(json_path, output_path, original_size, new_size)
    
    print("모든 JSON 파일 변환 완료!")

# 사용 예시
if __name__ == "__main__":
    # 원본 이미지 크기 (너비, 높이)
    original_size = (1789, 1006)
    
    # 새 이미지 크기 (너비, 높이)
    new_size = (3840, 2160)
    
    # 1. 단일 파일 변환
    input_json = "/home/protove/vsc/CapstoneD/Capstone_Final/lang-segment-anything/output/final/json/mask_info_test1.json"
    output_json = "/home/protove/vsc/CapstoneD/Capstone_Final/lang-segment-anything/output/final/json/mask_info_test1_4k.json"
    
    convert_coordinates(input_json, output_json, original_size, new_size)
    
    # 2. 폴더 내 모든 JSON 파일 변환 (필요 시 주석 해제)
    # input_dir = "/home/protove/vsc/CapstoneD/Capstone_Final/lang-segment-anything/output/final/json"
    # output_dir = "/home/protove/vsc/CapstoneD/Capstone_Final/lang-segment-anything/output/final/json_4k"
    # 
    # batch_convert_coordinates(input_dir, output_dir, original_size, new_size, "mask_info_*.json")