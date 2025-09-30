import os
import cv2
from PIL import Image
from tqdm import tqdm

def resize_images(input_folder, output_folder, target_size, keep_aspect_ratio=True, file_types=None):
    """
    입력 폴더의 모든 이미지를 원하는 크기로 조정하여 출력 폴더에 저장합니다.
    
    Args:
        input_folder (str): 입력 이미지가 있는 폴더 경로
        output_folder (str): 크기 조정된 이미지를 저장할 폴더 경로
        target_size (tuple): 목표 이미지 크기 (width, height)
        keep_aspect_ratio (bool): 가로세로 비율을 유지할지 여부
        file_types (list): 처리할 이미지 파일 확장자 목록 (기본값: jpg, jpeg, png, bmp)
    """
    # 지원되는 이미지 파일 확장자
    if file_types is None:
        file_types = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    # 출력 폴더가 없으면 생성
    os.makedirs(output_folder, exist_ok=True)
    
    # 입력 폴더의 모든 파일 목록 가져오기
    all_files = os.listdir(input_folder)
    
    # 이미지 파일만 필터링
    image_files = [f for f in all_files if os.path.splitext(f.lower())[1] in file_types]
    
    if not image_files:
        print(f"입력 폴더 '{input_folder}'에 이미지 파일이 없습니다.")
        return
    
    print(f"총 {len(image_files)}개의 이미지 파일을 처리합니다.")
    
    # 모든 이미지 처리
    for filename in tqdm(image_files):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        try:
            # PIL을 사용하여 이미지 로드
            img = Image.open(input_path)
            
            # 원본 크기 기록
            original_width, original_height = img.size
            target_width, target_height = target_size
            
            if keep_aspect_ratio:
                # 가로세로 비율 유지 (원본 비율에 맞추어 크기 조정)
                ratio = min(target_width / original_width, target_height / original_height)
                new_width = int(original_width * ratio)
                new_height = int(original_height * ratio)
                
                # 이미지 리사이즈
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # 목표 크기의 검은색 배경 생성
                new_img = Image.new("RGB", target_size, (0, 0, 0))
                
                # 가운데 정렬
                paste_x = (target_width - new_width) // 2
                paste_y = (target_height - new_height) // 2
                
                # 리사이즈된 이미지를 중앙에 붙여넣기
                new_img.paste(resized_img, (paste_x, paste_y))
                
                # 저장
                new_img.save(output_path)
                
            else:
                # 비율 무시하고 지정된 크기로 조정
                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                resized_img.save(output_path)
            
            print(f"처리 완료: {filename} ({original_width}x{original_height} → {target_size[0]}x{target_size[1]})")
            
        except Exception as e:
            print(f"이미지 '{filename}' 처리 중 오류 발생: {e}")
    
    print(f"모든 이미지 크기 조정 완료. 결과는 '{output_folder}'에 저장되었습니다.")

# 사용 예시
if __name__ == "__main__":
    input_folder = "/home/protove/vsc/CapstoneD/Capstone_Final/lang-segment-anything/assets/space_data"  # 입력 이미지 폴더
    output_folder = "/home/protove/vsc/CapstoneD/Capstone_Final/lang-segment-anything/assets/space_data/resized_images"  # 출력 폴더
    
    # 원하는 크기 설정 (너비, 높이)
    target_size = (3840, 2160)
    
    # 가로세로 비율 유지 여부 (True: 유지, False: 무시)
    keep_aspect_ratio = False
    
    # 이미지 크기 조정 실행
    resize_images(input_folder, output_folder, target_size, keep_aspect_ratio)