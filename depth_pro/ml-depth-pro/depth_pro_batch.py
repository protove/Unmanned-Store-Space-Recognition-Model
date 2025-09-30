import os
import glob
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import sys

# depth_pro 모듈을 정확히 찾을 수 있도록 경로 추가
sys.path.append("/home/protove/vsc/CapstoneD/Capstone_Final/depth_pro/ml-depth-pro/src")

# 모듈 및 클래스 임포트
from depth_pro.depth_pro import create_model_and_transforms, DepthProConfig

def process_folder_with_depth_pro(input_folder, output_folder, device=None, file_types=None):
    """
    입력 폴더의 모든 이미지에 대해 Depth Pro 모델을 실행하고 결과를 출력 폴더에 저장합니다.
    
    Args:
        input_folder (str): 처리할 이미지가 있는 입력 폴더 경로
        output_folder (str): 깊이 맵을 저장할 출력 폴더 경로
        device (torch.device, optional): 사용할 장치 (None이면 자동 선택)
        file_types (list, optional): 처리할 이미지 파일 확장자 리스트
    """
    # 기본 파일 타입 설정
    if file_types is None:
        file_types = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # GPU 사용 가능 여부 확인 및 장치 설정
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")
    
    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)
    
    # 체크포인트 경로 설정
    checkpoint_path = "/home/protove/vsc/CapstoneD/Capstone_Final/depth_pro/ml-depth-pro/checkpoints/depth_pro.pt"
    if not os.path.exists(checkpoint_path):
        print(f"오류: 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        return
    
    # DepthPro 설정 및 모델 로드
    config = DepthProConfig(
        patch_encoder_preset="dinov2l16_384",
        image_encoder_preset="dinov2l16_384",
        checkpoint_uri=checkpoint_path,
        decoder_features=256,
        use_fov_head=True,
        fov_encoder_preset="dinov2l16_384",
    )
    
    try:
        print("모델 로드 중...")
        model, transform = create_model_and_transforms(
            config=config,
            device=device
        )
        model.eval()
        print("모델 로드 완료!")
    except Exception as e:
        print(f"모델 로드 실패: {str(e)}")
        return
    
    # 입력 폴더에서 이미지 파일 찾기
    image_files = []
    for file_type in file_types:
        image_files.extend(glob.glob(os.path.join(input_folder, f'*{file_type}')))
        image_files.extend(glob.glob(os.path.join(input_folder, f'*{file_type.upper()}')))
    
    if not image_files:
        print(f"입력 폴더 '{input_folder}'에 이미지 파일이 없습니다.")
        return
    
    print(f"총 {len(image_files)}개의 이미지 파일을 처리합니다.")
    
    # 각 이미지 파일 처리
    for image_path in tqdm(image_files, desc="깊이 맵 생성 중"):
        try:
            # 파일명 추출
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            
            # 이미지 로드 및 변환
            img = Image.open(image_path).convert('RGB')
            input_tensor = transform(img).unsqueeze(0).to(device)
            
            # 깊이 추론
            with torch.no_grad():
                result = model.infer(input_tensor)
                depth_map = result["depth"].cpu().numpy()
                focal_length = result["focallength_px"].item()
            
            # 깊이 맵 시각화 및 저장
            depth_output_path = os.path.join(output_folder, f"{name}_depth.png")
            
            # 깊이 맵 시각화
            plt.figure(figsize=(8, 8))
            plt.imshow(depth_map, cmap='viridis')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(depth_output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # 원시 깊이 데이터 저장 (NumPy 배열)
            raw_depth_output_path = os.path.join(output_folder, f"{name}_depth.npy")
            np.save(raw_depth_output_path, depth_map)
            
            # 메타데이터 저장
            meta_output_path = os.path.join(output_folder, f"{name}_meta.txt")
            with open(meta_output_path, 'w') as f:
                f.write(f"Focal Length (px): {focal_length}\n")
                f.write(f"Min Depth (m): {np.min(depth_map):.4f}\n")
                f.write(f"Max Depth (m): {np.max(depth_map):.4f}\n")
                f.write(f"Mean Depth (m): {np.mean(depth_map):.4f}\n")
            
        except Exception as e:
            print(f"이미지 '{filename}' 처리 중 오류 발생: {e}")
    
    print(f"처리 완료! 결과가 '{output_folder}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Depth Pro - 폴더 내 이미지의 깊이 맵 생성')
    parser.add_argument('--input', type=str, help='입력 이미지 폴더 경로')
    parser.add_argument('--output', type=str, help='출력 폴더 경로')
    parser.add_argument('--gpu', action='store_true', help='GPU 사용 (사용 가능한 경우)')
    parser.add_argument('--cpu', action='store_true', help='CPU 사용 강제')
    
    args = parser.parse_args()
    
    # 인자가 없는 경우 기본 경로 사용
    input_folder = args.input if args.input else "/home/protove/vsc/CapstoneD/Capstone_Final/lang-segment-anything/output/total_s1_fps3"
    output_folder = args.output if args.output else "/home/protove/vsc/CapstoneD/Capstone_Final/lang-segment-anything/output/total_s1_fps3/depth_maps"
    
    # 장치 설정
    device = None
    if args.cpu:
        device = torch.device('cpu')
    elif args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    
    # 폴더 처리 실행
    process_folder_with_depth_pro(input_folder, output_folder, device)