import cv2
import os
import glob
from tqdm import tqdm

def extract_frames_from_videos(input_folder, output_folder):
    """
    입력 폴더 내의 모든 동영상을 프레임별로 추출하여 출력 폴더에 저장합니다.
    
    Args:
        input_folder (str): 동영상 파일이 있는 입력 폴더 경로
        output_folder (str): 추출된 프레임을 저장할 출력 폴더 경로
    """
    # 출력 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 지원하는 비디오 확장자
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    
    # 입력 폴더에서 모든 비디오 파일 찾기
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_folder, f'*{ext}')))
        # 대문자 확장자도 검색
        video_files.extend(glob.glob(os.path.join(input_folder, f'*{ext.upper()}')))
    
    if not video_files:
        print(f"입력 폴더 '{input_folder}'에 비디오 파일이 없습니다.")
        return
    
    print(f"총 {len(video_files)}개의 비디오 파일을 처리합니다.")
    
    # 각 비디오 파일 처리
    for video_path in video_files:
        # 파일명 추출 (확장자 제외)
        video_filename = os.path.basename(video_path)
        video_name = os.path.splitext(video_filename)[0]
        
        print(f"\n처리 중: {video_filename}")
        
        # 비디오 열기
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  - 오류: {video_filename} 파일을 열 수 없습니다.")
            continue
        
        # 비디오 정보
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"  - 정보: {width}x{height}, {fps}fps, {total_frames}프레임")
        
        # 비디오용 하위 폴더 생성 (선택사항)
        # video_output_folder = os.path.join(output_folder, video_name)
        # os.makedirs(video_output_folder, exist_ok=True)
        
        # 프레임 추출 및 저장
        frame_count = 0
        
        # tqdm으로 진행 상태 표시
        with tqdm(total=total_frames, desc=f"  - 프레임 추출") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                # 프레임 저장 (0이 아닌 1부터 시작하는 번호 사용)
                frame_filename = f"{video_name}_frame{frame_count}.jpg"
                frame_path = os.path.join(output_folder, frame_filename)
                
                # 프레임 저장
                cv2.imwrite(frame_path, frame)
                
                # 진행 상태 업데이트
                pbar.update(1)
        
        # 자원 해제
        cap.release()
        
        print(f"  - 완료: {frame_count}개 프레임 추출됨")
    
    print("\n모든 비디오 처리 완료!")

# 폴더 경로 설정
input_folder = './CapstoneD/Capstone_Final/lang-segment-anything/assets/video/total_s1'   # 동영상 파일이 있는 폴더
output_folder = './CapstoneD/Capstone_Final/lang-segment-anything/output/total_s1_fps3' # 프레임이 저장될 폴더

# 프레임 추출 실행
extract_frames_from_videos(input_folder, output_folder)