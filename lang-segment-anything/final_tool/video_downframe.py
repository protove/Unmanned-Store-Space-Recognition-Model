import cv2
import os
import glob

# 입력 및 출력 폴더 설정
input_folder = './CapstoneD/Capstone_Final/lang-segment-anything/assets/video/total_s1'  # 입력 폴더 경로
output_folder = './CapstoneD/Capstone_Final/lang-segment-anything/assets/video/total_s1/fps3'  # 출력 폴더 경로

# 출력 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 지원하는 비디오 확장자
video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']

# 입력 폴더에서 모든 비디오 파일 찾기
video_files = []
for ext in video_extensions:
    video_files.extend(glob.glob(os.path.join(input_folder, f'*{ext}')))

print(f"총 {len(video_files)}개의 비디오 파일을 찾았습니다.")

# 각 비디오 파일 처리
for i, input_path in enumerate(video_files):
    # 파일명 추출
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_folder, f"{name}_3fps{ext}")
    
    print(f"[{i+1}/{len(video_files)}] 처리 중: {filename}")
    
    # 비디오 열기
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"  - 오류: {filename} 파일을 열 수 없습니다.")
        continue
    
    # 원본 비디오 정보
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  - 원본: {width}x{height}, {fps}fps, {total_frames}프레임")
    
    # 몇 프레임마다 하나씩 선택할지 계산
    frame_interval = max(1, int(fps / 3))  # 최소 1 이상
    expected_frames = total_frames // frame_interval
    
    # 비디오 라이터 설정 (3fps로 저장)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 3, (width, height))
    
    # 프레임 추출 및 저장
    frame_count = 0
    saved_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # frame_interval 프레임마다 하나씩 저장
        if frame_count % frame_interval == 0:
            out.write(frame)
            saved_frames += 1
        
        frame_count += 1
        
        # 진행상황 표시 (10% 간격)
        if frame_count % (total_frames // 10) == 0:
            progress = int((frame_count / total_frames) * 100)
            print(f"  - 진행률: {progress}%")
    
    # 자원 해제
    cap.release()
    out.release()
    
    print(f"  - 완료: {output_path} (3fps, {saved_frames}프레임)")

print("모든 비디오 변환 완료!")
