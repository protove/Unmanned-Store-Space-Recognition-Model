import cv2
import numpy as np

def highlight_depth_edges(depth_map_path, output_path, image_num, threshold=10, highlight_color=(0, 0, 255)):
    # 깊이맵 로드 (흑백 이미지로 읽음)
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)

    if depth_map is None:
        print("Error: Could not load the depth map.")
        return

    # 입력된 이미지와 동일한 크기의 검은색 이미지 생성
    black_image = np.zeros_like(depth_map)  # 모든 픽셀 값이 0인 이미지
    cv2.imwrite("./CapstoneD/Capstone_Final/depth_pro/edges/black_image.jpg", black_image)
    black_image = cv2.imread("./CapstoneD/Capstone_Final/depth_pro/edges/black_image.jpg")



    # Sobel 필터 적용 (X, Y 방향의 기울기 계산)
    grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=5)

    # 경계 강도 계산 (그래디언트 크기)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))  # 0~255로 정규화

    # 임계값 이상인 부분을 경계선으로 간주
    edges = magnitude > threshold

    # 컬러 이미지로 변환 (경계를 강조하기 위해)
    depth_map_colored = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)

    # 경계선 부분에 색을 칠함 (빨간색)
    depth_map_colored[edges] = highlight_color
    black_image[edges] = highlight_color

    # 결과 저장
    cv2.imwrite(output_path, depth_map_colored)
    cv2.imwrite(f"./CapstoneD/Capstone_Final/depth_pro/edges/test{image_num}_edges.jpg", black_image)

    print(f"Output saved to {output_path}")

# 실행 예제
for i in range(14, 19):
    highlight_depth_edges(f"./CapstoneD/Capstone_Final/depth_pro/result/test{i}.jpg", f"./CapstoneD/Capstone_Final/depth_pro/highlight/test{i}_highlighted_depth_map.png", i)
