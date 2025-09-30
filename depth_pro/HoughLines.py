import cv2
import numpy as np
import matplotlib.pyplot as plt

for i in range(14, 19):
    # 이미지 불러오기 (그레이스케일 변환)
    image = cv2.imread(f'./CapstoneD/Capstone_Final/depth_pro/edges/test{i}_edges.jpg', cv2.IMREAD_GRAYSCALE)

    # 흰색과 검은색이 섞여 있는 이미지 불러오기
    mask_image = cv2.imread(f'./CapstoneD/Capstone_Final/lang-segment-anything/output/mask_only/combined_masks/combined_test{i}.png', cv2.IMREAD_GRAYSCALE)

    # 캐니 엣지 검출
    edges = cv2.Canny(image, 50, 150)

    # 확장된 허프 변환 (직선 검출)
    length_threshold = 150
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=length_threshold, maxLineGap=10)

    # 원본 이미지를 컬러로 변환
    image_colored = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    white_lines_image = np.zeros_like(image_colored)  # 흰색 포함된 선만 그릴 이미지
    lang_white_lines_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)  # langsam+depth 이미지
    # 검출된 선 바로 그림
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 선이 흰색 픽셀을 지나가는지 확인
            line_mask = np.zeros_like(mask_image, dtype=np.uint8)
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 1)  # 선을 마스크에 그림
            if np.any(cv2.bitwise_and(mask_image, line_mask) == 255):  # 흰색 픽셀과 교차 여부 확인
                cv2.line(image_colored, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.line(white_lines_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.line(lang_white_lines_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 결과 저장
    cv2.imwrite(f"./CapstoneD/Capstone_Final/depth_pro/hughline/test{i}_hughline.jpg", image_colored)
    cv2.imwrite(f"./CapstoneD/Capstone_Final/depth_pro/langsam_depth/white_test{i}_hughline.jpg", white_lines_image)
    cv2.imwrite(f"./CapstoneD/Capstone_Final/depth_pro/langsam_depth/lang_white_test{i}_hughline.jpg", lang_white_lines_image)

    # 결과 출력
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
    plt.subplot(1, 4, 2), plt.imshow(image_colored), plt.title(f'All Lines Longer Than {length_threshold} px')
    plt.subplot(1, 4, 3), plt.imshow(mask_image), plt.title('Mask Image')
    plt.subplot(1, 4, 4), plt.imshow(white_lines_image), plt.title('White Pixel Lines Only')
    plt.show()
