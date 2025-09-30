import cv2
import numpy as np

def detect_parallel_lines(image_path, angle_threshold=5, min_line_length=120, max_line_gap=20, hough_threshold=50, min_lines_threshold=4, max_distance_threshold=700):
    # 이미지 로드 및 그레이스케일 변환
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    depth_image = cv2.imread("./CapstoneD/CapstoneD/depth_pro/result/test12.jpg")
    
    # Canny 엣지 검출
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # 허프 변환을 사용한 직선 검출
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, hough_threshold, 
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    if lines is None:
        print("No lines detected.")
        return []

    # 검출된 직선의 기울기 계산
    angles = []
    centers = []
    line_segments = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        center = ((x1 + x2) / 2, (y1 + y2) / 2)  # 선분의 중심 좌표
        angles.append(angle)
        centers.append(center)
        line_segments.append((x1, y1, x2, y2))

    # 특정 기울기 + 거리 기준으로 그룹화
    grouped_lines = []
    used_lines = set()

    for i in range(len(angles)):
        if i in used_lines:
            continue
        group = [line_segments[i]]
        used_lines.add(i)
        
        for j in range(i + 1, len(angles)):
            if j in used_lines:
                continue

            # 각도 차이
            angle_diff = abs(angles[i] - angles[j])
            angle_diff = min(angle_diff, 180 - angle_diff)  # 180도 넘어가는 케이스 처리

            # 중심점 거리
            dist = np.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])

            if angle_diff < angle_threshold and dist <= max_distance_threshold:
                group.append(line_segments[j])
                used_lines.add(j)

        # 임계값 이상의 직선 개수를 가진 그룹만 추가
        if len(group) >= min_lines_threshold:
            grouped_lines.append(group)

    
    # 영역 좌표 추출 (경계 박스) 및 색칠
    bounding_boxes = []
    for group in grouped_lines:
        x_coords = [x for line in group for x in [line[0], line[2]]]
        y_coords = [y for line in group for y in [line[1], line[3]]]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        bounding_box = (x_min, y_min, x_max, y_max)
        bounding_boxes.append(bounding_box)
        
        # # 색칠하여 표시
        # cv2.rectangle(depth_image, (x_min, y_min-100), (x_max+10, y_max+10), (255, 255, 255), 2)
        # cv2.putText(depth_image, f"Region ({len(group)} lines)", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # cv2.rectangle(image, (x_min, y_min-100), (x_max+10, y_max+10), (255, 255, 255), 2)
        # cv2.putText(image, f"Region ({len(group)} lines)", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    processed_rois = process_overlapping_rois(depth_image, bounding_boxes)
    sel_black_rois = []
    for i in range(len(processed_rois)):
        x_min, y_min, x_max, y_max = processed_rois[i]
        sel_black_rois.append((x_min, y_min, (x_max-x_min), (y_max-y_min)))
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, f"Region ({len(group)} lines)", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    

    # 결과 이미지 저장 및 표시
    output_path = "./CapstoneD/CapstoneD/depth_pro/final_depth_image/test17.jpg"
    cv2.imwrite(output_path, image)

    
    # cv2.imwrite("../final_depth_image/test15.jpg", image)
    # cv2.imshow("Detected Parallel Lines", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(f"Bounding Boxes: {bounding_boxes}")
    return sel_black_rois

def process_overlapping_rois(depth_map, rois, threshold=40):
    """
    깊이맵에서 겹치는 ROI 영역을 분석하고 특정 조건에 따라 병합 또는 제거하는 함수

    Parameters:
        depth_map (numpy.ndarray): 입력 깊이맵 이미지 (RGB)
        rois (list of tuples): ROI 영역 리스트 [(x1, y1, x2, y2), ...]
        threshold (int): 색상 평균 차이 임계값

    Returns:
        list of tuples: 처리된 ROI 리스트
    """
    updated_rois = rois.copy()
    
    for i in range(len(rois)):
        for j in range(i + 1, len(rois)):
            if updated_rois[i] is None or updated_rois[j] is None:
                continue
            
            x1_1, y1_1, x2_1, y2_1 = updated_rois[i]
            x1_2, y1_2, x2_2, y2_2 = updated_rois[j]
            
            # ROI 겹치는 부분 확인
            x_overlap_start = max(x1_1, x1_2)
            y_overlap_start = max(y1_1, y1_2)
            x_overlap_end = min(x2_1, x2_2)
            y_overlap_end = min(y2_1, y2_2)
            
            x_overlap = max(0, x_overlap_end - x_overlap_start)
            y_overlap = max(0, y_overlap_end - y_overlap_start)

            semi_depth_image = cv2.cvtColor(depth_map, cv2.COLOR_BGR2RGB)
            gray_depth = cv2.cvtColor(semi_depth_image, cv2.COLOR_RGB2GRAY)
            
            if x_overlap > 0 and y_overlap > 0:
                roi1 = gray_depth[y1_1:y2_1, x1_1:x2_1]  
                roi2 = gray_depth[y1_2:y2_2, x1_2:x2_2]  
                
                mean1 = np.mean(roi1)  
                mean2 = np.mean(roi2)  
                
                # 색상 평균 차이 계산
                color_diff = abs(mean1 - mean2)
                
                if color_diff <= threshold:
                    # 두 ROI를 병합
                    new_roi = (min(x1_1, x1_2), min(y1_1, y1_2), max(x2_1, x2_2), max(y2_1, y2_2))
                    updated_rois[i] = new_roi
                    updated_rois[j] = None  # 삭제 표시
                else:
                    # 파란색에 가까운 ROI에서 겹치는 부분 제거
                    if mean1 > mean2:
                        if x_overlap_start == x1_2:
                            updated_rois[j] = (x_overlap_end, y1_2, x2_2, y2_2)
                        if x_overlap_end == x2_2:
                            updated_rois[j] = (x1_2, y1_2, x_overlap_start, y2_2)
                        
                           
                    else:
                        if x_overlap_start == x1_1:
                            updated_rois[i] = (x_overlap_end, y1_1, x2_1, y2_1)
                        if x_overlap_end == x2_1:
                            updated_rois[i] = (x1_1, y1_1, x_overlap_start, y2_1)
    
    
    # None 제거 후 ROI 리스트 정리
    updated_rois = [roi for roi in updated_rois if roi is not None]
    
    return updated_rois

def sel_black(bboxes, image_path):
    image = cv2.imread(image_path)
    # 임계값 (검은색 픽셀 비율)
    black_pixel_threshold = 0.4  # 50%
    sel_black_bbox = []

    for idx, (x, y, w, h) in enumerate(bboxes):
        # bounding box 영역 잘라내기
        roi = image[y:y+h, x:x+w]
        if roi is None or roi.size == 0:
            continue
        result = image.copy()
        # ROI를 그레이스케일로 변환
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"./CapstoneD/CapstoneD/SAM2/semi_gray_sel_black/{idx}_gray.jpg", gray)

        # 검정색 픽셀 개수 세기
        black_pixels = np.sum(gray <= 0)  # 0: 검은색 픽셀로 간주
        total_pixels = gray.size
        black_ratio = black_pixels / total_pixels if total_pixels > 0 else 0

        if black_ratio >= black_pixel_threshold:
            sel_black_bbox.append((x, y, w, h))
            print(f"✅ Bounding Box {idx} (x={x}, y={y}, w={w}, h={h}) -> black_ratio={black_ratio:.2f} -> 복사됨")
        else:
            print(f"❌ Bounding Box {idx} (x={x}, y={y}, w={w}, h={h}) -> black_ratio={black_ratio:.2f} -> 무시됨")
    
    # 선택된 bbox들만 결과에 그리기
    for (x, y, w, h) in sel_black_bbox:
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        cv2.rectangle(result, top_left, bottom_right, (0, 255, 0), thickness=2)
    cv2.imwrite("./CapstoneD/CapstoneD/SAM2/sel_black/test17.jpg", result)
    return

# 사용 예제
image_path = "./CapstoneD/CapstoneD/depth_pro/hughline/black_test17_hughline.jpg"  # 분석할 이미지 경로
bounding_boxes = detect_parallel_lines(image_path)
sel_black_image = "./CapstoneD/CapstoneD/SAM2/filter_small_mask/test17_filtered_mask.png"
sel_black(bounding_boxes, sel_black_image)
