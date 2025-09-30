import cv2
import numpy as np

def detect_parallel_lines(image_path, angle_threshold=5, min_line_length=80, max_line_gap=20, hough_threshold=50, min_lines_threshold=8):
    # 이미지 로드 및 그레이스케일 변환
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    depth_image = cv2.imread("result/test18.jpg")
    
    # Canny 엣지 검출
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # 허프 변환을 사용한 직선 검출
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, hough_threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    if lines is None:
        print("No lines detected.")
        return []
    
    # 검출된 직선의 기울기 계산
    angles = []
    line_segments = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)
        line_segments.append((x1, y1, x2, y2))
    
    # 특정한 기울기를 갖는 직선 그룹화
    grouped_lines = []
    used_lines = set()
    
    for i in range(len(angles)):
        if i in used_lines:
            continue
        group = [line_segments[i]]
        used_lines.add(i)
        
        for j in range(i + 1, len(angles)):
            if j not in used_lines and abs(angles[i] - angles[j]) < angle_threshold:
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
        
        # # 색칠하여 표시시
        # cv2.rectangle(depth_image, (x_min, y_min-100), (x_max+10, y_max+10), (255, 255, 255), 2)
        # cv2.putText(depth_image, f"Region ({len(group)} lines)", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(image, (x_min, y_min-100), (x_max+10, y_max+10), (255, 255, 255), 2)
        cv2.putText(image, f"Region ({len(group)} lines)", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    processed_rois = process_overlapping_rois(depth_image, bounding_boxes)
    for i in range(len(processed_rois)):
        x_min, y_min, x_max, y_max = processed_rois[i]
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, f"Region ({len(group)} lines)", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    

    # 결과 이미지 저장 및 표시
    output_path = "../final_depth_image/test18.jpg"
    cv2.imwrite(output_path, image)

    
    # cv2.imwrite("../final_depth_image/test15.jpg", image)
    cv2.imshow("Detected Parallel Lines", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"Bounding Boxes: {bounding_boxes}")
    return bounding_boxes

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

# 사용 예제
image_path = "../hughline/black_test18_hughline.jpg"  # 분석할 이미지 경로
bounding_boxes = detect_parallel_lines(image_path)
