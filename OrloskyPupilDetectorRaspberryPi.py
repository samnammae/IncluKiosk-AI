import cv2
import numpy as np
import time

'''
requirememts.txt

numpy
opencv-python
'''

# Crop the image to maintain a specific aspect ratio (width:height) before resizing.
def crop_to_aspect_ratio(image, width=640, height=480):
    current_height, current_width = image.shape[:2]
    desired_ratio = width / height
    current_ratio = current_width / current_height

    if current_ratio > desired_ratio:
        # Current image is too wide
        new_width = int(desired_ratio * current_height)
        offset = (current_width - new_width) // 2
        cropped_img = image[:, offset:offset + new_width]
    else:
        # Current image is too tall
        new_height = int(current_width / desired_ratio)
        offset = (current_height - new_height) // 2
        cropped_img = image[offset:offset + new_height, :]

    return cv2.resize(cropped_img, (width, height))
    
# Apply thresholding to an image
def apply_binary_threshold(image, darkestPixelValue, addedThreshold):
    threshold = darkestPixelValue + addedThreshold
    _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    return thresholded_image

# Finds a square area of dark pixels in the image
def get_darkest_area(image):
    ignoreBounds = 20
    imageSkipSize = 20
    searchArea = 20
    internalSkipSize = 10

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_sum = float('inf')
    darkest_point = None

    for y in range(ignoreBounds, gray.shape[0] - ignoreBounds, imageSkipSize):
        for x in range(ignoreBounds, gray.shape[1] - ignoreBounds, imageSkipSize):
            current_sum = 0
            num_pixels = 0
            for dy in range(0, searchArea, internalSkipSize):
                if y + dy >= gray.shape[0]:
                    break
                for dx in range(0, searchArea, internalSkipSize):
                    if x + dx >= gray.shape[1]:
                        break
                    current_sum += gray[y + dy][x + dx]
                    num_pixels += 1

            if current_sum < min_sum and num_pixels > 0:
                min_sum = current_sum
                darkest_point = (x + searchArea // 2, y + searchArea // 2)

    return darkest_point
    
# Mask all pixels outside a square defined by center and size
def mask_outside_square(image, center, size):
    x, y = center
    half_size = size // 2

    mask = np.zeros_like(image)
    top_left_x = max(0, x - half_size)
    top_left_y = max(0, y - half_size)
    bottom_right_x = min(image.shape[1], x + half_size)
    bottom_right_y = min(image.shape[0], y + half_size)
    mask[top_left_y:bottom_right_y, top_left_x:top_left_x + size] = 255
    return cv2.bitwise_and(image, mask)
    
# Returns the largest contour that is not extremely long or tall
def filter_contours_by_area_and_return_largest(contours, pixel_thresh, ratio_thresh):
    max_area = 0
    largest_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= pixel_thresh:
            x, y, w, h = cv2.boundingRect(contour)
            length_to_width_ratio = max(w / h, h / w)
            if length_to_width_ratio <= ratio_thresh:
                if area > max_area:
                    max_area = area
                    largest_contour = contour

    return [largest_contour] if largest_contour is not None else []

# Process frames for pupil detection
def process_frames(thresholded_image_medium, frame, gray_frame, darkest_point, debug_mode_on, render_cv_window):
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    dilated_image = cv2.dilate(thresholded_image_medium, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    reduced_contours = filter_contours_by_area_and_return_largest(contours, 1000, 3)

    final_rotated_rect = ((0, 0), (0, 0), 0)
    if len(reduced_contours) > 0 and len(reduced_contours[0]) > 5:
        ellipse = cv2.fitEllipse(reduced_contours[0])
        cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
        center_x, center_y = map(int, ellipse[0])
        cv2.circle(frame, (center_x, center_y), 3, (255, 255, 0), -1)
        final_rotated_rect = ellipse

    # Calculate FPS
    current_time = time.time()
    fps = int(1 / (current_time - process_frames.last_time)) if hasattr(process_frames, "last_time") else 0
    process_frames.last_time = current_time

    # Display FPS on the frame
    cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    frame = cv2.resize(frame, (320, 240))
    cv2.imshow("Frame with Ellipse", frame)

    if render_cv_window:
        cv2.imshow("Best Thresholded Image Contours on Frame", frame)

    return final_rotated_rect
    
# Process a single frame for pupil detection
def process_frame(frame):
    start_time = time.time()
    
    frame = crop_to_aspect_ratio(frame)
    #print(f"Time after crop_to_aspect_ratio: {time.time() - start_time:.6f} seconds")
    
    darkest_point = get_darkest_area(frame)
    #print(f"Time after get_darkest_area: {time.time() - start_time:.6f} seconds")
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #print(f"Time after cvtColor to gray: {time.time() - start_time:.6f} seconds")
    
    darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]
    thresholded_image_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, 15)
    #print(f"Time after apply_binary_threshold: {time.time() - start_time:.6f} seconds")
    
    thresholded_image_medium = mask_outside_square(thresholded_image_medium, darkest_point, 250)
    #print(f"Time after mask_outside_square: {time.time() - start_time:.6f} seconds")
    
    result = process_frames(thresholded_image_medium, frame, gray_frame, darkest_point, False, False)
    #print(f"Time after process_frames: {time.time() - start_time:.6f} seconds")
    
    return result

# Process video frames for pupil detection using OpenCV
def process_video_with_opencv():
    cap = cv2.VideoCapture(0)  # Open USB camera (adjust index if needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        process_frame(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 눈 ROI에서만 동공 추출을 수행하고, 결과 타원을 원본 프레임 좌표로 변환해 주는 래퍼
def process_eye_roi_and_draw(frame, eye_rect, show_debug=False):
    x, y, w, h = eye_rect
    roi = frame[y:y+h, x:x+w]

    # --- 당신 파이프라인의 핵심 단계들을 ROI에 적용 ---
    roi_proc = crop_to_aspect_ratio(roi)                    # 640x480로 정규화
    darkest_point = get_darkest_area(roi_proc)
    gray = cv2.cvtColor(roi_proc, cv2.COLOR_BGR2GRAY)
    darkest_val = gray[darkest_point[1], darkest_point[0]]
    th = apply_binary_threshold(gray, darkest_val, 15)
    th = mask_outside_square(th, darkest_point, 250)

    # process_frames는 타원을 그리면서 ellipse 반환함 (ellipse: ((cx,cy),(ma,mi),angle))
    # 여기서는 ROI에만 그리게 두고, 반환된 타원 중심을 원본 좌표로 변환해 사용.
    roi_proc_color = roi_proc.copy()
    ellipse = process_frames(th, roi_proc_color, gray, darkest_point, False, False)

    if ellipse[1] != (0, 0):  # 타원 검출 성공
        (cx, cy), (MA, ma), ang = ellipse
        # roi_proc는 640x480 고정. 원래 eye_rect(w,h)와의 비율로 좌표 되돌리기
        sx = w / 640.0
        sy = h / 480.0
        cx_abs = int(x + cx * sx)
        cy_abs = int(y + cy * sy)
        # 원본 프레임에 결과를 그려줌
        cv2.circle(frame, (cx_abs, cy_abs), 3, (0, 255, 255), -1)
        cv2.ellipse(frame, ((cx_abs, cy_abs), (MA*sx, ma*sy), ang), (0, 255, 0), 2)

    if show_debug:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)


def run_with_haar_eyes(camera_index=0, quit_key='q'):
    # Haar 분류기 로드 (안경 포함 눈 검출이 상대적으로 안정적)
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
    )

    cap = cv2.VideoCapture(camera_index)
    # 동일 처리량으로 맞추려면 해상도 낮추기
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Error: Failed to capture frame.")
            break

        gray_small = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 눈은 보통 얼굴 상단에 2개 → 너무 작은/이상치 제거 위해 minSize 조정
        eyes = eye_cascade.detectMultiScale(
            gray_small, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        # 여러 개면 넓이 큰 상위 1~2개만 사용
        eyes = sorted(eyes, key=lambda r: r[2]*r[3], reverse=True)[:2]

        # 각 눈마다 ROI 처리
        for (ex, ey, ew, eh) in eyes:
            # 약간의 패딩을 주면 동공이 잘리지 않음
            pad = int(min(ew, eh)*0.2)
            x = max(0, ex - pad); y = max(0, ey - pad)
            w = min(frame.shape[1]-x, ew + 2*pad)
            h = min(frame.shape[0]-y, eh + 2*pad)
            process_eye_roi_and_draw(frame, (x, y, w, h), show_debug=False)

        # FPS 오버레이(참고)
        cv2.imshow("Pupil on Eyes (ROI)", frame)
        if cv2.waitKey(1) & 0xFF == ord(quit_key):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_with_haar_eyes(camera_index=0)
    # process_video_with_opencv()