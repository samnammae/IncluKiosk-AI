#!/usr/bin/env python3
"""
라즈베리파이용 경량 아이트래킹 시스템
TensorFlow/TFLite 의존성 완전 제거 버전
순수 OpenCV만으로 동작
"""

import cv2
import numpy as np
import pyautogui
import time
from collections import deque

# ================== 설정 부분 ==================
# pyautogui 설정
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
screen_w, screen_h = pyautogui.size()

# 성능 관련 파라미터
FRAME_SKIP = 10  # 10프레임마다 1번 처리
INPUT_WIDTH = 320  # 라즈베리파이에서 적절한 해상도
INPUT_HEIGHT = 240
EYE_ROI_SIZE = 60  # 눈 영역 크기

# 추적 관련 파라미터
MAX_TRACKING_LOSS = 10  # 추적 실패 허용 횟수
SMOOTHING_WINDOW = 5  # 스무딩 윈도우 크기
EAR_THRESHOLD = 0.22  # 눈 감김 임계값
BLINK_FRAMES = 2
DOUBLE_BLINK_TIME = 0.8

print("=" * 50)
print("라즈베리파이 아이트래킹 시스템 v2.0")
print("TensorFlow 없이 순수 OpenCV로 동작")
print("=" * 50)

# ================== 순수 OpenCV 기반 추적기 ==================
class PureOpenCVEyeTracker:
    def __init__(self):
        """OpenCV만 사용하는 경량 눈 추적기"""
        print("초기화 중...")
        
        # Haar Cascade 로드
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # 검증
        if self.face_cascade.empty():
            print("❌ 얼굴 검출기 로드 실패!")
            print("다음 명령으로 설치: sudo apt-get install opencv-data")
            exit(1)
        
        print("✅ Haar Cascade 로드 완료")
        
        # ROI 추적 변수
        self.face_roi = None
        self.left_eye_roi = None
        self.right_eye_roi = None
        self.tracking_confidence = 0
        self.roi_padding = 30
        
        # Kalman 필터 (움직임 예측용)
        self.kalman = self._init_kalman_filter()
        
        # 스무딩용 큐
        self.position_queue = deque(maxlen=SMOOTHING_WINDOW)
        
        # 동공 검출용 파라미터
        self.pupil_threshold = 30
        
        print("✅ 초기화 완료")
    
    def _init_kalman_filter(self):
        """칼만 필터 초기화 - 움직임 예측용"""
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                           [0, 1, 0, 1],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]], np.float32)
        kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        return kalman
    
    def detect_face_and_eyes(self, gray_frame):
        """전체 프레임에서 얼굴과 눈 검출"""
        # 성능 향상을 위한 다운스케일
        scale_factor = 0.5
        small_frame = cv2.resize(gray_frame, None, fx=scale_factor, fy=scale_factor)
        
        faces = self.face_cascade.detectMultiScale(
            small_frame, 
            scaleFactor=1.1, 
            minNeighbors=4, 
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # 스케일 복원
            x, y, w, h = faces[0]
            x, y, w, h = int(x/scale_factor), int(y/scale_factor), int(w/scale_factor), int(h/scale_factor)
            
            # ROI 업데이트
            self.face_roi = self._expand_roi(x, y, w, h, gray_frame.shape, self.roi_padding)
            
            # 얼굴 영역 내에서 눈 검출
            face_region = gray_frame[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(
                face_region, 
                scaleFactor=1.05, 
                minNeighbors=3,
                minSize=(15, 15)
            )
            
            if len(eyes) >= 2:
                # 두 눈 위치 저장
                eyes = sorted(eyes, key=lambda e: e[0])[:2]
                
                for i, (ex, ey, ew, eh) in enumerate(eyes):
                    eye_x, eye_y = x + ex, y + ey
                    
                    if i == 0:
                        self.left_eye_roi = (eye_x, eye_y, ew, eh)
                    else:
                        self.right_eye_roi = (eye_x, eye_y, ew, eh)
                
                self.tracking_confidence = MAX_TRACKING_LOSS
                return True
        
        self.tracking_confidence -= 1
        return False
    
    def track_in_roi(self, gray_frame):
        """ROI 영역 내에서만 빠르게 추적"""
        if self.face_roi is None or self.tracking_confidence <= 0:
            # ROI가 없거나 신뢰도가 낮으면 전체 검출
            return self.detect_face_and_eyes(gray_frame)
        
        x, y, w, h = self.face_roi
        
        # 경계 체크
        x = max(0, x)
        y = max(0, y)
        w = min(gray_frame.shape[1] - x, w)
        h = min(gray_frame.shape[0] - y, h)
        
        roi = gray_frame[y:y+h, x:x+w]
        
        # ROI 내에서 빠른 얼굴 검출
        faces = self.face_cascade.detectMultiScale(
            roi, 
            scaleFactor=1.2, 
            minNeighbors=2,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            fx, fy, fw, fh = faces[0]
            self.face_roi = self._expand_roi(x+fx, y+fy, fw, fh, gray_frame.shape, 20)
            self.tracking_confidence = min(self.tracking_confidence + 1, MAX_TRACKING_LOSS)
            return True
        
        self.tracking_confidence -= 1
        return False
    
    def _expand_roi(self, x, y, w, h, frame_shape, padding):
        """ROI 영역 확장"""
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame_shape[1] - x, w + 2*padding)
        h = min(frame_shape[0] - y, h + 2*padding)
        return (x, y, w, h)
    
    def get_pupil_position(self, gray_frame, eye_roi):
        """동공 위치 검출 - 순수 OpenCV"""
        if eye_roi is None:
            return None
        
        x, y, w, h = eye_roi
        eye_region = gray_frame[y:y+h, x:x+w]
        
        # 전처리 강화
        eye_region = cv2.equalizeHist(eye_region)
        eye_region = cv2.GaussianBlur(eye_region, (5, 5), 0)
        
        # 적응형 임계값으로 동공 검출
        _, threshold = cv2.threshold(
            eye_region, 
            self.pupil_threshold, 
            255, 
            cv2.THRESH_BINARY_INV
        )
        
        # 모폴로지 연산으로 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(
            threshold, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            # 크기와 원형도 기준으로 동공 선택
            valid_contours = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 10 and area < (w*h*0.3):  # 너무 작거나 큰 것 제외
                    # 원형도 체크
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.3:  # 어느 정도 원형인 것만
                            valid_contours.append(cnt)
            
            if valid_contours:
                # 가장 적절한 컨투어 선택
                best_contour = max(valid_contours, key=cv2.contourArea)
                M = cv2.moments(best_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (x + cx, y + cy)
        
        # 검출 실패시 중앙 반환
        return (x + w//2, y + h//2)
    
    def calculate_ear(self, eye_roi, gray_frame):
        """간단한 EAR 계산"""
        if eye_roi is None:
            return 1.0
        
        x, y, w, h = eye_roi
        aspect_ratio = h / max(w, 1)
        return aspect_ratio
    
    def adjust_threshold(self, brightness_level):
        """조명에 따라 동공 검출 임계값 자동 조정"""
        # 어두우면 임계값 낮춤, 밝으면 높임
        self.pupil_threshold = int(np.interp(brightness_level, [0, 255], [20, 50]))

# ================== 메인 실행 부분 ==================
def main():
    print("\n시스템 시작...")
    
    # 초기화
    tracker = PureOpenCVEyeTracker()
    
    # 카메라 설정
    print("카메라 초기화 중...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다!")
        return
    
    # 카메라 최적화 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print(f"✅ 카메라 준비 완료: {INPUT_WIDTH}x{INPUT_HEIGHT}")
    
    # 상태 변수
    frame_count = 0
    prev_x, prev_y = screen_w//2, screen_h//2
    blink_counter = 0
    total_blinks = 0
    blink_times = []
    last_click_time = 0
    
    # FPS 측정
    fps_start = time.time()
    fps_counter = 0
    current_fps = 0
    
    print(f"\n📊 화면 해상도: {screen_w}x{screen_h}")
    print("🎯 빠르게 두 번 깜빡이면 클릭됩니다!")
    print("⌨️  'q' 또는 ESC를 눌러 종료\n")
    print("-" * 50)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        
        # 프레임 스킵
        if frame_count % FRAME_SKIP != 0:
            # 스킵된 프레임도 화면에는 표시
            cv2.imshow("Eye Tracking", frame)
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break
            continue
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 조명 레벨 체크 (자동 임계값 조정용)
        mean_brightness = np.mean(gray)
        tracker.adjust_threshold(mean_brightness)
        
        # 추적 수행
        if frame_count == FRAME_SKIP:
            tracker.detect_face_and_eyes(gray)
        else:
            tracker.track_in_roi(gray)
        
        # 동공 위치 추정 및 마우스 제어
        if tracker.left_eye_roi:
            pupil_pos = tracker.get_pupil_position(gray, tracker.left_eye_roi)
            
            if pupil_pos:
                px, py = pupil_pos
                
                # Kalman 필터 적용
                measurement = np.array([[np.float32(px)], [np.float32(py)]])
                tracker.kalman.correct(measurement)
                prediction = tracker.kalman.predict()
                
                pred_x, pred_y = int(prediction[0]), int(prediction[1])
                tracker.position_queue.append((pred_x, pred_y))
                
                if len(tracker.position_queue) > 0:
                    # 스무딩
                    avg_x = np.mean([p[0] for p in tracker.position_queue])
                    avg_y = np.mean([p[1] for p in tracker.position_queue])
                    
                    # 화면 좌표 변환 (좌우 반전)
                    screen_x = np.interp(INPUT_WIDTH - avg_x, [0, INPUT_WIDTH], [0, screen_w])
                    screen_y = np.interp(avg_y, [0, INPUT_HEIGHT], [0, screen_h])
                    
                    # 추가 스무딩
                    smooth_x = prev_x * 0.6 + screen_x * 0.4
                    smooth_y = prev_y * 0.6 + screen_y * 0.4
                    
                    # 마우스 이동
                    try:
                        pyautogui.moveTo(smooth_x, smooth_y, duration=0)
                        prev_x, prev_y = smooth_x, smooth_y
                    except:
                        pass  # 마우스 이동 실패 무시
                
                # 시각화
                cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)
                cv2.putText(frame, "Pupil", (px-20, py-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # 깜빡임 감지
            ear = tracker.calculate_ear(tracker.left_eye_roi, gray)
            
            if ear < EAR_THRESHOLD:
                blink_counter += 1
                if blink_counter == 1:
                    cv2.putText(frame, "BLINK!", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                if blink_counter >= BLINK_FRAMES:
                    current_time = time.time()
                    blink_times.append(current_time)
                    total_blinks += 1
                    
                    blink_times = [t for t in blink_times if current_time - t <= DOUBLE_BLINK_TIME]
                    
                    if len(blink_times) >= 2 and (current_time - last_click_time) > 1.0:
                        time_diff = blink_times[-1] - blink_times[-2]
                        if time_diff <= DOUBLE_BLINK_TIME:
                            print(f"🖱️ 클릭! (간격: {time_diff:.2f}초)")
                            pyautogui.click()
                            last_click_time = current_time
                            blink_times.clear()
                
                blink_counter = 0
        
        # ROI 표시
        if tracker.face_roi:
            x, y, w, h = tracker.face_roi
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "Face", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        if tracker.left_eye_roi:
            x, y, w, h = tracker.left_eye_roi
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "L-Eye", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        if tracker.right_eye_roi:
            x, y, w, h = tracker.right_eye_roi
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "R-Eye", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # FPS 계산
        fps_counter += 1
        if fps_counter % 10 == 0:
            current_fps = 10 / (time.time() - fps_start)
            fps_start = time.time()
        
        # 정보 표시
        info_y = 20
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        info_y += 20
        
        cv2.putText(frame, f"Blinks: {total_blinks}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        info_y += 20
        
        cv2.putText(frame, f"Confidence: {tracker.tracking_confidence}/{MAX_TRACKING_LOSS}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        info_y += 20
        
        cv2.putText(frame, f"Brightness: {mean_brightness:.0f}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 255), 1)
        info_y += 20
        
        cv2.putText(frame, f"Threshold: {tracker.pupil_threshold}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 255), 1)
        
        # 화면 표시
        cv2.imshow("Eye Tracking", frame)
        
        # 종료 체크
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q 또는 ESC
            print("\n종료 중...")
            break
        elif key == ord(' '):  # 스페이스바로 수동 재검출
            print("수동 재검출 실행")
            tracker.detect_face_and_eyes(gray)
    
    # 정리
    cap.release()
    cv2.destroyAllWindows()
    print("✅ 프로그램 종료")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()