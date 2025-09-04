import cv2
import numpy as np
import pyautogui
import time
import tensorflow as tf
from collections import deque

# ================== 설정 부분 ==================
# pyautogui 설정
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
screen_w, screen_h = pyautogui.size()

# 성능 관련 파라미터
FRAME_SKIP = 15  # 15프레임마다 1번 처리 (라즈베리파이 최적화)
INPUT_WIDTH = 160  # 입력 해상도 (더 낮춤)
INPUT_HEIGHT = 120
EYE_ROI_SIZE = 60  # 눈 영역 크기
GRAYSCALE = True  # 그레이스케일 사용

# 추적 관련 파라미터
MAX_TRACKING_LOSS = 10  # 추적 실패 허용 횟수
SMOOTHING_WINDOW = 5  # 스무딩 윈도우 크기
EAR_THRESHOLD = 0.22  # 눈 감김 임계값 (조정)
BLINK_FRAMES = 2
DOUBLE_BLINK_TIME = 0.8

# ================== TensorFlow Lite 모델 로드 ==================
class LightweightEyeTracker:
    def __init__(self):
        """경량 눈 추적기 초기화"""
        # Haar Cascade 로드 (초기 얼굴/눈 검출용)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # TFLite 모델 로드 (눈 랜드마크용)
        # 실제 사용시 적절한 모델 파일로 교체 필요
        try:
            # BlazeFace 또는 유사한 경량 모델 사용
            self.interpreter = tf.lite.Interpreter(model_path='eye_landmark_model.tflite')
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.use_tflite = True
            print("TFLite 모델 로드 성공")
        except:
            print("TFLite 모델 없음 - 기본 CV 방식 사용")
            self.use_tflite = False
        
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
        
    def _init_kalman_filter(self):
        """칼만 필터 초기화"""
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
        faces = self.face_cascade.detectMultiScale(gray_frame, 1.1, 4, minSize=(60, 60))
        
        if len(faces) > 0:
            # 가장 큰 얼굴 선택
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            
            # ROI 업데이트 (패딩 추가)
            self.face_roi = self._expand_roi(x, y, w, h, gray_frame.shape, self.roi_padding)
            
            # 얼굴 영역 내에서 눈 검출
            face_region = gray_frame[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(face_region, 1.05, 3, minSize=(20, 20))
            
            if len(eyes) >= 2:
                # 두 눈의 위치 구분 (왼쪽/오른쪽)
                eyes = sorted(eyes, key=lambda e: e[0])[:2]
                
                for i, (ex, ey, ew, eh) in enumerate(eyes):
                    # 전체 프레임 기준 좌표로 변환
                    eye_x, eye_y = x + ex, y + ey
                    
                    if i == 0:  # 왼쪽 눈
                        self.left_eye_roi = (eye_x, eye_y, ew, eh)
                    else:  # 오른쪽 눈
                        self.right_eye_roi = (eye_x, eye_y, ew, eh)
                
                self.tracking_confidence = MAX_TRACKING_LOSS
                return True
        
        self.tracking_confidence -= 1
        return False
    
    def track_in_roi(self, gray_frame):
        """ROI 영역 내에서만 추적"""
        if self.face_roi is None or self.tracking_confidence <= 0:
            return self.detect_face_and_eyes(gray_frame)
        
        x, y, w, h = self.face_roi
        roi = gray_frame[y:y+h, x:x+w]
        
        # ROI 내에서 얼굴 재검출
        faces = self.face_cascade.detectMultiScale(roi, 1.2, 2, minSize=(40, 40))
        
        if len(faces) > 0:
            fx, fy, fw, fh = faces[0]
            # 전체 프레임 기준으로 좌표 업데이트
            self.face_roi = self._expand_roi(x+fx, y+fy, fw, fh, gray_frame.shape, 20)
            self.tracking_confidence = min(self.tracking_confidence + 1, MAX_TRACKING_LOSS)
            return True
        
        self.tracking_confidence -= 1
        return False
    
    def _expand_roi(self, x, y, w, h, frame_shape, padding):
        """ROI 영역 확장 (패딩 추가)"""
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame_shape[1] - x, w + 2*padding)
        h = min(frame_shape[0] - y, h + 2*padding)
        return (x, y, w, h)
    
    def get_pupil_position(self, gray_frame, eye_roi):
        """동공 위치 추정"""
        if eye_roi is None:
            return None
        
        x, y, w, h = eye_roi
        eye_region = gray_frame[y:y+h, x:x+w]
        
        # 전처리: 히스토그램 평활화 및 가우시안 블러
        eye_region = cv2.equalizeHist(eye_region)
        eye_region = cv2.GaussianBlur(eye_region, (5, 5), 0)
        
        if self.use_tflite:
            # TFLite 모델 사용
            return self._predict_with_tflite(eye_region, x, y)
        else:
            # 기본 CV 방식 (임계값 + 컨투어)
            return self._detect_pupil_cv(eye_region, x, y)
    
    def _predict_with_tflite(self, eye_region, offset_x, offset_y):
        """TFLite 모델로 동공 위치 예측"""
        try:
            # 입력 크기에 맞게 리사이즈
            input_size = self.input_details[0]['shape'][1:3]
            resized = cv2.resize(eye_region, (input_size[1], input_size[0]))
            
            # 정규화 및 차원 확장
            input_data = np.expand_dims(resized, axis=(0, -1)).astype(np.float32) / 255.0
            
            # 추론
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            # 결과 파싱 (모델에 따라 수정 필요)
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # 상대 좌표를 절대 좌표로 변환
            rel_x, rel_y = output_data[0][:2]
            pupil_x = offset_x + int(rel_x * eye_region.shape[1])
            pupil_y = offset_y + int(rel_y * eye_region.shape[0])
            
            return (pupil_x, pupil_y)
        except:
            return None
    
    def _detect_pupil_cv(self, eye_region, offset_x, offset_y):
        """전통적 CV 방식으로 동공 검출"""
        # 적응형 임계값
        _, threshold = cv2.threshold(eye_region, 30, 255, cv2.THRESH_BINARY_INV)
        
        # 모폴로지 연산으로 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
        
        # 컨투어 검출
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 가장 큰 컨투어 선택
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 최소 크기 확인
            if cv2.contourArea(largest_contour) > 10:
                # 중심점 계산
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (offset_x + cx, offset_y + cy)
        
        # 검출 실패시 중앙값 반환
        return (offset_x + eye_region.shape[1]//2, offset_y + eye_region.shape[0]//2)
    
    def calculate_ear(self, eye_roi, gray_frame):
        """간단한 EAR 계산 (ROI 기반)"""
        if eye_roi is None:
            return 1.0
        
        x, y, w, h = eye_roi
        aspect_ratio = h / max(w, 1)
        
        # 정규화 (일반적으로 0.2~0.4 범위)
        return aspect_ratio

# ================== 메인 실행 부분 ==================
def main():
    # 초기화
    tracker = LightweightEyeTracker()
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    # 카메라 설정 (라즈베리파이 최적화)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 최소화
    
    # 상태 변수
    frame_count = 0
    prev_x, prev_y = screen_w//2, screen_h//2
    blink_counter = 0
    total_blinks = 0
    blink_times = []
    last_click_time = 0
    
    # FPS 측정용
    fps_start = time.time()
    fps_counter = 0
    
    print(f"화면 해상도: {screen_w}x{screen_h}")
    print(f"입력 해상도: {INPUT_WIDTH}x{INPUT_HEIGHT}")
    print("ESC 또는 'q'를 눌러 종료")
    print("빠르게 두 번 깜빡이면 클릭됩니다!")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        
        # 프레임 스킵 (성능 최적화)
        if frame_count % FRAME_SKIP != 0:
            continue
        
        # 전처리
        if GRAYSCALE:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # 추적 수행
        if frame_count == FRAME_SKIP:  # 첫 프레임
            tracker.detect_face_and_eyes(gray)
        else:
            tracker.track_in_roi(gray)
        
        # 동공 위치 추정
        if tracker.left_eye_roi:
            pupil_pos = tracker.get_pupil_position(gray, tracker.left_eye_roi)
            
            if pupil_pos:
                px, py = pupil_pos
                
                # Kalman 필터 적용
                measurement = np.array([[np.float32(px)], [np.float32(py)]])
                tracker.kalman.correct(measurement)
                prediction = tracker.kalman.predict()
                
                # 예측값 사용
                pred_x, pred_y = int(prediction[0]), int(prediction[1])
                
                # 큐에 추가 (스무딩용)
                tracker.position_queue.append((pred_x, pred_y))
                
                if len(tracker.position_queue) > 0:
                    # 스무딩 적용
                    avg_x = np.mean([p[0] for p in tracker.position_queue])
                    avg_y = np.mean([p[1] for p in tracker.position_queue])
                    
                    # 화면 좌표로 변환
                    screen_x = np.interp(avg_x, [0, INPUT_WIDTH], [0, screen_w])
                    screen_y = np.interp(avg_y, [0, INPUT_HEIGHT], [0, screen_h])
                    
                    # 추가 스무딩
                    smooth_x = prev_x * 0.7 + screen_x * 0.3
                    smooth_y = prev_y * 0.7 + screen_y * 0.3
                    
                    # 마우스 이동
                    pyautogui.moveTo(smooth_x, smooth_y, duration=0)
                    prev_x, prev_y = smooth_x, smooth_y
                
                # 시각화
                cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)
            
            # 깜빡임 감지
            ear = tracker.calculate_ear(tracker.left_eye_roi, gray)
            
            if ear < EAR_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= BLINK_FRAMES:
                    current_time = time.time()
                    blink_times.append(current_time)
                    total_blinks += 1
                    
                    # 오래된 기록 제거
                    blink_times = [t for t in blink_times if current_time - t <= DOUBLE_BLINK_TIME]
                    
                    # 더블 깜빡임 감지
                    if len(blink_times) >= 2 and (current_time - last_click_time) > 1.0:
                        time_diff = blink_times[-1] - blink_times[-2]
                        if time_diff <= DOUBLE_BLINK_TIME:
                            print(f"🖱️ 더블 깜빡임 클릭!")
                            pyautogui.click()
                            last_click_time = current_time
                            blink_times.clear()
                
                blink_counter = 0
        
        # ROI 시각화
        if tracker.face_roi:
            x, y, w, h = tracker.face_roi
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
        
        if tracker.left_eye_roi:
            x, y, w, h = tracker.left_eye_roi
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
        
        # FPS 계산 및 표시
        fps_counter += 1
        if fps_counter % 10 == 0:
            fps = 10 / (time.time() - fps_start)
            fps_start = time.time()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 정보 표시
        cv2.putText(frame, f"Blinks: {total_blinks}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        cv2.putText(frame, f"Confidence: {tracker.tracking_confidence}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 출력
        cv2.imshow("Lightweight Eye Tracker", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q 또는 ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()