import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# pyautogui 설정
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
screen_w, screen_h = pyautogui.size()

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

mp_face = mp.solutions.face_mesh

# 방법 1: 다른 초기화 방식 시도
try:
    face_mesh = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,  # 홍채 추적을 위해 필요
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    use_iris = True
    print("홍채 추적 모드로 초기화 성공")
except Exception as e:
    print(f"홍채 추적 모드 실패: {e}")
    # 대안: 기본 모드로 초기화하고 눈 코너로 홍채 위치 추정
    face_mesh = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    use_iris = False
    print("기본 모드로 초기화, 눈 추정 방식 사용")

# 홍채 및 눈 랜드마크 인덱스
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# 기본 눈 랜드마크 (홍채가 안될 경우 사용)
LEFT_EYE_CORNERS = [33, 133]  # 왼쪽 눈 양 끝
LEFT_EYE_TOP_BOTTOM = [159, 145]  # 왼쪽 눈 위아래
RIGHT_EYE_CORNERS = [362, 263]  # 오른쪽 눈 양 끝
RIGHT_EYE_TOP_BOTTOM = [386, 374]  # 오른쪽 눈 위아래

# 눈 깜빡임 감지를 위한 랜드마크 (EAR 계산용)
LEFT_EYE_POINTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_POINTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# EAR 계산을 위한 주요 6개 포인트
LEFT_EYE_MAIN = [33, 160, 158, 133, 153, 144]  # [left, top1, top2, right, bottom1, bottom2]
RIGHT_EYE_MAIN = [362, 385, 387, 263, 373, 380]

def calculate_ear(landmarks, eye_points, frame_w, frame_h):
    """EAR(Eye Aspect Ratio) 계산"""
    try:
        # 눈의 6개 주요 포인트 좌표 추출
        points = []
        for point_idx in eye_points:
            x = landmarks.landmark[point_idx].x * frame_w
            y = landmarks.landmark[point_idx].y * frame_h
            points.append([x, y])
        
        # EAR 계산: (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        p1, p2, p3, p4, p5, p6 = points
        
        # 수직 거리들
        dist1 = np.linalg.norm(np.array(p2) - np.array(p6))
        dist2 = np.linalg.norm(np.array(p3) - np.array(p5))
        
        # 수평 거리
        dist3 = np.linalg.norm(np.array(p1) - np.array(p4))
        
        if dist3 == 0:
            return 0
        
        ear = (dist1 + dist2) / (2.0 * dist3)
        return ear
    except:
        return 0

def estimate_iris_from_eye(landmarks, eye_corners, eye_tb, frame_w, frame_h):
    """기본 눈 랜드마크로부터 홍채 위치 추정"""
    try:
        # 눈의 양 끝점
        left_corner = [landmarks.landmark[eye_corners[0]].x * frame_w, 
                      landmarks.landmark[eye_corners[0]].y * frame_h]
        right_corner = [landmarks.landmark[eye_corners[1]].x * frame_w, 
                       landmarks.landmark[eye_corners[1]].y * frame_h]
        
        # 눈의 위아래점
        top_point = [landmarks.landmark[eye_tb[0]].x * frame_w, 
                    landmarks.landmark[eye_tb[0]].y * frame_h]
        bottom_point = [landmarks.landmark[eye_tb[1]].x * frame_w, 
                       landmarks.landmark[eye_tb[1]].y * frame_h]
        
        # 눈의 중심점 계산 (좌우 중점, 상하 중점)
        center_x = (left_corner[0] + right_corner[0]) / 2
        center_y = (top_point[1] + bottom_point[1]) / 2
        
        return int(center_x), int(center_y)
    except:
        return None, None

# 카메라 설정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

frame_count = 0
smoothing_factor = 0.3  # 마우스 움직임 스무딩
prev_x, prev_y = screen_w//2, screen_h//2

# 깜빡임 감지 변수들
EAR_THRESHOLD = 0.25  # 눈 감김 임계값
BLINK_FRAMES = 2  # 깜빡임으로 인정할 최소 프레임 수
DOUBLE_BLINK_TIME = 1  # 더블 깜빡임 인정 시간 (초)

blink_counter = 0
total_blinks = 0
blink_times = []  # 깜빡임 시간 기록
last_click_time = 0  # 마지막 클릭 시간 (중복 클릭 방지)

print(f"화면 해상도: {screen_w}x{screen_h}")
print("'q'를 눌러 종료")
print("빠르게 두 번 깜빡이면 클릭됩니다!")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("카메라 프레임을 불러올 수 없습니다!")
        break
    
    frame_count += 1
    if frame_count % 10 != 0:  # 성능을 위해 프레임 스킵
        continue
    
    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 손 인식
    hand_results = hands.process(frame_rgb)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
    
    # 얼굴(눈동자) 인식
    face_results = face_mesh.process(frame_rgb)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            try:
                if use_iris:
                    # 방법 1: 홍채 랜드마크 사용 (refine_landmarks=True일 때)
                    left_coords = np.array([
                        [int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)]
                        for i in LEFT_IRIS
                    ])
                    (lx, ly) = np.mean(left_coords, axis=0).astype(int)
                else:
                    # 방법 2: 눈 코너로부터 홍채 위치 추정
                    lx, ly = estimate_iris_from_eye(face_landmarks, LEFT_EYE_CORNERS, LEFT_EYE_TOP_BOTTOM, w, h)
                    if lx is None or ly is None:
                        continue
                
                # 화면 좌표로 변환 (좌우 반전 적용)
                screen_x = np.interp(w - lx, [0, w], [0, screen_w])  # 좌우 반전
                screen_y = np.interp(ly, [0, h], [0, screen_h])
                
                # 스무딩 적용
                smooth_x = prev_x * (1 - smoothing_factor) + screen_x * smoothing_factor
                smooth_y = prev_y * (1 - smoothing_factor) + screen_y * smoothing_factor
                
                # 마우스 이동
                pyautogui.moveTo(smooth_x, smooth_y, duration=0.01)
                
                # 이전 위치 업데이트
                prev_x, prev_y = smooth_x, smooth_y
                
                # 깜빡임 감지 (여기에 디버깅 출력 추가!)
                left_ear = calculate_ear(face_landmarks, LEFT_EYE_MAIN, w, h)
                right_ear = calculate_ear(face_landmarks, RIGHT_EYE_MAIN, w, h)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # 디버깅: EAR 값 실시간 출력 (너무 많으면 주석 처리)
                # print(f"현재 EAR: {avg_ear:.3f} (임계값: {EAR_THRESHOLD})")
                
                # 눈이 감겼는지 확인
                if avg_ear < EAR_THRESHOLD:
                    blink_counter += 1
                    # 디버깅: 눈 감김 감지
                    if blink_counter == 1:  # 첫 프레임에만 출력
                        print(f"👁️ 눈 감김 감지! EAR: {avg_ear:.3f} (프레임: {blink_counter})")
                else:
                    # 눈이 뜨였을 때 깜빡임 처리
                    if blink_counter >= BLINK_FRAMES:
                        current_time = time.time()
                        blink_times.append(current_time)
                        total_blinks += 1
                        
                        # 디버깅: 깜빡임 완료
                        print(f"✨ 깜빡임 완료! 총 {blink_counter}프레임, EAR: {avg_ear:.3f}, 총 깜빡임: {total_blinks}")
                        
                        # 오래된 깜빡임 기록 제거 (DOUBLE_BLINK_TIME 이전의 것들)
                        blink_times = [t for t in blink_times if current_time - t <= DOUBLE_BLINK_TIME]
                        
                        # 디버깅: 최근 깜빡임 기록
                        print(f"📊 최근 {DOUBLE_BLINK_TIME}초 내 깜빡임: {len(blink_times)}번")
                        
                        # 더블 깜빡임 감지
                        if len(blink_times) >= 2 and (current_time - last_click_time) > 1.0:
                            # 최근 두 깜빡임이 빠른 간격으로 발생했는지 확인
                            time_diff = blink_times[-1] - blink_times[-2]
                            if time_diff <= DOUBLE_BLINK_TIME:
                                print(f"🖱️ 더블 깜빡임 감지! 간격: {time_diff:.2f}초 -> 클릭 실행!")
                                pyautogui.click()
                                last_click_time = current_time
                                blink_times.clear()  # 깜빡임 기록 초기화
                                print("🔄 깜빡임 기록 초기화")
                            else:
                                print(f"⏰ 깜빡임 간격이 너무 김: {time_diff:.2f}초 (최대: {DOUBLE_BLINK_TIME}초)")
                    elif blink_counter > 0:
                        # 디버깅: 너무 짧은 깜빡임
                        print(f"⚡ 너무 짧은 깜빡임 무시: {blink_counter}프레임 (최소: {BLINK_FRAMES}프레임)")
                    
                    blink_counter = 0
                
                # 디버그용 시각화
                cv2.circle(frame, (lx, ly), 3, (0, 255, 255), -1)
                cv2.putText(frame, f"Eye: ({lx},{ly})", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Mouse: ({int(smooth_x)},{int(smooth_y)})", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"Blinks: {total_blinks}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                
            except Exception as e:
                print(f"눈 추적 중 오류: {e}")
                continue
    
    # 화면 중앙에 십자선 표시 (캘리브레이션용)
    cv2.line(frame, (w//2-10, h//2), (w//2+10, h//2), (0, 0, 255), 1)
    cv2.line(frame, (w//2, h//2-10), (w//2, h//2+10), (0, 0, 255), 1)
    
    # 출력
    cv2.imshow("Eye Tracking Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
