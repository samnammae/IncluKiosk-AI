import cv2
import numpy as np
import mediapipe as mp
import time
import math
from scipy.spatial.transform import Rotation as Rscipy
from collections import deque
import pyautogui
import threading
import queue  
import asyncio  
import websockets 
import json   

from . import config
from . import utils
from . import detection
from .click_controller import ClickController
import threading
import queue

# 마우스 제어 전역 변수
mouse_command_queue = queue.Queue()
mouse_action_lock = threading.Lock()  # pyautogui 호출 전용 락

# ============ WebSocket 통신 관련 추가 ============
HUB_URI = "ws://localhost:8766"
hub_ws = None
message_queue = queue.Queue()  # Hub → Worker 메시지 큐
send_queue = queue.Queue()     # Worker → Hub 메시지 큐

# 상태 플래그 (메시지 수신용)
calib_requested = False
screen_calib_requested = False
mouse_only_requested = False
mouse_click_requested = False
stop_requested = False
calib_just_completed = False
calib_start_time = None
CALIB_TIMEOUT = 8.0  # 캘리브레이션 타임아웃

pyautogui.PAUSE = 0           # ← 기본 0.1초 대기 제거
pyautogui.FAILSAFE = False    # ← 선택: 좌상단 구석에 가면 예외나는 기본 안전장치 비활성화

mouse_control_enabled = False       # 마우스 제어 토글 플래그(F7로 on/off). True일 때 보조 스레드가 mouse_target으로 커서를 이동
filter_length = 10                  # 시선 벡터 스무딩 버퍼 길이(최근 N개 평균)

# ============ 주먹 감지 관련 변수 (추가) ============
fist_detected = False
fist_debounce_time = 0.5  # 주먹 감지 디바운스 (0.5초)
fist_hold_time = 2.0      # 주먹 유지 시간 (2초)
fist_min_hand_size = config.FIST_MIN_HAND_SIZE  # 최소 손 크기 (픽셀, 손목~중지 끝 거리)
fist_thumb_threshold = 1.3  # 엄지 감지 완화 비율 (1.0=엄격, 1.3=권장, 1.5=관대)
last_fist_toggle_time = 0
fist_start_time = None    # 주먹을 처음 감지한 시간

# ============ WebSocket 클라이언트 (별도 스레드) ============
def websocket_thread_func():
    """WebSocket 통신을 담당하는 스레드 (asyncio 이벤트 루프 실행)"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(websocket_client())

async def websocket_client():
    """Hub와 WebSocket 연결 및 메시지 송수신"""
    global hub_ws
    
    try:
        async with websockets.connect(HUB_URI) as ws:
            hub_ws = ws
            print("🟠 [Eye Worker] Hub에 연결됨 (8766)")
            
            # 연결 직후 READY 신호 전송
            await ws.send(json.dumps({"type": "EYE_READY"}))
            
            # 송신 태스크 시작
            send_task = asyncio.create_task(send_messages(ws))
            
            # 수신 루프
            async for raw in ws:
                try:
                    data = json.loads(raw)
                    msg_type = data.get("type")
                    print(f"🟠 [Eye Worker] Hub로부터 수신: {msg_type}")
                    message_queue.put(data)
                except Exception as e:
                    print(f"🟠 [Eye Worker] 메시지 처리 오류: {e}")
            
            send_task.cancel()
            
    except Exception as e:
        print(f"🟠 [Eye Worker] WebSocket 오류: {e}")
    finally:
        hub_ws = None

async def send_messages(ws):
    """send_queue에 있는 메시지를 Hub로 전송"""
    while True:
        try:
            # 0.1초마다 큐 확인
            await asyncio.sleep(0.1)
            
            while not send_queue.empty():
                msg = send_queue.get_nowait()
                await ws.send(json.dumps(msg))
                print(f"🟠 [Eye Worker] Hub로 전송: {msg.get('type')}")
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"🟠 [Eye Worker] 전송 오류: {e}")

# =========================
# 3D 모니터 평면 상태(월드 좌표계)
# =========================
monitor_corners = None   # 모니터 모서리 [p0,p1,p2,p3], 시계/반시계(?) 순 회전으로 가정
monitor_center_w = None  # 평면 중심(월드)
monitor_normal_w = None  # 평면 법선(월드)
units_per_cm = None      # 월드 단위/센티미터(대략적 스케일)

# 마우스 목표 좌표(보조 스레드와 공유) + 동기화 락
mouse_target = [config.CENTER_X, config.CENTER_Y]
mouse_lock = threading.Lock()

# Calibration offsets for screen mapping
# 화면 매핑 보정값(캘리브레이션 시 's'로 중앙 정렬하기 위해 yaw/pitch 오프셋 저장)
calibration_offset_yaw = 0
calibration_offset_pitch = 0

# Buffers to store recent gaze data for smoothing
# 최근 결합 시선(combined gaze) 방향 벡터 버퍼(평활화용)
combined_gaze_directions = deque(maxlen=filter_length)

# reference matrices to fix coordinate flipping issue
# These help keep the axes consistent from frame to frame by stabilizing eigenvector directions
# ===== 고정 참조 회전행렬 컨테이너 =====
# PCA 고유벡터 기반 회전에서 프레임 사이 부호 뒤집힘(sign flip)을 막기 위한 참조
R_ref_nose = [None]
R_ref_forehead = [None]
calibration_nose_scale = None

# =========================
# MediaPipe FaceMesh 초기화
# =========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============ MediaPipe Hands 초기화 ============
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# =========================
# 카메라 열기
# =========================
cap = cv2.VideoCapture(config.CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === Nose-only landmark indices (for stable up/down eye sphere tracking) ===
# These landmarks are near the nose and are less affected by lateral head movement
# =========================
# 코 주변 안정 영역(시선 추정에서 상하 흔들림 억제용 샘플 인덱스)
# =========================
# MediaPipe FaceMesh의 468+ 랜드마크 중 코/미간/비근 주변 인덱스들
nose_indices = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241, 
                461, 125, 354, 218, 438, 195, 167, 393, 165, 391,
                3, 248]

# =========================
# 코 주변 소영역의 PCA 좌표계 계산 및 그리기
# =========================
def compute_and_draw_coordinate_box(frame, face_landmarks, indices, ref_matrix_container, color=(0, 255, 0), size=80):
    """선택된 랜드마크(indices)의 3D 좌표로 공분산→고유분해(PCA)하여 주축(eigvecs) 획득.
    오른손 좌표계를 강제(det<0이면 축 하나 부호 반전), 이후 참조 회전행렬과 비교하여
    축 부호를 안정화한 R_final을 산출. 2D 프레임 위에 큐브와 XYZ축을 그린 뒤,
    중심점(center), R_final, 사용된 3D 점들을 반환.
    """
    # Extract 3D positions of selected landmarks
    # 선택된 랜드마크의 3D 위치(픽셀 단위 스케일: x*w, y*h, z*w)
    points_3d = np.array([
        [face_landmarks[i].x * w, face_landmarks[i].y * h, face_landmarks[i].z * w]
        for i in indices
    ])

    # Compute the average position as the center of this substructure
    center = np.mean(points_3d, axis=0) # 중심(평균)

    # PCA-based orientation: Compute eigenvectors of the covariance matrix
    # 공분산 → 고유분해(내림차순 정렬)
    centered = points_3d - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:, np.argsort(-eigvals)]  # Sort by descending eigenvalue (major axes)

    # Ensure the orientation matrix is right-handed
    # 오른손 좌표계 강제(det<0이면 마지막 축 부호 반전)
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1

    # Convert to Euler angles and re-construct rotation matrix (optional but clarifies the transform)
    # (선택) 오일러 변환 거쳐 재구성 — 큰 의미는 없지만 회전표현을 명시적으로 만드는 과정
    r = Rscipy.from_matrix(eigvecs)
    roll, pitch, yaw = r.as_euler('zyx', degrees=False)
    yaw *= 1
    roll *= 1
    R_final = Rscipy.from_euler('zyx', [roll, pitch, yaw]).as_matrix()

    # === Stabilize rotation with reference matrix to avoid flipping during eigenvector sign change ===
    # 참조 행렬과 축 정렬(연속 프레임 간 축 부호 뒤집힘 방지)
    if ref_matrix_container[0] is None:
        ref_matrix_container[0] = R_final.copy()
    else:
        R_ref = ref_matrix_container[0]
        for i in range(3):
            if np.dot(R_final[:, i], R_ref[:, i]) < 0:
                R_final[:, i] *= -1

    return center, R_final, points_3d

# =========================
# 시선 벡터 → 화면 좌표 변환(각도 기반 간단 매핑)
# =========================
def convert_gaze_to_screen_coordinates(combined_gaze_direction, calibration_offset_yaw, calibration_offset_pitch):
    """
    This function is adapted from the old script's vector-to-screen mapping logic
    """
    """결합 3D 시선 벡터(월드 기준)를 yaw/pitch 각도로 투영 후 화면 해상도로 선형 매핑.
    - yawDegrees/pitchDegrees 범위 내에서 모니터 좌표로 변환
    - 's' 키로 중앙 보정(calibration_offset_*)을 적용
    반환: (screen_x, screen_y, raw_yaw_deg, raw_pitch_deg)
    """
    # Reference forward direction (camera looking straight ahead)
    # 기준 정면 방향(카메라에서 화면 안쪽 -Z)
    reference_forward = np.array([0, 0, -1])  # Z-axis into the screen

    # Normalize the gaze direction // 단위화
    avg_direction = combined_gaze_direction / np.linalg.norm(combined_gaze_direction)

    # Horizontal (yaw) angle from reference (project onto XZ plane)
    # 수평(yaw): XZ 평면 투영
    xz_proj = np.array([avg_direction[0], 0, avg_direction[2]])
    xz_proj /= np.linalg.norm(xz_proj)
    yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz_proj), -1.0, 1.0))
    if avg_direction[0] < 0:
        yaw_rad = -yaw_rad  # 왼쪽을 음수

    # Vertical (pitch) angle from reference (project onto YZ plane)
    # 수직(pitch): YZ 평면 투영
    yz_proj = np.array([0, avg_direction[1], avg_direction[2]])
    yz_proj /= np.linalg.norm(yz_proj)
    pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz_proj), -1.0, 1.0))
    if avg_direction[1] > 0:
        pitch_rad = -pitch_rad  # 위쪽을 양수

    # Convert to degrees and re-center around 0
    # 도 단위
    yaw_deg = np.degrees(yaw_rad)
    pitch_deg = np.degrees(pitch_rad)

    # Convert left rotations to 0-180 (from old script logic)
    # 왼쪽(-)을 양수로 뒤집는 보정(기존 스크립트 호환)
    if yaw_deg < 0:
        yaw_deg = -(yaw_deg)
    elif yaw_deg > 0:
        yaw_deg = - yaw_deg

    #yaw is now converted to -90 (looking directly left) to +90 (looking directly right), wrt camera
    #pitch is now converted to +90 (looking straight up) and -90 (looking straight down), wrt camera
    raw_yaw_deg = yaw_deg
    raw_pitch_deg = pitch_deg

    # Specify degrees at which screen border will be reached
    # 화면 경계에 해당하는 각도 범위(경험값)
    yawDegrees = config.YAW_SENSITIVITY  # x degrees left or right # 좌우 한계(도)
    pitchDegrees = config.PITCH_SENSITIVITY  # x degrees up or down # 상하 한계(도)

    # Apply calibration offsets
    # 중앙 보정 적용('s' 키로 저장된 오프셋)
    yaw_deg += calibration_offset_yaw
    pitch_deg += calibration_offset_pitch
    
    # Map to full screen resolution
    # 해상도로 선형 매핑
    screen_x = int(((yaw_deg + yawDegrees) / (2 * yawDegrees)) * config.MONITOR_WIDTH)
    screen_y = int(((pitchDegrees - pitch_deg) / (2 * pitchDegrees)) * config.MONITOR_HEIGHT)

    # Clamp screen position to monitor bounds
    # 모니터 가장자리 밖으로 나가지 않도록 클램프(여백 10px)
    screen_x = max(10, min(screen_x, config.MONITOR_WIDTH - 10))
    screen_y = max(10, min(screen_y, config.MONITOR_HEIGHT - 10))

    return screen_x, screen_y, raw_yaw_deg, raw_pitch_deg

# =========================
# 마우스 이동 보조 스레드(토글 시 mouse_target으로 이동)
# =========================
def mouse_mover():
    """마우스 이동 스레드 - 개선 버전"""
    while True:
        try:
            # 큐에서 명령 확인 (click 명령 우선)
            try:
                command = mouse_command_queue.get_nowait()
                if command['type'] == 'click':
                    with mouse_action_lock:
                        pyautogui.click(command['x'], command['y'])
                        print(f"🟢 [Eye Worker] [Click] ✔ at ({command['x']}, {command['y']})")
                        time.sleep(0.1)  # 클릭 후 안정화
            except queue.Empty:
                pass
            
            # 일반 마우스 이동
            if mouse_control_enabled:
                with mouse_lock:
                    x, y = mouse_target
                with mouse_action_lock:
                    pyautogui.moveTo(x, y, _pause=False)
            
            time.sleep(0.016)
            
        except Exception as e:
            print(f"🔴 [Mouse Mover] Error: {e}")
            time.sleep(0.1)

# Start mouse movement thread
# 데몬 스레드 시작
threading.Thread(target=mouse_mover, daemon=True).start()

# ============ WebSocket 스레드 시작 (추가) ============
threading.Thread(target=websocket_thread_func, daemon=True).start()
print("🟠 [Eye Worker] WebSocket 스레드 시작")

# 잠시 대기 (WebSocket 연결 완료 대기)
time.sleep(1.0)

# ============ 클릭 컨트롤러 초기화 (추가) ============
click_controller = ClickController(
    prepare_time=0.4,    # 0.4초 후 준비 단계 시작
    progress_time=0.8,   # 0.8초 후 진행 단계 시작
    click_time=1.2,      # 1.2초 후 클릭 실행
    radius=50,           # 50픽셀 반경 허용
    cooldown=0.5         # 클릭 후 0.5초 대기
)

# Eye sphere tracking variables (from new script)
# =========================
# 눈 구체(eye sphere) 추정/고정 관련 상태값
# =========================
left_sphere_locked = False
left_sphere_local_offset = None
left_calibration_nose_scale = None

right_sphere_locked = False
right_sphere_local_offset = None
right_calibration_nose_scale = None

# =========================
# 메인 루프: 프레임 처리
# =========================

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # ============ Hub 메시지 처리 ============
    try:
        while not message_queue.empty():
            msg = message_queue.get_nowait()
            msg_type = msg.get("type")
            
            if msg_type == "EYE_CALIB_ON":
                print("🟠 [Eye Worker] got EYE_CALIB_ON")
                calib_requested = True
                calib_start_time = time.time()  # 캘리브레이션 시작 시간 기록
            elif msg_type == "MOUSE_ON":
                print("🟠 [Eye Worker] got MOUSE_ON")
                mouse_only_requested = True
            elif msg_type == "MOUSE_OFF":
                print("🟠 [Eye Worker] got MOUSE_OFF")
                mouse_control_enabled = False
                click_controller.set_enabled(False)
            elif msg_type == "EYE_ORDER_ON":
                print("🟠 [Eye Worker] got EYE_ORDER_ON")
                mouse_click_requested = True
            elif msg_type == "STOP_ALL":
                print("🟠 [Eye Worker] got STOP_ALL")
                stop_requested = True
    except queue.Empty:
        pass

    combined_dir = None  # will be filled once you compute a smoothed direction // 현재 프레임의 결합 시선(평활화 후)

    # MediaPipe는 RGB 입력 요구
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    hands_results = hands.process(frame_rgb)
    
    current_fist_detected = False
    if hands_results.multi_hand_landmarks:
        print(f"[HAND DETECTION] Hand detected (count={len(hands_results.multi_hand_landmarks)})")
        for hand_landmarks in hands_results.multi_hand_landmarks:
            if detection.is_fist(hand_landmarks, w, h, 
                      min_hand_size=fist_min_hand_size, 
                      thumb_threshold=fist_thumb_threshold):
                print(f"[HAND DETECTION] 주먹 감지 됨")
                current_fist_detected = True
                break
    # 주먹 감지 유지 시간 체크
    current_time = time.time()
    
    if current_fist_detected:
        # 주먹이 감지됨
        if fist_start_time is None:
            # 주먹을 처음 감지 시작
            fist_start_time = current_time
            print(f"🟢 [Eye Worker] 주먹 감지 시작... (2초 유지 필요)")
        else:
            # 주먹을 계속 유지중
            hold_duration = current_time - fist_start_time
            if hold_duration >= fist_hold_time and not fist_detected:
                # 2초 이상 유지 → 주먹 인식 완료
                if current_time - last_fist_toggle_time > fist_debounce_time:
                    fist_detected = True
                    last_fist_toggle_time = current_time
                    
                    send_queue.put({"type": "FIST_DETECTED"})
                    print(f"🟢 [Eye Worker] ✅ 주먹 인식 완료! ({hold_duration:.1f}초 유지)")
    else:
        # 주먹이 감지되지 않음
        if fist_start_time is not None:
            # 주먹을 풀었음
            hold_duration = current_time - fist_start_time
            if hold_duration < fist_hold_time:
                print(f"🟡 [Eye Worker] 주먹 감지 취소 ({hold_duration:.1f}초 < {fist_hold_time}초)")
            fist_start_time = None
        
        # fist_detected 플래그 리셋
        if fist_detected:
            if current_time - last_fist_toggle_time > fist_debounce_time:
                fist_detected = False
                last_fist_toggle_time = current_time
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark

        # 미디어 파이프 iris 모델 양안 홍채 중심 인덱스
        left_iris_idx = 468
        right_iris_idx = 473
        left_iris = face_landmarks[left_iris_idx]
        right_iris = face_landmarks[right_iris_idx]

        # Compute and draw stabilized coordinate frame from nose region
        # 코 주변 영역의 PCA 좌표계 및 중심/회전 획득 + 디버그 큐브/축 그리기
        head_center, R_final, nose_points_3d = compute_and_draw_coordinate_box(
            frame,
            face_landmarks,
            nose_indices,
            R_ref_nose,
            color=(0, 255, 0),
            size=80
        )

        # TODO compute this radius using canthus during calibration
        # 눈 구체 반경(기준) — 실제로는 canthus(안각) 등으로 보정하는 것이 좋음
        base_radius = 20  # radius at calibration distance

        # (좌안) 홍채 위치 표시(잠금 전에는 홍채만, 잠금 후엔 구체를 그림)
        x_iris_l = int(left_iris.x * w)
        y_iris_l = int(left_iris.y * h)
        # === LEFT EYE visualization ===
        if not left_sphere_locked:
            pass
        else:
            current_nose_scale = utils.compute_scale(nose_points_3d)
            scale_ratio = current_nose_scale / left_calibration_nose_scale if left_calibration_nose_scale else 1.0
            scaled_offset = left_sphere_local_offset * scale_ratio
            sphere_world_l = head_center + R_final @ scaled_offset
            x_sphere_l, y_sphere_l = int(sphere_world_l[0]), int(sphere_world_l[1])
            scaled_radius_l = int(base_radius * scale_ratio)

        # (우안)
        x_iris_r = int(right_iris.x * w)
        y_iris_r = int(right_iris.y * h)
        # === RIGHT EYE visualization ===
        if not right_sphere_locked:
            pass
        else:
            current_nose_scale = utils.compute_scale(nose_points_3d)
            scale_ratio_r = current_nose_scale / right_calibration_nose_scale if right_calibration_nose_scale else 1.0
            scaled_offset_r = right_sphere_local_offset * scale_ratio_r
            sphere_world_r = head_center + R_final @ scaled_offset_r
            x_sphere_r, y_sphere_r = int(sphere_world_r[0]), int(sphere_world_r[1])
            scaled_radius_r = int(base_radius * scale_ratio_r)

        # 홍채의 3D 위치(픽셀 스케일)
        iris_3d_left = np.array([left_iris.x * w, left_iris.y * h, left_iris.z * w])
        iris_3d_right = np.array([right_iris.x * w, right_iris.y * h, right_iris.z * w])

        # 양안 구체가 잠겼다면 시선 계산/시각화 및 화면 좌표 변환 수행
        if left_sphere_locked and right_sphere_locked:
            # ==== COMPUTE COMBINED GAZE DIRECTION FOR SCREEN MAPPING ====
            # Calculate individual gaze directions
            # (1) 개별 시선 벡터 → (2) 평균 → (3) 정규화
            left_gaze_dir = iris_3d_left - sphere_world_l
            left_gaze_dir /= np.linalg.norm(left_gaze_dir)
            
            right_gaze_dir = iris_3d_right - sphere_world_r
            right_gaze_dir /= np.linalg.norm(right_gaze_dir)
            
            # Combine gaze directions (average)
            # 최근 N개 방향 평균으로 평활화
            raw_combined_direction = (left_gaze_dir + right_gaze_dir) / 2
            raw_combined_direction /= np.linalg.norm(raw_combined_direction)

            # Update direction buffer for smoothing
            combined_gaze_directions.append(raw_combined_direction)

            # Smoothed direction
            avg_combined_direction = np.mean(combined_gaze_directions, axis=0)
            avg_combined_direction /= np.linalg.norm(avg_combined_direction)

            combined_dir = avg_combined_direction

            # ==== CONVERT GAZE TO SCREEN COORDINATES ====
            # 화면 좌표로 변환(중앙 보정 포함)
            screen_x, screen_y, raw_yaw, raw_pitch = convert_gaze_to_screen_coordinates(
                avg_combined_direction, 
                calibration_offset_yaw, 
                calibration_offset_pitch
            )
            
            # ============ 클릭 컨트롤러 업데이트 (추가) ============
            current_time = time.time()
            current_pos = (screen_x, screen_y) if mouse_control_enabled else None
            
            click_state = click_controller.update(current_pos, current_time)
            
            # 클릭 실행
            if click_state['should_click']:
                # 직접 click 호출 대신 큐에 명령 추가
                mouse_command_queue.put({
                    'type': 'click',
                    'x': screen_x,
                    'y': screen_y
                })
                print(f"🟢 [Eye Worker] [Click] Queued at ({screen_x}, {screen_y}) - Total: {click_controller.get_click_count()}")

            # 마우스 이동 목표 업데이트(스레드가 이동 수행)
            if mouse_control_enabled:
                with mouse_lock:
                    mouse_target[0] = screen_x
                    mouse_target[1] = screen_y

        # Build 3D landmarks in your existing scale (x*w, y*h, z*w)
        # 3D 디버그 뷰에 사용할 전체 랜드마크(월드 스케일) 구성
        landmarks3d = None
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            landmarks3d = np.array([[p.x * w, p.y * h, p.z * w] for p in lm], dtype=float)

    # -------------------------
    # 메시지 처리 (전역)
    # -------------------------
    
    # 종료 요청
    if stop_requested:
        print("🟠 [Eye Worker] 종료 요청 수신")
        break
    
    # 마우스 제어만 활성화 (MOUSE_ON)
    if mouse_only_requested:
        mouse_only_requested = False
        mouse_control_enabled = True
        click_controller.set_enabled(False)
        print(f"🟠 [Eye Worker] 마우스 제어만 활성화 (클릭 OFF)")
    
    # 마우스 + 클릭 활성화 (EYE_ORDER_ON)
    if mouse_click_requested:
        mouse_click_requested = False
        mouse_control_enabled = True
        click_controller.set_enabled(True)
        print(f"🟠 [Eye Worker] 마우스 제어 + 클릭 활성화")
    
    # cv2.waitKey는 유지 (더미 윈도우용)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):  # 수동 종료용으로 남겨둠
        break
    
    
    # ============ 캘리브레이션 직후 다음 프레임 처리 (추가) ============
    if calib_just_completed and left_sphere_locked and right_sphere_locked:
        print("🟠 [Eye Worker] 화면 중앙 보정 자동 실행...")
        
        # 현재 시선 방향 계산
        left_gaze_dir = iris_3d_left - sphere_world_l  # ✅ 정확히 동일!
        left_gaze_dir /= np.linalg.norm(left_gaze_dir)
        right_gaze_dir = iris_3d_right - sphere_world_r  # ✅ 정확히 동일!
        right_gaze_dir /= np.linalg.norm(right_gaze_dir)
        current_combined_direction = (left_gaze_dir + right_gaze_dir) / 2
        current_combined_direction /= np.linalg.norm(current_combined_direction)
        
        # 보정값 계산
        _, _, raw_yaw, raw_pitch = convert_gaze_to_screen_coordinates(
            current_combined_direction, 0, 0
        )
        
        calibration_offset_yaw = 0 - raw_yaw
        calibration_offset_pitch = 0 - raw_pitch
        
        print(f"🟠 [Eye Worker] 화면 보정 완료: Yaw={calibration_offset_yaw:.2f}°, Pitch={calibration_offset_pitch:.2f}°")

        send_queue.put({"type": "EYE_CALIB_COMPLETE"})
        # 플래그 해제
        calib_requested = False
        calib_just_completed = False
    
    # ============ 캘리브레이션 타임아웃 체크 ============
    if calib_requested and calib_start_time is not None:
        if time.time() - calib_start_time > CALIB_TIMEOUT:
            print("🟠 [Eye Worker] ❌ 캘리브레이션 타임아웃 (10초 초과)")
            send_queue.put({"type": "EYE_CALIB_ERR", "message": "얼굴 감지 실패 - 10초 초과"})
            calib_requested = False
            calib_start_time = None
    
    # ============ EYE_CALIB_ON 처리 (기존 'c' 키 로직) ============
    if calib_requested and not (left_sphere_locked and right_sphere_locked):
        # 얼굴이 감지되지 않으면 캘리브레이션 불가
        if not results.multi_face_landmarks:
            # 얼굴이 없으면 다음 프레임 대기 (타임아웃까지)
            pass
        else:
            try:
                calib_requested = False
                
                # 1) 현 프레임의 코 영역 스케일 측정
                current_nose_scale = utils.compute_scale(nose_points_3d)
                
                # 2) (좌안) 홍채의 머리 로컬 오프셋을 계산하고 구체 중심을 앞(z+)으로 base_radius만큼 이동
                left_sphere_local_offset = R_final.T @ (iris_3d_left - head_center)
                camera_dir_world = np.array([0, 0, 1])                      # 카메라 z+ 방향(프레임 전방)
                camera_dir_local = R_final.T @ camera_dir_world             # 머리 로컬로 변환
                left_sphere_local_offset += base_radius * camera_dir_local
                left_calibration_nose_scale = current_nose_scale
                left_sphere_locked = True # Lock LEFT eye

                # 3) (우안) 동일 로직
                right_sphere_local_offset = R_final.T @ (iris_3d_right - head_center)
                right_sphere_local_offset += base_radius * camera_dir_local  # use same camera_dir_local
                right_calibration_nose_scale = current_nose_scale
                right_sphere_locked = True # Lock RIGHT eye

                # 4) 캘리브레이션 시점의 월드 좌표 구체 중심(스케일 1 가정)
                # === Create 3D monitor plane at calibration ===
                # Compute instantaneous sphere positions at calibration distance (scale=1)
                sphere_world_l_calib = head_center + R_final @ left_sphere_local_offset
                sphere_world_r_calib = head_center + R_final @ right_sphere_local_offset

                # 5) 양안 시선 평균으로 정면 힌트(forward_hint) 계산
                # Estimate a forward gaze direction from the two eyes
                left_dir  = iris_3d_left  - sphere_world_l_calib
                right_dir = iris_3d_right - sphere_world_r_calib
                # Normalize (guard zero)
                if np.linalg.norm(left_dir)  > 1e-9: left_dir  /= np.linalg.norm(left_dir)
                if np.linalg.norm(right_dir) > 1e-9: right_dir /= np.linalg.norm(right_dir)
                forward_hint = (left_dir + right_dir) * 0.5
                if np.linalg.norm(forward_hint) > 1e-9:
                    forward_hint /= np.linalg.norm(forward_hint)
                else:
                    forward_hint = None  # fallback to head frame

                gaze_origin = (sphere_world_l_calib + sphere_world_r_calib) / 2
                gaze_dir = forward_hint  # already normalized
                
                # 6) 모니터 평면 생성 + 디버그 월드 고정(모니터 중심을 피벗으로)
                monitor_corners, monitor_center_w, monitor_normal_w, units_per_cm = utils.create_monitor_plane(
                    head_center, R_final, face_landmarks, w, h,
                    forward_hint=forward_hint,
                    gaze_origin=gaze_origin,
                    gaze_dir=gaze_dir
                )

                # Freeze the debug world's orbit pivot at the calibrated monitor center
                #global debug_world_frozen, orbit_pivot_frozen
                debug_world_frozen = True
                orbit_pivot_frozen = monitor_center_w.copy()
                print(f"🟠 [Eye Worker] units_per_cm={units_per_cm:.3f}, center={monitor_center_w}, normal={monitor_normal_w}")        
                print("🟠 [Eye Worker] 캘리브레이션 완료")
                
                calib_just_completed = True
                calib_start_time = None  # 타임아웃 타이머 리셋
                
            except (NameError, IndexError, AttributeError, ValueError) as e:
                # 얼굴/눈 감지 실패 또는 계산 오류
                print(f"🟠 [Eye Worker] ❌ 캘리브레이션 실패: {e}")
                send_queue.put({"type": "EYE_CALIB_ERR", "message": "얼굴 또는 눈 감지 실패"})
                calib_requested = False
                calib_start_time = None
                # 잠금 상태 리셋
                left_sphere_locked = False
                right_sphere_locked = False
            except Exception as e:
                # 기타 예상치 못한 오류
                print(f"🟠 [Eye Worker] ❌ 캘리브레이션 예외: {e}")
                send_queue.put({"type": "EYE_CALIB_ERR", "message": "캘리브레이션 오류 발생"})
                calib_requested = False
                calib_start_time = None
                left_sphere_locked = False
                right_sphere_locked = False

    # # -------------------------
    # # 키보드 입력 처리(전역)
    # # -------------------------
    # # F7: 마우스 제어 토글(디바운싱)
    # if keyboard.is_pressed('f7'):
    #     mouse_control_enabled = not mouse_control_enabled
    #     click_controller.set_enabled(mouse_control_enabled)
    #     print(f"[Mouse Control] {'Enabled' if mouse_control_enabled else 'Disabled'}")
    #     print(f"[Click Controller] {'Enabled' if mouse_control_enabled else 'Disabled'}") 
    #     time.sleep(0.3)  # debounce to prevent rapid toggling

    # key = cv2.waitKey(1) & 0xFF
    
    # if key == ord('q'):
    #     break
    # elif key == ord('c') and not (left_sphere_locked and right_sphere_locked):
    #     # 1) 현 프레임의 코 영역 스케일 측정
    #     current_nose_scale = utils.compute_scale(nose_points_3d)
        
    #     # 2) (좌안) 홍채의 머리 로컬 오프셋을 계산하고 구체 중심을 앞(z+)으로 base_radius만큼 이동
    #     left_sphere_local_offset = R_final.T @ (iris_3d_left - head_center)
    #     camera_dir_world = np.array([0, 0, 1])                      # 카메라 z+ 방향(프레임 전방)
    #     camera_dir_local = R_final.T @ camera_dir_world             # 머리 로컬로 변환
    #     left_sphere_local_offset += base_radius * camera_dir_local
    #     left_calibration_nose_scale = current_nose_scale
    #     left_sphere_locked = True # Lock LEFT eye

    #     # 3) (우안) 동일 로직
    #     right_sphere_local_offset = R_final.T @ (iris_3d_right - head_center)
    #     right_sphere_local_offset += base_radius * camera_dir_local  # use same camera_dir_local
    #     right_calibration_nose_scale = current_nose_scale
    #     right_sphere_locked = True # Lock RIGHT eye

    #     # 4) 캘리브레이션 시점의 월드 좌표 구체 중심(스케일 1 가정)
    #     # === Create 3D monitor plane at calibration ===
    #     # Compute instantaneous sphere positions at calibration distance (scale=1)
    #     sphere_world_l_calib = head_center + R_final @ left_sphere_local_offset
    #     sphere_world_r_calib = head_center + R_final @ right_sphere_local_offset

    #     # 5) 양안 시선 평균으로 정면 힌트(forward_hint) 계산
    #     # Estimate a forward gaze direction from the two eyes
    #     left_dir  = iris_3d_left  - sphere_world_l_calib
    #     right_dir = iris_3d_right - sphere_world_r_calib
    #     # Normalize (guard zero)
    #     if np.linalg.norm(left_dir)  > 1e-9: left_dir  /= np.linalg.norm(left_dir)
    #     if np.linalg.norm(right_dir) > 1e-9: right_dir /= np.linalg.norm(right_dir)
    #     forward_hint = (left_dir + right_dir) * 0.5
    #     if np.linalg.norm(forward_hint) > 1e-9:
    #         forward_hint /= np.linalg.norm(forward_hint)
    #     else:
    #         forward_hint = None  # fallback to head frame

    #     gaze_origin = (sphere_world_l_calib + sphere_world_r_calib) / 2
    #     gaze_dir = forward_hint  # already normalized
        
    #     # 6) 모니터 평면 생성 + 디버그 월드 고정(모니터 중심을 피벗으로)
    #     monitor_corners, monitor_center_w, monitor_normal_w, units_per_cm = utils.create_monitor_plane(
    #         head_center, R_final, face_landmarks, w, h,
    #         forward_hint=forward_hint,
    #         gaze_origin=gaze_origin,
    #         gaze_dir=gaze_dir
    #     )

    #     # Freeze the debug world's orbit pivot at the calibrated monitor center
    #     #global debug_world_frozen, orbit_pivot_frozen
    #     debug_world_frozen = True
    #     orbit_pivot_frozen = monitor_center_w.copy()
    #     print("[Debug View] World pivot frozen at monitor center.")
    #     print(f"[Monitor] units_per_cm={units_per_cm:.3f}, center={monitor_center_w}, normal={monitor_normal_w}")
    #     print("[Both Spheres Locked] Eye sphere calibration complete.")
        
    # elif key == ord('s') and left_sphere_locked and right_sphere_locked:
    #     # 화면 중앙 캘리브레이션: 현재 시선을 (0,0) 기준으로 간주하여 오프셋 저장
    #     # Screen calibration - user should look at center of screen when pressing 's'
    #     # Get current gaze direction
    #     left_gaze_dir = iris_3d_left - sphere_world_l
    #     left_gaze_dir /= np.linalg.norm(left_gaze_dir)
    #     right_gaze_dir = iris_3d_right - sphere_world_r
    #     right_gaze_dir /= np.linalg.norm(right_gaze_dir)
    #     current_combined_direction = (left_gaze_dir + right_gaze_dir) / 2
    #     current_combined_direction /= np.linalg.norm(current_combined_direction)
        
    #     # Calculate what the raw angles would be without calibration
    #     _, _, raw_yaw, raw_pitch = convert_gaze_to_screen_coordinates(
    #         current_combined_direction, 0, 0  # no calibration offset
    #     )
        
    #     # Set calibration offsets to center the gaze
    #     calibration_offset_yaw = 0 - raw_yaw
    #     calibration_offset_pitch = 0 - raw_pitch
        
    #     print(f"[Screen Calibrated] Offset Yaw: {calibration_offset_yaw:.2f}, Offset Pitch: {calibration_offset_pitch:.2f}")

cap.release()
cv2.destroyAllWindows()

# ============ 종료 신호 전송 (추가) ============
if hub_ws:
    send_queue.put({"type": "WORKER_EXIT"})
    time.sleep(0.5)