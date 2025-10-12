import cv2
import numpy as np
import mediapipe as mp
import time
import math
from scipy.spatial.transform import Rotation as Rscipy
from collections import deque
import pyautogui
import threading
import keyboard
from types import SimpleNamespace
from collections import defaultdict
import asyncio
import websockets
import json
import sys
from pathlib import Path
import subprocess

touch_active = False
last_touch_end = 0.0
TOUCH_HOLDOFF = 0.5  # 터치가 끝난 뒤 0.5초 동안 주입 보류

sent_ready = False  # 첫 landmarks 확보 시 EYE_READY 1회 전송

async def send_internal(payload: dict):
    try:
        async with websockets.connect(WS_URL) as ws:
            await ws.send(json.dumps(payload, ensure_ascii=False))
            await asyncio.sleep(0.05)
            dbg(f"[WS] sent {payload}")
    except Exception as e:
        dbg(f"[WS] send_internal failed: {e}")

# --- WebSocket flags ---
toggle_mouse_requested = False  # F7 대체 (토글)
eye_calib_requested    = False  # c  대체 (1회)
force_mouse_on         = False  # 강제 마우스 ON (MOUSE_ON)
fist_enabled           = True   # True: 주먹 인식, False: 비활성 (EYE_ORDER_ON에서 끔)

BASE_DIR = Path(__file__).resolve().parent
# === 수정: 내부 서버 포트로 변경 ===
WS_URL = "ws://localhost:8766"  # websocket.py의 내부 서버로 연결

import logging, sys, os, datetime
BASE_DIR = Path(__file__).resolve().parent

LOG_FILE = "/tmp/eye_tracking_optimized.log"  # /tmp는 항상 쓰기 가능
try:
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
except PermissionError:
    # 권한 문제 시 콘솔에만 출력
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    print(f"⚠ 로그 파일 생성 실패, 콘솔 출력만 사용: {LOG_FILE}")

def dbg(msg):
    # 콘솔 + 파일 모두 찍고 즉시 flush
    print(msg, flush=True)
    logging.debug(msg)

dbg("=== [BOOT] eye_tracking_worker.py start ===")
dbg(f"WS_URL={WS_URL}")
dbg(f"PWD={os.getcwd()}, USER={os.getenv('USER')}, DISPLAY={os.getenv('DISPLAY')}")


async def ws_receiver():
    global toggle_mouse_requested, eye_calib_requested, force_mouse_on, fist_enabled
    while True:
        try:
            dbg("[WS] trying to connect...")
            async with websockets.connect(WS_URL) as ws:
                dbg("[WS] connected")
                while True:
                    raw = await ws.recv()
                    dbg(f"[WS] recv raw={raw}")
                    try:
                        data = json.loads(raw)
                        msg_type = data.get("type")
                    except Exception:
                        continue
                    dbg(f"[WS] parsed type={msg_type}")

                    if msg_type == "EYE_CALIB_ON":
                        eye_calib_requested = True
                        dbg("[WS] → eye_calib_requested=True")
                        print("[WS] EYE_CALIB_ON → eye_calib_requested=True")

                    elif msg_type == "EYE_ORDER_ON":
                        # 주먹 인식 끄고 (요구5), 마우스 제어는 계속
                        fist_enabled = False
                        force_mouse_on = True
                        dbg("[WS] → fist_enabled=False, force_mouse_on=True")
                        print("[WS] EYE_ORDER_ON → fist_enabled=False, force_mouse_on=True")

                    elif msg_type == "MOUSE_ON":
                        force_mouse_on = True
                        dbg("[WS] → force_mouse_on=True")
                        print("[WS] MOUSE_ON → force_mouse_on=True")
                    
                    # === 새로 추가: 모든 기능 정지 (CHAT_ORDER_ON, NORMAL_ORDER_ON, ALL_RESET 시) ===
                    elif msg_type == "STOP_ALL":
                        fist_enabled = False
                        force_mouse_on = False
                        dbg("[WS] → STOP_ALL: fist_enabled=False, force_mouse_on=False")
                        print("[WS] STOP_ALL → 모든 기능 비활성화")
                        
                    elif msg_type == "TOUCH_ACTIVE":
                        touch_active = True
                        dbg("[WS] TOUCH_ACTIVE → touch_active=True")

                    elif msg_type == "TOUCH_IDLE":
                        touch_active = False
                        last_touch_end = time.time()
                        dbg("[WS] TOUCH_IDLE → touch_active=False, last_touch_end updated")

                        
        except Exception as e:
            dbg(f"[WS] connect failed/disconnected: {e} (retry in 2s)")
            print(f"[WS] 연결 끊김/실패: {e}. 2초 후 재시도...")
            await asyncio.sleep(2)

def _start_ws_client_in_background():
    def runner():
        asyncio.run(ws_receiver())
    t = threading.Thread(target=runner, daemon=True)
    t.start()

_start_ws_client_in_background()
dbg("[WS] receiver thread starting...")

# === 수정: 주먹 감지 시 FIST_DETECTED 전송 ===
async def send_fist_detected():
    """주먹 감지 시 메인 허브로 FIST_DETECTED 전송"""
    try:
        async with websockets.connect(WS_URL) as ws:
            payload = {"type": "FIST_DETECTED"}
            await ws.send(json.dumps(payload, ensure_ascii=False))
            await asyncio.sleep(0.05)
            dbg("[Exit] FIST_DETECTED sent to hub")
    except Exception as e:
        print(f"[WS] 전송 실패: {e}")

def run_tts_stt_process():
    """요구7: 주먹 감지 시 tts_stt.py 실행."""
    script = str(BASE_DIR / "tts_stt.py")
    try:
        subprocess.Popen(["python3", script],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        print("[PROC] tts_stt.py started")
    except Exception as e:
        print(f"[PROC] tts_stt.py start failed: {e}")

# === 수정: 주먹 감지 시 동작 변경 (프로세스 종료하지 않고 메시지만 전송) ===
def notify_fist_detected():
    """주먹 감지 시 메인 허브로 알림만 전송 (프로세스는 계속 실행)"""
    dbg("[Fist] sending FIST_DETECTED to hub")
    try:
        asyncio.run(send_fist_detected())
        dbg("[Fist] FIST_DETECTED sent")
        print("[Fist] ✓ FIST_DETECTED 전송 완료")
    except Exception as e:
        dbg(f"[Fist] send_fist_detected error: {e}")
        print(f"[WS] asyncio 전송 오류: {e}")
    # === 수정: 프로세스 종료하지 않음 (주석 처리) ===
    # dbg("[Exit] releasing camera & destroying windows, then sys.exit(0)")
    # try:
    #     cap.release()
    # except:
    #     pass
    # try:
    #     cv2.destroyAllWindows()
    # except:
    #     pass
    # sys.exit(0)

# 캐시
last_head_center = None
last_R_final = None
last_nose_points_3d = None
last_iris_3d_left = None
last_iris_3d_right = None

# 성능 측정
last_ts = None
fps_ema = None

# Face Detection 설정
last_face_bbox = None
last_face_time = 0.0

# 최적화된 파라미터
FACEMESH_EVERY = 3
FACE_TTL = 3.0
DETECT_EVERY = 4
ROI_MARGIN = 0.25
ALLOW_FALLBACK_FULLFRAME = True

# Hand detection 최적화
HAND_EVERY = 4
HAND_ROI_SCALE = 1.5

# Monitor parameters
USER_MONITOR_DISTANCE = 40.0
MONITOR_WIDTH_CM = 39.6
MONITOR_HEIGHT_CM = 19.42

# Screen setup
MONITOR_WIDTH_PX, MONITOR_HEIGHT_PX = pyautogui.size()
CENTER_X = MONITOR_WIDTH_PX // 2
CENTER_Y = MONITOR_HEIGHT_PX // 2
mouse_control_enabled = False
filter_length = 8
gaze_length = 350

# 3D monitor state
monitor_corners = None
monitor_center_w = None
monitor_normal_w = None
units_per_cm = None

# Mouse target
mouse_target = [CENTER_X, CENTER_Y]
mouse_lock = threading.Lock()

# Calibration
calibration_offset_yaw = 0
calibration_offset_pitch = 0

# Smoothing buffer
combined_gaze_directions = deque(maxlen=filter_length)

# Reference matrices
R_ref_nose = [None]

# 디버깅 플래그
DEBUG = False

# MediaPipe Face Detection (빠른 얼굴 검출용)
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

# MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

hand_last_state = False
last_fist_time = 0.0
FIST_COOLDOWN = 0.6

def _lm_xy(hand_landmarks, idx, w, h):
    lm = hand_landmarks.landmark[idx]
    return np.array([lm.x * w, lm.y * h], dtype=float)

def is_finger_curled(hand_landmarks, tip_idx, pip_idx, wrist_idx, w, h):
    tip = _lm_xy(hand_landmarks, tip_idx, w, h)
    pip = _lm_xy(hand_landmarks, pip_idx, w, h)
    wrist = _lm_xy(hand_landmarks, wrist_idx, w, h)
    # wrist = _lm_xy(hand_landmarks, 0, w, h)
    return np.linalg.norm(tip - wrist) < np.linalg.norm(pip - wrist)

def is_thumb_curled(hand_landmarks, w, h):
    wrist = _lm_xy(hand_landmarks, 0, w, h)
    tip = _lm_xy(hand_landmarks, 4, w, h)
    mcp = _lm_xy(hand_landmarks, 2, w, h)
    return np.linalg.norm(tip - wrist) < np.linalg.norm(mcp - wrist)

def is_fist(hand_landmarks, w, h):
    curled = 0
    curled += int(is_finger_curled(hand_landmarks, 8, 6, 0, w, h))
    curled += int(is_finger_curled(hand_landmarks, 12, 10, 0, w, h))
    curled += int(is_finger_curled(hand_landmarks, 16, 14, 0, w, h))
    curled += int(is_finger_curled(hand_landmarks, 20, 18, 0, w, h))
    curled += int(is_thumb_curled(hand_landmarks, w, h))
    return curled >= 4

def mediapipe_face_detect(frame_bgr):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)
    bboxes = []
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            score = detection.score[0]
            x0 = int(bbox.xmin * w)
            y0 = int(bbox.ymin * h)
            x1 = int((bbox.xmin + bbox.width) * w)
            y1 = int((bbox.ymin + bbox.height) * h)
            x0 = max(0, min(x0, w-1))
            y0 = max(0, min(y0, h-1))
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            if x1 > x0 and y1 > y0:
                bboxes.append((x0, y0, x1, y1, score))
    return bboxes

# Camera setup
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
dbg(f"[Camera] open={cap.isOpened()}")
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"[Camera] Resolution: {w}x{h}")
print(f"[Face Detection] Using MediaPipe (CPU-optimized)")
dbg(f"[Camera] set resolution: {w}x{h}, fps={cap.get(cv2.CAP_PROP_FPS)}")

frame_count = 0

nose_indices = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241, 
                461, 125, 354, 218, 438, 195, 167, 393, 165, 391,
                3, 248]

screen_position_file = str(BASE_DIR / "screen_position.txt")

def write_screen_position(x, y):
    try:
        with open(screen_position_file, 'w') as f:
            f.write(f"{x},{y}\n")
    except Exception:
        pass

def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def create_monitor_plane(head_center, R_final, face_landmarks, w, h, 
                         forward_hint=None, gaze_origin=None, gaze_dir=None):
    try:
        lm_chin = face_landmarks[152]
        lm_fore = face_landmarks[10]
        chin_w = np.array([lm_chin.x * w, lm_chin.y * h, lm_chin.z * w], dtype=float)
        fore_w = np.array([lm_fore.x * w, lm_fore.y * h, lm_fore.z * w], dtype=float)
        face_h_units = np.linalg.norm(fore_w - chin_w)
        upc = face_h_units / 15.0
    except:
        upc = 5.0
    
    dist_cm = USER_MONITOR_DISTANCE
    mon_w_cm, mon_h_cm = MONITOR_WIDTH_CM, MONITOR_HEIGHT_CM
    half_w = (mon_w_cm * 0.5) * upc
    half_h = (mon_h_cm * 0.5) * upc

    head_forward = -R_final[:, 2]
    if forward_hint is not None:
        head_forward = forward_hint / np.linalg.norm(forward_hint)

    if gaze_origin is not None and gaze_dir is not None:
        gaze_dir = gaze_dir / np.linalg.norm(gaze_dir)
        plane_point = head_center + head_forward * (USER_MONITOR_DISTANCE * upc)
        plane_normal = head_forward
        denom = np.dot(plane_normal, gaze_dir)
        if abs(denom) > 1e-6:
            t = np.dot(plane_normal, plane_point - gaze_origin) / denom
            center_w = gaze_origin + t * gaze_dir
        else:
            center_w = head_center + head_forward * (USER_MONITOR_DISTANCE * upc)
    else:
        center_w = head_center + head_forward * (USER_MONITOR_DISTANCE * upc)

    world_up = np.array([0, -1, 0], dtype=float)
    head_right = np.cross(world_up, head_forward); head_right /= np.linalg.norm(head_right)
    head_up = np.cross(head_forward, head_right); head_up /= np.linalg.norm(head_up)

    p0 = center_w - head_right * half_w - head_up * half_h
    p1 = center_w + head_right * half_w - head_up * half_h
    p2 = center_w + head_right * half_w + head_up * half_h
    p3 = center_w - head_right * half_w + head_up * half_h

    normal_w = head_forward / (np.linalg.norm(head_forward) + 1e-9)
    return [p0, p1, p2, p3], center_w, normal_w, upc

def expand_and_clip_bbox(b, margin, w, h):
    x0, y0, x1, y1 = b
    bw = x1 - x0
    bh = y1 - y0
    mx = int(bw * margin)
    my = int(bh * margin)
    x0 = max(0, x0 - mx)
    y0 = max(0, y0 - my)
    x1 = min(w-1, x1 + mx)
    y1 = min(h-1, y1 + my)
    return x0, y0, x1, y1

def to_global_landmarks(landmarks, roi, full_w, full_h):
    x0, y0, x1, y1 = roi
    cw = x1 - x0
    ch = y1 - y0
    out = []
    for lm in landmarks:
        gx = (x0 + lm.x * cw) / float(full_w)
        gy = (y0 + lm.y * ch) / float(full_h)
        gz = lm.z
        out.append(SimpleNamespace(x=gx, y=gy, z=gz))
    return out

def compute_scale(points_3d):
    n = len(points_3d)
    total = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(points_3d[i] - points_3d[j])
            total += dist
            count += 1
    return total / count if count > 0 else 1.0

def compute_coordinate_box(face_landmarks, indices, ref_matrix_container, w, h):
    points_3d = np.array([
        [face_landmarks[i].x * w, face_landmarks[i].y * h, face_landmarks[i].z * w]
        for i in indices
    ])
    center = np.mean(points_3d, axis=0)
    centered = points_3d - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:, np.argsort(-eigvals)]
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1
    r = Rscipy.from_matrix(eigvecs)
    roll, pitch, yaw = r.as_euler('zyx', degrees=False)
    R_final = Rscipy.from_euler('zyx', [roll, pitch, yaw]).as_matrix()
    if ref_matrix_container[0] is None:
        ref_matrix_container[0] = R_final.copy()
    else:
        R_ref = ref_matrix_container[0]
        for i in range(3):
            if np.dot(R_final[:, i], R_ref[:, i]) < 0:
                R_final[:, i] *= -1
    return center, R_final, points_3d

def convert_gaze_to_screen_coordinates(combined_gaze_direction, calibration_offset_yaw, calibration_offset_pitch):
    reference_forward = np.array([0, 0, -1])
    avg_direction = combined_gaze_direction / np.linalg.norm(combined_gaze_direction)
    xz_proj = np.array([avg_direction[0], 0, avg_direction[2]]); xz_proj /= np.linalg.norm(xz_proj)
    yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz_proj), -1.0, 1.0))
    if avg_direction[0] < 0:
        yaw_rad = -yaw_rad
    yz_proj = np.array([0, avg_direction[1], avg_direction[2]]); yz_proj /= np.linalg.norm(yz_proj)
    pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz_proj), -1.0, 1.0))
    if avg_direction[1] > 0:
        pitch_rad = -pitch_rad
    yaw_deg = math.degrees(yaw_rad); pitch_deg = math.degrees(pitch_rad)
    if yaw_deg < 0: yaw_deg = -yaw_deg
    elif yaw_deg > 0: yaw_deg = -yaw_deg
    raw_yaw_deg = yaw_deg; raw_pitch_deg = pitch_deg
    yawDegrees = 15; pitchDegrees = 5
    yaw_deg += calibration_offset_yaw; pitch_deg += calibration_offset_pitch
    screen_x = int(((yaw_deg + yawDegrees) / (2 * yawDegrees)) * MONITOR_WIDTH_PX)
    screen_y = int(((pitchDegrees - pitch_deg) / (2 * pitchDegrees)) * MONITOR_HEIGHT_PX)
    screen_x = MONITOR_WIDTH_PX - screen_x  # 좌우반전
    screen_x = max(10, min(screen_x, MONITOR_WIDTH_PX - 10))
    screen_y = max(10, min(screen_y, MONITOR_HEIGHT_PX - 10))
    return screen_x, screen_y, raw_yaw_deg, raw_pitch_deg

def mouse_mover():
    dbg("[Mouse] mover thread start")
    last_xy = (0, 0)  # === 수정: 초기화 추가 ===
    
    while True:
        if touch_active or (time.time() - last_touch_end < TOUCH_HOLDOFF):
            time.sleep(0.01)
            continue

        with mouse_lock:
            xy = (mouse_target[0], mouse_target[1])
        
        if mouse_control_enabled:
            with mouse_lock:
                xy = (mouse_target[0], mouse_target[1])  # === 수정: 튜플로 변경 ===
            if xy != last_xy:
                dbg(f"[Mouse] moveTo {xy}")
                last_xy = xy
            pyautogui.moveTo(xy[0], xy[1])  # === 수정: 튜플 언팩 ===
        time.sleep(0.01)

threading.Thread(target=mouse_mover, daemon=True).start()

# Eye sphere tracking
left_sphere_locked = False
left_sphere_local_offset = None
left_calibration_nose_scale = None

right_sphere_locked = False
right_sphere_local_offset = None
right_calibration_nose_scale = None

base_radius = 20

print("[Info] 'c'=calibrate, 's'=screen center, F7=mouse, 'q'=quit")
print(f"[Settings] FaceMesh every {FACEMESH_EVERY}, Hand every {HAND_EVERY}, Face Det every {DETECT_EVERY}")

# 성능 모니터링
perf_counters = defaultdict(list)
last_perf_print = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        dbg("[Frame] read failed!")
        print("[ERROR] Failed to read frame!")
        break
    
    now_f = time.perf_counter()
    if last_ts is not None:
        inst_fps = 1.0 / max(1e-6, (now_f - last_ts))
        fps_ema = inst_fps if fps_ema is None else (fps_ema * 0.85 + inst_fps * 0.15)
    last_ts = now_f

    frame_count += 1
    now = time.time()

    # MediaPipe Face Detection
    if frame_count % DETECT_EVERY == 0:
        dets = mediapipe_face_detect(frame)
        dbg(f"[FaceDet] dets={len(dets) if dets else 0}")

        if dets:
            dets.sort(key=lambda x: x[4], reverse=True)
            x0, y0, x1, y1, sc = dets[0]
            last_face_bbox = (x0, y0, x1, y1)
            last_face_time = now

    # ROI 결정
    roi = (0, 0, w, h)
    run_facemesh = False
    frame_rgb = None

    roi_valid = (last_face_bbox is not None) and (now - last_face_time <= FACE_TTL)

    if roi_valid:
        x0, y0, x1, y1 = expand_and_clip_bbox(last_face_bbox, ROI_MARGIN, w, h)
        roi = (x0, y0, x1, y1)
        roi_bgr = frame[y0:y1, x0:x1]
        frame_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        run_facemesh = (frame_count % FACEMESH_EVERY == 0)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
    elif ALLOW_FALLBACK_FULLFRAME:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        run_facemesh = (frame_count % FACEMESH_EVERY == 0)

    # Hand detection (요구5: fist_enabled가 False면 스킵)
    if fist_enabled and (frame_count % HAND_EVERY == 0):
        if roi_valid:
            hx0, hy0, hx1, hy1 = expand_and_clip_bbox(last_face_bbox, HAND_ROI_SCALE, w, h)
            hand_roi_bgr = frame[hy0:hy1, hx0:hx1]
            hands_rgb = cv2.cvtColor(hand_roi_bgr, cv2.COLOR_BGR2RGB)
            hand_w, hand_h = hx1 - hx0, hy1 - hy0
        else:
            hands_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_w, hand_h = w, h
        
        hand_results = hands.process(hands_rgb)

        curr_fist = False
        if hand_results.multi_hand_landmarks:
            for hlm in hand_results.multi_hand_landmarks:
                if is_fist(hlm, hand_w, hand_h):
                    curr_fist = True

        dbg(f"[Hand] fist={curr_fist}, last_state={hand_last_state}")
        # === 수정: notify_frontend_and_exit → notify_fist_detected (프로세스 종료하지 않음) ===
        if curr_fist and (not hand_last_state) and (now - last_fist_time > FIST_COOLDOWN):
            last_fist_time = now
            dbg("[Hand] FIST TRIGGER -> notify_fist_detected()")
            print("[Hand] FIST TRIGGER -> Sending FIST_DETECTED to hub")
            cv2.putText(frame, "FIST -> Sending to Hub", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            notify_fist_detected()  # === 메시지만 전송, 프로세스는 계속 실행 ===
        hand_last_state = curr_fist

    # FaceMesh 실행
    results = None
    if run_facemesh and frame_rgb is not None:
        dbg("[FaceMesh] processing...")
        results = face_mesh.process(frame_rgb)
        dbg(f"[FaceMesh] result faces={len(results.multi_face_landmarks) if (results and results.multi_face_landmarks) else 0}")

    # 결과 처리
    face_landmarks = None
    if results and results.multi_face_landmarks:
        raw_lms = results.multi_face_landmarks[0].landmark
        face_landmarks = to_global_landmarks(raw_lms, roi, w, h)

        head_center, R_final, nose_points_3d = compute_coordinate_box(
            face_landmarks, nose_indices, R_ref_nose, w, h
        )

        left_iris_idx = 468
        right_iris_idx = 473
        l = face_landmarks[left_iris_idx]
        r = face_landmarks[right_iris_idx]
        iris_3d_left = np.array([l.x * w, l.y * h, l.z * w], dtype=float)
        iris_3d_right = np.array([r.x * w, r.y * h, r.z * w], dtype=float)

        last_head_center = head_center.copy()
        last_R_final = R_final.copy()
        last_nose_points_3d = nose_points_3d.copy()
        last_iris_3d_left = iris_3d_left.copy()
        last_iris_3d_right = iris_3d_right.copy()
        
        if not sent_ready:
            try:
                asyncio.run(send_internal({"type": "EYE_READY"}))
            except Exception as e:
                dbg(f"[WS] EYE_READY send error: {e}")
            sent_ready = True

    # Eye sphere tracking & gaze
    if last_head_center is not None and last_iris_3d_left is not None:
        if left_sphere_locked:
            current_nose_scale = compute_scale(last_nose_points_3d)
            scale_ratio = current_nose_scale / left_calibration_nose_scale if left_calibration_nose_scale else 1.0
            scaled_offset = left_sphere_local_offset * scale_ratio
            sphere_world_l = last_head_center + last_R_final @ scaled_offset

        if right_sphere_locked:
            current_nose_scale = compute_scale(last_nose_points_3d)
            scale_ratio_r = current_nose_scale / right_calibration_nose_scale if right_calibration_nose_scale else 1.0
            scaled_offset_r = right_sphere_local_offset * scale_ratio_r
            sphere_world_r = last_head_center + last_R_final @ scaled_offset_r

        if left_sphere_locked and right_sphere_locked:
            left_gaze_dir = last_iris_3d_left - sphere_world_l; left_gaze_dir /= np.linalg.norm(left_gaze_dir)
            right_gaze_dir = last_iris_3d_right - sphere_world_r; right_gaze_dir /= np.linalg.norm(right_gaze_dir)
            raw_combined_direction = (left_gaze_dir + right_gaze_dir) / 2; raw_combined_direction /= np.linalg.norm(raw_combined_direction)
            combined_gaze_directions.append(raw_combined_direction)
            avg_combined_direction = np.mean(combined_gaze_directions, axis=0); avg_combined_direction /= np.linalg.norm(avg_combined_direction)

            screen_x, screen_y, raw_yaw, raw_pitch = convert_gaze_to_screen_coordinates(
                avg_combined_direction, calibration_offset_yaw, calibration_offset_pitch
            )
            dbg(f"[Gaze] screen=({screen_x},{screen_y}) raw_yaw={raw_yaw:.2f} raw_pitch={raw_pitch:.2f}")

            # 강제 마우스 ON 명령 처리
            if force_mouse_on:
                mouse_control_enabled = True
                dbg("[Mouse] ON via MOUSE_ON")

            if mouse_control_enabled:
                # 터치 중이거나, 터치 종료 후 홀드오프 동안은 주입 금지
                if (not touch_active) and (time.time() - last_touch_end >= TOUCH_HOLDOFF):
                    with mouse_lock:
                        mouse_target[0] = screen_x
                        mouse_target[1] = screen_y

            write_screen_position(screen_x, screen_y)
            cv2.putText(frame, f"Screen: ({screen_x}, {screen_y})", (10, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # FPS 및 상태 표시
    status_text = []
    if fps_ema:
        status_text.append(f"FPS: {fps_ema:.1f}")
    status_text.append("Face: OK" if last_face_bbox else "Face: NONE")
    status_text.append("Mesh: OK" if last_head_center is not None else "Mesh: NONE")
    if left_sphere_locked and right_sphere_locked:
        status_text.append("Calib: OK")
    if not fist_enabled:
        status_text.append("Fist: OFF")

    for i, text in enumerate(status_text):
        color = (0, 255, 0) if "OK" in text or "OFF" in text else (0, 0, 255)
        cv2.putText(frame, text, (10, 30 + i*25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Eye Tracking", frame)

    # --- Keyboard/WS 제어 ---
    # 0) 강제 마우스 ON 신호가 오면 즉시 반영
    if force_mouse_on and not mouse_control_enabled:
        mouse_control_enabled = True
        print("[Mouse] ON (via WS MOUSE_ON)")

    # 1) 토글 신호(WS)
    if toggle_mouse_requested:
        mouse_control_enabled = not mouse_control_enabled
        dbg(f"[Mouse] {'ON' if mouse_control_enabled else 'OFF'} via WS toggle")
        print(f"[Mouse] {'ON' if mouse_control_enabled else 'OFF'}  (via WS toggle)")
        toggle_mouse_requested = False
        time.sleep(0.1)

    # 2) 키보드(F7) 토글 (fallback)
    try:
        if keyboard.is_pressed('f7'):
            mouse_control_enabled = not mouse_control_enabled
            dbg(f"[Mouse] {'ON' if mouse_control_enabled else 'OFF'} via keyboard F7")
            print(f"[Mouse] {'ON' if mouse_control_enabled else 'OFF'} (via keyboard)")
            time.sleep(0.3)
    except Exception as e:  # === 수정: 변수명 추가 ===
        dbg(f"[Keyboard] keyboard.is_pressed error: {e}")
        pass

    # 3) 키 입력 먼저 읽고, 그 다음 c_pressed 계산(버그 수정 포인트)
    key = cv2.waitKey(1) & 0xFF
    # ⚠️ 여기서 'eye_calib_requested'는 메쉬 있을 때만 소비되게 바꿈
    c_pressed = (key == ord('c')) or (eye_calib_requested and (last_head_center is not None))

    if key == ord('q'):
        dbg("[Key] q → quit")
        break
    elif c_pressed and not (left_sphere_locked and right_sphere_locked):
        dbg(f"[Calib] trigger. have_mesh={last_head_center is not None}")
        if last_head_center is None:
            dbg("[Calib] ✗ No face mesh data")
            print("[Calib] ✗ No face mesh data - wait for face detection")
            pass
        else:
            current_nose_scale = compute_scale(last_nose_points_3d)
            # LEFT
            left_sphere_local_offset = last_R_final.T @ (last_iris_3d_left - last_head_center)
            camera_dir_world = np.array([0, 0, 1], dtype=float)
            camera_dir_local = last_R_final.T @ camera_dir_world
            left_sphere_local_offset += base_radius * camera_dir_local
            left_calibration_nose_scale = current_nose_scale
            # RIGHT
            right_sphere_local_offset = last_R_final.T @ (last_iris_3d_right - last_head_center)
            right_sphere_local_offset += base_radius * camera_dir_local
            right_calibration_nose_scale = current_nose_scale
            left_sphere_locked = True
            right_sphere_locked = True

            # Monitor plane
            sphere_world_l_calib = last_head_center + last_R_final @ left_sphere_local_offset
            sphere_world_r_calib = last_head_center + last_R_final @ right_sphere_local_offset
            left_dir = last_iris_3d_left - sphere_world_l_calib
            right_dir = last_iris_3d_right - sphere_world_r_calib
            if np.linalg.norm(left_dir) > 1e-9: left_dir /= np.linalg.norm(left_dir)
            if np.linalg.norm(right_dir) > 1e-9: right_dir /= np.linalg.norm(right_dir)
            forward_hint = (left_dir + right_dir) * 0.5
            if np.linalg.norm(forward_hint) > 1e-9:
                forward_hint /= np.linalg.norm(forward_hint)
            else:
                forward_hint = None
            gaze_origin = (sphere_world_l_calib + sphere_world_r_calib) / 2
            gaze_dir = forward_hint
            monitor_corners, monitor_center_w, monitor_normal_w, units_per_cm = create_monitor_plane(
                last_head_center, last_R_final, face_landmarks, w, h,
                forward_hint=forward_hint,
                gaze_origin=gaze_origin,
                gaze_dir=gaze_dir
            )
            dbg("[Calib] ✓ Complete (left/right locked)")
            print("[Calibration] ✓ Complete")

            if eye_calib_requested:
                dbg("[Calib] consumed eye_calib_requested=True → False")
                eye_calib_requested = False

    elif key == ord('s') and left_sphere_locked and right_sphere_locked:
        if last_head_center is None:
            print("[Screen Calib] ✗ No face")
        else:
            current_nose_scale = compute_scale(last_nose_points_3d)
            scale_ratio_l = current_nose_scale / left_calibration_nose_scale if left_calibration_nose_scale else 1.0
            scale_ratio_r = current_nose_scale / right_calibration_nose_scale if right_calibration_nose_scale else 1.0
            sphere_world_l_now = last_head_center + last_R_final @ (left_sphere_local_offset * scale_ratio_l)
            sphere_world_r_now = last_head_center + last_R_final @ (right_sphere_local_offset * scale_ratio_r)
            left_gaze_dir = last_iris_3d_left - sphere_world_l_now
            right_gaze_dir = last_iris_3d_right - sphere_world_r_now
            if np.linalg.norm(left_gaze_dir) > 1e-9: left_gaze_dir /= np.linalg.norm(left_gaze_dir)
            if np.linalg.norm(right_gaze_dir) > 1e-9: right_gaze_dir /= np.linalg.norm(right_gaze_dir)
            current_combined_direction = (left_gaze_dir + right_gaze_dir) / 2.0
            if np.linalg.norm(current_combined_direction) > 1e-9:
                current_combined_direction /= np.linalg.norm(current_combined_direction)
            _, _, raw_yaw, raw_pitch = convert_gaze_to_screen_coordinates(
                current_combined_direction, 0, 0
            )
            calibration_offset_yaw = -raw_yaw
            calibration_offset_pitch = -raw_pitch
            print(f"[Screen Calibrated] ✓ Yaw: {calibration_offset_yaw:.2f}, Pitch: {calibration_offset_pitch:.2f}")

dbg("[Shutdown] releasing resources")
cap.release()
cv2.destroyAllWindows()
dbg("=== [BOOT] eye_tracking_worker.py end ===")