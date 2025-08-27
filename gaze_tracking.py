# object_eyecontrol.py (final tuned: stronger horizontal movement)
import cv2
import pyautogui
import time
from collections import deque

# (선택) 손 시각화를 계속 쓰고 싶다면 유지
# import mediapipe as mp
from gaze_tracking import GazeTracking

# ── 마우스/화면 설정 ─────────────────────────────────────────────
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
SCREEN_W, SCREEN_H = pyautogui.size()

# ── 튜닝 파라미터 ────────────────────────────────────────────────
MIRRORED = True
ALPHA_X = 0.50      # 가로 EMA: 더 빠르게 따라감 (기존 0.35 → 0.50)
ALPHA_Y = 0.12      # 세로 EMA(더 부드럽게)
DEAD_X  = 0.01      # 가로 데드존 축소 (기존 0.02 → 0.01)
DEAD_Y  = 0.04      # 세로 데드존 크게
GAIN_X  = 1.35      # ⬅ 가로 감도 확대(1.2~1.6 사이 튜닝)
GAIN_Y  = 0.85      # 세로 감도 축소(0.7~0.95 조절)
EDGE_MARGIN = 0.00
DOUBLE_BLINK_WINDOW = 1.0
FRAME_WIDTH, FRAME_HEIGHT = 640, 480

# ▶ 중앙 얼굴 크롭 설정
ENABLE_CENTER_CROP = True
FACE_RATIO = 0.4  # 0~1, 프레임 대비 정사각형 ROI 비율

# ▶ 수직 신호 미디안 필터
MEDIAN_WIN = 5     # 3/5/7 권장(홀수)

# ── (선택) MediaPipe Hands 초기화 ────────────────────────────────
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

# ── GazeTracking 초기화 ──────────────────────────────────────────
gaze = GazeTracking()
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# ▶ 컬러 프레임 유도를 위한 FOURCC (M-JPEG 권장)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# ── 상태 변수 ────────────────────────────────────────────────────
mx, my = SCREEN_W // 2, SCREEN_H // 2   # 커서 중앙 시작
pyautogui.moveTo(mx, my)
last_blinks = []                        # (클릭 비활성)
vy_hist = deque(maxlen=MEDIAN_WIN)

# ▶ 워밍업/캘리브레이션
frame_i = 0
WARMUP_FRAMES = 30        # 시작 ~1초 잠금(30fps 가정)
calibrated = False
bias_x = 0.0
bias_y = 0.0

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

print("Press 'q' to quit.")
while True:
    ok, frame = cap.read()
    if not ok:
        print("❌ 카메라 프레임을 불러올 수 없습니다.")
        break

    frame_i += 1

    # === 프레임 180도 회전 ===
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    # === 채널 보정: 단일 채널이면 BGR 3채널로 변환 ===
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[-1] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # === 중앙 얼굴 크롭(정사각형) ===
    roi = frame
    if ENABLE_CENTER_CROP:
        h, w = frame.shape[:2]
        side = int(min(w, h) * FACE_RATIO)
        cx, cy = w // 2, h // 2
        x0 = max(0, cx - side // 2)
        y0 = max(0, cy - side // 2)
        x1 = min(w, x0 + side)
        y1 = min(h, y0 + side)
        roi = frame[y0:y1, x0:x1]

    # ── 시선 분석 ────────────────────────────────────────────────
    gaze.refresh(roi)
    vis = roi  # 더 가볍게: annotated_frame() 대신 ROI 직접 표시

    # ── 시선 비율 읽기(0~1) → 화면 좌표 ─────────────────────────
    hx = gaze.horizontal_ratio()
    vy = gaze.vertical_ratio()

    if hx is not None and vy is not None:
        # ▶ 캘리브레이션(워밍업 끝나면 1회)
        if not calibrated and frame_i > WARMUP_FRAMES:
            bias_x = 0.5 - hx
            bias_y = 0.5 - vy
            calibrated = True

        # 바이어스 적용 + 0~1 클램프
        hx = min(max(hx + bias_x, 0.0), 1.0)
        vy = min(max(vy + bias_y, 0.0), 1.0)

        # ▶ 가로 신호: 중심 기준 선형 게인으로 확대
        dx = hx - 0.5
        hx = 0.5 + dx * GAIN_X
        hx = 0.0 if hx < 0.0 else 1.0 if hx > 1.0 else hx  # clamp

        # ▶ 세로 신호 안정화: 미디안 + 감도 축소
        vy_hist.append(vy)
        vy_med = sorted(vy_hist)[len(vy_hist)//2] if vy_hist else vy
        vy = 1.0 - (1.0 - vy_med) * GAIN_Y if vy_med >= 0.5 else vy_med * GAIN_Y

        # 좌우 반전 보정
        if MIRRORED:
            hx = 1.0 - hx

        # 화면 좌표 변환
        target_x = (EDGE_MARGIN + hx * (1 - 2 * EDGE_MARGIN)) * SCREEN_W
        target_y = (EDGE_MARGIN + vy * (1 - 2 * EDGE_MARGIN)) * SCREEN_H

        # ▶ 워밍업 동안 커서 고정
        if frame_i <= WARMUP_FRAMES:
            pass
        else:
            # 축별 데드존 + EMA
            if mx is None or my is None:
                mx, my = target_x, target_y
            else:
                if abs(target_x - mx) >= DEAD_X * SCREEN_W:
                    mx = ALPHA_X * target_x + (1 - ALPHA_X) * mx
                if abs(target_y - my) >= DEAD_Y * SCREEN_H:
                    my = ALPHA_Y * target_y + (1 - ALPHA_Y) * my

            # 경계 클램프 + 실제 이동
            mx = clamp(mx, 0, SCREEN_W - 1)
            my = clamp(my, 0, SCREEN_H - 1)
            pyautogui.moveTo(mx, my)

        # 디버그 텍스트
        info = (
            f"crop={roi.shape[1]}x{roi.shape[0]} "
            f"hx={hx:.3f} vy={vy:.3f} mouse=({int(mx)},{int(my)}) "
            f"[warmup:{frame_i<=WARMUP_FRAMES} cal:{calibrated}]"
        )
        cv2.putText(vis, info, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 180, 50), 1, cv2.LINE_AA)

    # ── 화면 출력 ────────────────────────────────────────────────
    cv2.imshow("Eye Control (GazeTracking - Center Crop)", vis)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
