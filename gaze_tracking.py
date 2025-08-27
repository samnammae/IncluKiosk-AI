# object_eyecontrol.py (revised: GazeTracking + center face crop + warmup & calibration)
import cv2
import pyautogui
import time

# (선택) 손 시각화를 계속 쓰고 싶다면 유지
# import mediapipe as mp
from gaze_tracking import GazeTracking

# ── 마우스/화면 설정 ─────────────────────────────────────────────
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
SCREEN_W, SCREEN_H = pyautogui.size()

# ── 튜닝 파라미터 ────────────────────────────────────────────────
MIRRORED = True      # 웹캠 미러링(좌우 반전) 보정
ALPHA = 0.25         # EMA 스무딩(0~1, 높을수록 빠르게/덜 부드럽게)
DEAD = 0.02          # 데드존(비율 기준), 미세 흔들림 무시
EDGE_MARGIN = 0.00   # 0~0.10 권장, 가장자리로 못 가면 줄이기
DOUBLE_BLINK_WINDOW = 1.0  # 초, 이 시간 내 2회 깜빡이면 클릭
FRAME_WIDTH, FRAME_HEIGHT = 640, 480  # 필요시 해상도 조정

# ▶ 추가: 중앙 얼굴 크롭 설정
ENABLE_CENTER_CROP = True
FACE_RATIO = 0.6  # 0~1 사이. 0.6이면 프레임의 60% 크기 정사각형을 중앙에서 크롭

# ── (선택) MediaPipe Hands 초기화 ────────────────────────────────
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

# ── GazeTracking 초기화 ──────────────────────────────────────────
gaze = GazeTracking()
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# ▶ 컬러 프레임 유도를 위한 FOURCC (M-JPEG 권장)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
# 지연 줄이고 싶으면 해상도 낮추기
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# ── 상태 변수 ────────────────────────────────────────────────────
mx, my = SCREEN_W // 2, SCREEN_H // 2   # 커서의 현재 좌표(중앙 시작)
pyautogui.moveTo(mx, my)                # 시작 시 커서 중앙으로 이동
last_blinks = []                        # 최근 깜빡임 타임스탬프

# ▶ 추가: 워밍업/캘리브레이션 상태
frame_i = 0
WARMUP_FRAMES = 30        # 시작 1초 정도(30fps 가정) 마우스 잠금
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

    # === 프레임 180도 회전 === <--임시!
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
    # 가벼운 표시: annotated_frame() 대신 ROI 자체를 사용해도 됨
    vis = roi  # gaze.annotated_frame() 써도 되지만 속도는 ROI 직접 표시가 더 낫다

    # ── (선택) 손 시각화 오버레이 ───────────────────────────────
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # hand_results = hands.process(frame_rgb)
    # if hand_results.multi_hand_landmarks:
    #     for hand_landmarks in hand_results.multi_hand_landmarks:
    #         mp_drawing.draw_landmarks(vis, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ── 시선 비율 읽기(0~1) → 화면 좌표 매핑 ────────────────────
    hx = gaze.horizontal_ratio()
    vy = gaze.vertical_ratio()

    if hx is not None and vy is not None:
        # ▶ 캘리브레이션(한 번만): 워밍업이 끝나면 그때의 시선을 중앙(0.5, 0.5)으로 보정
        if not calibrated and frame_i > WARMUP_FRAMES:
            bias_x = 0.5 - hx
            bias_y = 0.5 - vy
            calibrated = True

        # ▶ 보정 적용 + 경계 클램프
        hx = min(max(hx + bias_x, 0.0), 1.0)
        vy = min(max(vy + bias_y, 0.0), 1.0)

        # 좌우 반전 카메라면 보정
        if MIRRORED:
            hx = 1.0 - hx

        # 화면 가장자리 여유(EDGE_MARGIN) 적용
        target_x = (EDGE_MARGIN + hx * (1 - 2 * EDGE_MARGIN)) * SCREEN_W
        target_y = (EDGE_MARGIN + vy * (1 - 2 * EDGE_MARGIN)) * SCREEN_H

        # ── 워밍업 동안 커서 고정(중앙 유지) ─────────────────────
        if frame_i <= WARMUP_FRAMES:
            pass
        else:
            # 초기엔 바로 세팅, 이후엔 데드존/EMA 스무딩 적용
            if mx is None or my is None:
                mx, my = target_x, target_y
            else:
                if abs(target_x - mx) < DEAD * SCREEN_W and abs(target_y - my) < DEAD * SCREEN_H:
                    pass  # 변화 작으면 유지
                else:
                    mx = ALPHA * target_x + (1 - ALPHA) * mx
                    my = ALPHA * target_y + (1 - ALPHA) * my

            # 화면 경계 클램프 & 실제 이동
            mx = clamp(mx, 0, SCREEN_W - 1)
            my = clamp(my, 0, SCREEN_H - 1)
            pyautogui.moveTo(mx, my)

        # 디버그 텍스트(ROI 크기/상태 표기)
        info = (
            f"crop={roi.shape[1]}x{roi.shape[0]} "
            f"hx={hx:.3f}, vy={vy:.3f}  mouse=({int(mx)},{int(my)}) "
            f"[warmup:{frame_i<=WARMUP_FRAMES} cal:{calibrated}]"
        )
        cv2.putText(vis, info, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 180, 50), 1, cv2.LINE_AA)

    # # ── 더블-블링크 클릭(예: 눈 두 번 빠르게 감기면 클릭) ─────────
    # if gaze.is_blinking():
    #     now = time.time()
    #     # 최근 기록 중 오래된 것 제거
    #     last_blinks = [t for t in last_blinks if now - t <= DOUBLE_BLINK_WINDOW]
    #     # 직전 프레임에서 뜬 상태였다가 이번 프레임에서 감김을 감지하려면
    #     # 단순화를 위해 프레임 기반 중복 방지 없이 타임스탬프 추가
    #     if not last_blinks or (now - last_blinks[-1]) > 0.12:  # 과도한 중복 방지(약간의 간격)
    #         last_blinks.append(now)
    #     if len(last_blinks) >= 2 and (last_blinks[-1] - last_blinks[-2]) <= DOUBLE_BLINK_WINDOW:
    #         pyautogui.click()
    #         last_blinks.clear()  # 다음 감지를 위해 초기화
    #         cv2.putText(vis, "CLICK (double blink)", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 200, 255), 2)

    # ── 화면 출력 ────────────────────────────────────────────────
    cv2.imshow("Eye Control (GazeTracking - Center Crop)", vis)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()