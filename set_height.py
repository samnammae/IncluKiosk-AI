# face_height_guide.py
import cv2
import time
import mediapipe as mp

# ====== 설정 ======
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480        # RPi면 640x480 또는 480x360 권장
FDETECT_MODEL = 0                  # 0: short-range(가까운 얼굴), 1: full-range
MIN_DET_CONF = 0.6
DEADBAND_PCT = 0.04                # 중앙에서 ±4%면 '중앙'으로 간주
EMA_ALPHA = 0.3                    # 0.2~0.4 정도 추천
STABLE_FRAMES = 10                 # 중앙 유지 프레임 수(n)
PRINT_EVERY = 0.15                 # 메시지 너무 자주 안 찍게(초)
# ===================

mp_face = mp.solutions.face_detection

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    face_det = mp_face.FaceDetection(
        model_selection=FDETECT_MODEL,
        min_detection_confidence=MIN_DET_CONF
    )

    target_y = 0.5  # 화면 세로 중앙 (정규화 좌표)
    deadband = DEADBAND_PCT
    ema_y = None
    stable_count = 0
    last_print_t = 0
    last_state = None  # "up", "down", "center"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            # 성능을 위해 작게 리사이즈/수평반전 선택 (옵션)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_det.process(frame_rgb)

            state = None
            if result.detections:
                # 가장 큰 얼굴 하나만 사용(보통 0번째가 최대)
                det = result.detections[0]
                box = det.location_data.relative_bounding_box
                y_center = box.ymin + box.height * 0.5  # 0~1 정규화

                # EMA로 y 안정화
                if ema_y is None:
                    ema_y = y_center
                else:
                    ema_y = EMA_ALPHA * y_center + (1 - EMA_ALPHA) * ema_y

                # 중앙/위/아래 판정
                diff = ema_y - target_y
                if abs(diff) <= deadband:
                    state = "center"
                    stable_count += 1
                elif diff < 0:
                    state = "up"      # 얼굴이 아래쪽 -> 카메라를 올려야 함? (문구는 사용자에게 지시)
                    stable_count = 0
                else:
                    state = "down"    # 얼굴이 위쪽 -> 카메라를 내려야 함
                    stable_count = 0
            else:
                # 얼굴 없음
                state = "no-face"
                stable_count = 0

            # 너무 자주 안 찍게 쿨다운
            now = time.time()
            if now - last_print_t >= PRINT_EVERY:
                if state == "center":
                    if stable_count >= STABLE_FRAMES:
                        if last_state != "center":
                            print("Centered ✅ (stable)")
                    else:
                        print(f"Centered… ({stable_count}/{STABLE_FRAMES})")
                elif state == "up":
                    print("Go up!")     # 얼굴이 화면 아래 -> 카메라를 위로 이동
                elif state == "down":
                    print("Go down!")   # 얼굴이 화면 위 -> 카메라를 아래로 이동
                elif state == "no-face":
                    print("No face detected. Move camera up/down to find a face…")
                last_print_t = now
                last_state = state

            # 디버그용 가이드 라인(원하면 주석 해제)
            # cv2.line(frame, (0, int(FRAME_H*0.5)), (FRAME_W, int(FRAME_H*0.5)), (0,255,0), 1)
            # if ema_y is not None:
            #     y_px = int(ema_y * FRAME_H)
            #     cv2.line(frame, (0, y_px), (FRAME_W, y_px), (255,0,0), 1)
            # cv2.imshow("guide", frame)
            # if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            #     break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        # cv2.destroyAllWindows()
        face_det.close()

if __name__ == "__main__":
    main()
