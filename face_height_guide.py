# face_height_guide.py
import cv2
import time
import mediapipe as mp

# ====== 설정 ======
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480        # RPi면 640x480 또는 480x360 권장
FDETECT_MODEL = 0                  # 0: short-range(가까운 얼굴), 1: full-range
MIN_DET_CONF = 0.6
DEADBAND_PCT = 0.06               # 중앙에서 ±6%면 '중앙'으로 간주
EMA_ALPHA = 0.3                    # 0.2~0.4 정도 추천
STABLE_FRAMES = 10                 # 중앙 유지 프레임 수(n)
PRINT_EVERY = 0.15                 # 메시지 너무 자주 안 찍게(초)

# 성능 최적화
FACE_EVERY = 1                     # 얼굴 검출 주기(프레임 간격). 1=매 프레임
HAAR_EVERY = 2                     # 상체 힌트 주기. 2=2프레임마다
HAAR_DOWNSCALE = 0.5               # 힌트용 검출은 다운스케일해서 속도↑
# ===================

mp_face = mp.solutions.face_detection

def main():
    cv2.setUseOptimized(True)
    # cv2.setNumThreads(2)  # 상황에 맞게 조절

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    face_det = mp_face.FaceDetection(
        model_selection=FDETECT_MODEL,
        min_detection_confidence=MIN_DET_CONF
    )

    # Haar upperbody (힌트용, 매우 경량)
    upperbody_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_upperbody.xml'
    )
    if upperbody_cascade.empty():
        print("[WARN] haarcascade_upperbody.xml 로드 실패. OpenCV 설치를 확인하세요.")

    target_y = 0.5  # 화면 세로 중앙 (정규화 좌표)
    deadband = DEADBAND_PCT
    ema_y = None
    stable_count = 0
    last_print_t = 0
    last_state = None  # "up", "down", "center" 등

    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            frame_idx += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ---------- 1) 얼굴 검출 (주기 FACE_EVERY) ----------
            state = None
            if frame_idx % FACE_EVERY == 0:
                result = face_det.process(frame_rgb)
            else:
                result = None

            if result and result.detections:
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
                    state = "up"      # 얼굴이 중앙보다 위 → 카메라를 "Go down!" (문구는 아래에서 통일)
                    stable_count = 0
                else:
                    state = "down"    # 얼굴이 중앙보다 아래 → "Go up!"
                    stable_count = 0

            else:
                # ---------- 2) 얼굴 없음 → 경량 상체 힌트 (주기 HAAR_EVERY) ----------
                if (frame_idx % HAAR_EVERY == 0) and not upperbody_cascade.empty():
                    # 다운스케일 후 회색화 (속도↑)
                    small = cv2.resize(frame, None, fx=HAAR_DOWNSCALE, fy=HAAR_DOWNSCALE, interpolation=cv2.INTER_LINEAR)
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

                    # 파라미터는 라이트/거리 따라 조절
                    # scaleFactor 1.05~1.2, minNeighbors 2~5, minSize는 50~100 추천
                    bodies = upperbody_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.08,
                        minNeighbors=3,
                        minSize=(int(80*HAAR_DOWNSCALE), int(80*HAAR_DOWNSCALE))
                    )
                    if len(bodies) > 0:
                        # 상체만 보이는 상황으로 간주 → 키 큼 → 위로 스캔 권고
                        state = "hint_up"     # Go up
                    else:
                        # 상체도 안 보임 → 키 작음 → 아래로 스캔 권고
                        state = "hint_down"   # Go down
                else:
                    state = "no-face"

                stable_count = 0  # 얼굴이 없으므로 중앙 안정화 카운트 리셋

            # ---------- 출력(쿨다운) ----------
            now = time.time()
            if now - last_print_t >= PRINT_EVERY:
                if state == "center":
                    if stable_count >= STABLE_FRAMES:
                        if last_state != "center":
                            print("Centered ✅ (stable)")
                    else:
                        print(f"Centered… ({stable_count}/{STABLE_FRAMES})")
                elif state == "up":
                    # 요구사항: 화면 중앙보다 '위'에 있으면 "Go down!"
                    print("Go down! (face above center)")
                elif state == "down":
                    # 요구사항: 화면 중앙보다 '아래'에 있으면 "Go up!"
                    print("Go up! (face below center)")
                elif state == "hint_up":
                    print("No face, upper-body detected → Go up (scan up)")
                elif state == "hint_down":
                    print("No face, no upper-body → Go down (scan down)")
                elif state == "no-face":
                    print("No face detected. Move camera up/down to find a face…")

                last_print_t = now
                last_state = state

            # ---------- 디버그용 가이드 라인 ----------
            cv2.line(frame, (0, int(FRAME_H*0.5)), (FRAME_W, int(FRAME_H*0.5)), (0,255,0), 1)
            if ema_y is not None:
                y_px = int(ema_y * FRAME_H)
                cv2.line(frame, (0, y_px), (FRAME_W, y_px), (255,0,0), 1)
            cv2.imshow("guide", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        face_det.close()
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
