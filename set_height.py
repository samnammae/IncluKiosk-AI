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

# Pose(상체 힌트) 설정
POSE_DET_CONF = 0.5
POSE_TRK_CONF = 0.5
POSE_VIS_TH   = 0.5               # landmark.visibility 임계값
# ===================

mp_face = mp.solutions.face_detection
mp_pose = mp.solutions.pose

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    face_det = mp_face.FaceDetection(
        model_selection=FDETECT_MODEL,
        min_detection_confidence=MIN_DET_CONF
    )

    pose = mp_pose.Pose(
        model_complexity=0,                 # 경량 모드
        enable_segmentation=False,
        min_detection_confidence=POSE_DET_CONF,
        min_tracking_confidence=POSE_TRK_CONF
    )

    target_y = 0.5  # 화면 세로 중앙 (정규화 좌표)
    deadband = DEADBAND_PCT
    ema_y = None
    stable_count = 0
    last_print_t = 0
    last_state = None  # "up", "down", "center" 등

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

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
                    state = "up"      # 얼굴이 중앙보다 위 → (사용자 스펙에 맞춰) Go down/Go up 문자열은 아래에서 출력 통일
                    stable_count = 0
                else:
                    state = "down"    # 얼굴이 중앙보다 아래
                    stable_count = 0
            else:
                # 얼굴 없음 → Pose로 상체 힌트 확인
                pose_result = pose.process(frame_rgb)
                upper_seen = False
                head_parts_seen = False

                if pose_result.pose_landmarks:
                    lms = pose_result.pose_landmarks.landmark
                    def vis(i):
                        return (0.0 <= lms[i].x <= 1.0) and (0.0 <= lms[i].y <= 1.0) and (lms[i].visibility >= POSE_VIS_TH)

                    # 어깨(상체) 가시성
                    LS = mp_pose.PoseLandmark.LEFT_SHOULDER.value
                    RS = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                    upper_seen = vis(LS) or vis(RS)

                    # 머리 파츠(코/눈/귀/입) 가시성
                    head_idxs = [
                        mp_pose.PoseLandmark.NOSE.value,
                        mp_pose.PoseLandmark.LEFT_EYE.value,
                        mp_pose.PoseLandmark.RIGHT_EYE.value,
                        mp_pose.PoseLandmark.LEFT_EAR.value,
                        mp_pose.PoseLandmark.RIGHT_EAR.value,
                        mp_pose.PoseLandmark.MOUTH_LEFT.value,
                        mp_pose.PoseLandmark.MOUTH_RIGHT.value,
                    ]
                    head_parts_seen = any(vis(i) for i in head_idxs)

                if upper_seen and (not head_parts_seen):
                    # 상체만 보이고 얼굴 파츠는 없음 → 키가 큰 경우로 간주 → 위로 스캔 시작 권장
                    state = "hint_up"     # Go up (카메라 위로)
                elif not upper_seen:
                    # 상체 자체가 없음 → 키가 작은 경우로 간주 → 아래로 스캔
                    state = "hint_down"   # Go down (카메라 아래로)
                else:
                    # 상체/머리 힌트가 애매 → 일반 no-face
                    state = "no-face"

                stable_count = 0  # 얼굴이 없으므로 중앙 안정화 카운트 리셋

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
                    # 스펙: 얼굴이 중앙보다 '위'에 있으면 "Go down!"
                    print("Go down! (face is above center)")
                elif state == "down":
                    # 스펙: 얼굴이 중앙보다 '아래'에 있으면 "Go up!"
                    print("Go up! (face is below center)")
                elif state == "hint_up":
                    print("No face, upper-body only → Go up (scan up)")
                elif state == "hint_down":
                    print("No face, no upper-body → Go down (scan down)")
                elif state == "no-face":
                    print("No face detected. Move camera up/down to find a face…")

                last_print_t = now
                last_state = state

            # 디버그용 가이드 라인(원하면 주석 해제 유지)
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
        pose.close()
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
