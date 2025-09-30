# face_height_guide_edgetpu_person.py
import cv2, time, numpy as np
import mediapipe as mp
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect

# ====== 설정 ======
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480
FDETECT_MODEL = 0
MIN_DET_CONF = 0.6
DEADBAND_PCT = 0.06 # range of center
EMA_ALPHA = 0.3
STABLE_FRAMES = 10
PRINT_EVERY = 0.15

# EdgeTPU person detector 모델/라벨
EDGETPU_MODEL = "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
EDGETPU_LABELS = "coco_labels.txt"
PERSON_LABEL = "person"
PERSON_SCORE_TH = 0.4
# ===================

mp_face = mp.solutions.face_detection

def load_labels(path):
    labels = {}
    with open(path, 'r', encoding='utf-8') as f:
        for idx, raw in enumerate(f):
            line = raw.strip()
            if not line:
                continue
            # 1) "0 person" / "0: person" 처리
            if ':' in line and line.split(':', 1)[0].strip().isdigit():
                k, v = line.split(':', 1)
                labels[int(k.strip())] = v.strip()
                continue
            parts = line.split()
            if parts and parts[0].isdigit():
                labels[int(parts[0])] = ' '.join(parts[1:]) if len(parts) > 1 else str(parts[0])
                continue
            # 2) "person" 처럼 이름만 있는 줄 → 그 줄 번호가 id
            labels[idx] = line
    return labels


def detect_person_bbox(interpreter, labels, bgr):
    # 1) 원본 크기
    H, W = bgr.shape[:2]

    # 2) RGB 변환
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 3) 모델 입력 크기 구해서 리사이즈
    in_w, in_h = common.input_size(interpreter)[:2]  # (width, height[, channels])
    resized = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

    # 4) 텐서 설정 & 추론
    common.set_input(interpreter, resized)
    interpreter.invoke()

    # 5) 감지 결과 받기
    objs = detect.get_objects(interpreter, score_threshold=PERSON_SCORE_TH)

    # 6) 입력텐서 좌표계 → 원본 프레임 좌표계로 역스케일
    x_scale = in_w / W
    y_scale = in_h / H

    best = None
    best_area = -1.0
    for o in objs:
        name = labels.get(o.id, str(o.id)).lower()
        if name != "person":
            continue

        # pycoral bbox는 입력텐서 기준 절대좌표
        bb = o.bbox  # has xmin, ymin, width, height

        x0 = int(bb.xmin / x_scale)
        y0 = int(bb.ymin / y_scale)
        x1 = int((bb.xmin + bb.width)  / x_scale)
        y1 = int((bb.ymin + bb.height) / y_scale)

        # 정규화 [0,1]
        x0n = max(0.0, min(1.0, x0 / W))
        y0n = max(0.0, min(1.0, y0 / H))
        x1n = max(0.0, min(1.0, x1 / W))
        y1n = max(0.0, min(1.0, y1 / H))

        area = (x1n - x0n) * (y1n - y0n)
        if area > best_area:
            best_area = area
            best = (x0n, y0n, x1n, y1n)

    return best  # (x0, y0, x1, y1) in [0,1] or None

def main():
    # 카메라
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 얼굴 검출 (BlazeFace 기반)
    face_det = mp_face.FaceDetection(
        model_selection=FDETECT_MODEL,
        min_detection_confidence=MIN_DET_CONF
    )

    # EdgeTPU person detector 준비
    labels = load_labels(EDGETPU_LABELS)
    interpreter = make_interpreter(EDGETPU_MODEL)
    interpreter.allocate_tensors()

    target_y = 0.5
    deadband = DEADBAND_PCT
    ema_y = None
    stable_count = 0
    last_print_t = 0
    last_state = None
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            frame_idx += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_det.process(frame_rgb)

            state = None
            if result and result.detections:
                det = result.detections[0]
                box = det.location_data.relative_bounding_box
                y_center = box.ymin + box.height * 0.5  # 0~1

                ema_y = y_center if ema_y is None else EMA_ALPHA * y_center + (1 - EMA_ALPHA) * ema_y
                diff = ema_y - target_y
                if abs(diff) <= deadband:
                    state = "center"; stable_count += 1
                elif diff < 0:
                    state = "up";     stable_count = 0   # 얼굴이 중앙보다 위 → 카메라 내려야
                else:
                    state = "down";   stable_count = 0   # 얼굴이 중앙보다 아래 → 카메라 올려야
            else:
                # 얼굴 없음 → EdgeTPU 사람 박스로 힌트
                person = detect_person_bbox(interpreter, labels, frame)
                stable_count = 0
                if person is None:
                    state = "hint_down"   # 화면에 사람 박스도 없으면 카메라가 너무 위일 확률↑ → 아래로 스캔
                else:
                    x0, y0, x1, y1 = person
                    y_center = (y0 + y1) * 0.5
                    # 경계 접촉이면 방향 확신 강화
                    if y0 <= 0.05:
                        state = "hint_up"     # 상단에 걸림 → 키 큼 → 카메라 위로
                    elif y1 >= 0.95:
                        state = "hint_down"   # 하단에 걸림 → 키 작음 → 카메라 아래로
                    else:
                        # 중앙 기준으로 간단 판정
                        state = "up" if y_center < target_y - deadband else "down"

            now = time.time()
            if now - last_print_t >= PRINT_EVERY:
                if state == "center":
                    if stable_count >= STABLE_FRAMES:
                        if last_state != "center":
                            print("Centered ✅ (stable)")
                    else:
                        print(f"Centered… ({stable_count}/{STABLE_FRAMES})")
                elif state == "up":
                    print("Go down! (face/person above center)")
                elif state == "down":
                    print("Go up! (face/person below center)")
                elif state == "hint_up":
                    print("No face, person near top → Go up (scan up)")
                elif state == "hint_down":
                    print("No face, person missing/near bottom → Go down (scan down)")
                else:
                    print("No face/person hint. Sweep to search…")
                last_print_t = now
                last_state = state

            # 디버그 라인
            cv2.line(frame, (0, int(FRAME_H*0.5)), (FRAME_W, int(FRAME_H*0.5)), (0,255,0), 1)
            if ema_y is not None:
                y_px = int(ema_y * FRAME_H)
                cv2.line(frame, (0, y_px), (FRAME_W, y_px), (255,0,0), 1)
            cv2.imshow("guide", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        face_det.close()
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
