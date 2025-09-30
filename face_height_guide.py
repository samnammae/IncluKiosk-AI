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
DEADBAND_PCT = 0.06
EMA_ALPHA = 0.3
STABLE_FRAMES = 10
PRINT_EVERY = 0.15

# EdgeTPU person detector 모델/라벨 (Coral 공식 COCO SSD MobileNet v2 예시)
EDGETPU_MODEL = "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
EDGETPU_LABELS = "coco_labels.txt"
PERSON_LABEL = "person"
PERSON_SCORE_TH = 0.4
# ===================

mp_face = mp.solutions.face_detection

def load_labels(path):
    labels = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            pair = line.strip().split(maxsplit=1)
            if len(pair) == 2:
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[len(labels)] = pair[0].strip()
    return labels

def detect_person_bbox(interpreter, labels, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # 입력 크기에 맞춰 리사이즈 & 세팅
    _, scale = common.set_resized_input(
        interpreter, rgb.shape[1], rgb.shape[0], lambda size: cv2.resize(rgb, size)
    )
    interpreter.invoke()
    objs = detect.get_objects(interpreter, score_threshold=PERSON_SCORE_TH)
    # 가장 큰 person 박스 하나만 반환
    H, W = rgb.shape[:2]
    best = None
    best_area = -1
    for o in objs:
        cls = labels.get(o.id, str(o.id))
        if cls != PERSON_LABEL:
            continue
        # bbox는 모델 입력 좌표 기준 → 원본 스케일로 복원
        bbox = o.bbox  # (x, y, w, h) in input space
        x0 = int(bbox.xmin / scale[0]); y0 = int(bbox.ymin / scale[1])
        x1 = int((bbox.xmin + bbox.width) / scale[0]); y1 = int((bbox.ymin + bbox.height) / scale[1])
        # 정규화(0~1)
        x0n = max(0, min(1, x0 / W)); x1n = max(0, min(1, x1 / W))
        y0n = max(0, min(1, y0 / H)); y1n = max(0, min(1, y1 / H))
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
