#!/usr/bin/env python3
import time
import cv2
import numpy as np
from collections import deque

# ====== 기본 설정 ======
MODEL_PATH = "models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"
SOURCE = 0                # 0 = 기본 웹캠
THRESHOLD = 0.5
TOP_K = 10
DEVICE = "usb:0"          # coral usb accelerator
USE_PYCORAL = True

# ====== pycoral or tflite_runtime ======
try:
    from pycoral.adapters import common, detect
    from pycoral.utils import edgetpu
except Exception:
    import tflite_runtime.interpreter as tflite
    USE_PYCORAL = False


def smooth_fps():
    buf = deque(maxlen=30)
    last = None

    def update():
        nonlocal last
        now = time.perf_counter()
        if last is None:
            last = now
            return None
        dt = now - last
        last = now
        if dt <= 0:
            return None
        buf.append(1.0 / dt)
        return sum(buf) / len(buf)

    return update


def draw_detections(frame, objs, scale_x, scale_y, color=(0, 255, 0)):
    h, w = frame.shape[:2]
    for o in objs:
        if hasattr(o, "bbox"):  # pycoral 객체
            bbox = o.bbox
            x0, y0 = int(bbox.xmin * scale_x), int(bbox.ymin * scale_y)
            x1, y1 = int(bbox.xmax * scale_x), int(bbox.ymax * scale_y)
            score = getattr(o, "score", 0.0)
        else:  # 수동 파싱 케이스
            x0, y0, x1, y1, score = o
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        cv2.putText(frame, f"{score*100:.1f}%", (x0, max(0, y0 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def run_pycoral():
    print("[INFO] Running with pycoral + EdgeTPU")
    interpreter = edgetpu.make_interpreter(MODEL_PATH, device=DEVICE)
    interpreter.allocate_tensors()
    in_w, in_h = common.input_size(interpreter)

    cap = cv2.VideoCapture(SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    fps_update = smooth_fps()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        # BGR -> RGB, 그리고 모델 입력 크기(in_w, in_h)로 리사이즈
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (in_w, in_h))
        common.set_input(interpreter, resized)
        start = time.perf_counter()
        interpreter.invoke()
        objs = detect.get_objects(interpreter, score_threshold=THRESHOLD, top_k=TOP_K)
        inf_ms = (time.perf_counter() - start) * 1000.0

        draw_detections(frame, objs, w / in_w, h / in_h)
        fps = fps_update()
        if fps:
            cv2.putText(frame, f"FPS: {fps:.1f}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Infer: {inf_ms:.1f} ms, Faces: {len(objs)}",
                    (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Coral Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_tflite():
    print("[INFO] Running with tflite_runtime + EdgeTPU delegate")
    delegates = []
    try:
        delegates.append(tflite.load_delegate("libedgetpu.so.1"))
    except Exception as e:
        print(f"[WARN] Failed to load EdgeTPU delegate: {e}")

    interpreter = tflite.Interpreter(model_path=MODEL_PATH, experimental_delegates=delegates)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    in_h, in_w = input_details[0]["shape"][1:3]

    cap = cv2.VideoCapture(SOURCE)
    fps_update = smooth_fps()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (in_w, in_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_tensor = np.expand_dims(rgb, axis=0).astype(np.uint8)

        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], input_tensor)
        interpreter.invoke()
        inf_ms = (time.perf_counter() - start) * 1000.0

        boxes = interpreter.get_tensor(output_details[0]["index"])[0]
        scores = interpreter.get_tensor(output_details[2]["index"])[0]
        count = int(interpreter.get_tensor(output_details[3]["index"])[0])

        dets = []
        for i in range(count):
            score = float(scores[i])
            if score < THRESHOLD:
                continue
            ymin, xmin, ymax, xmax = boxes[i]
            x0, y0 = int(xmin * w), int(ymin * h)
            x1, y1 = int(xmax * w), int(ymax * h)
            dets.append((x0, y0, x1, y1, score))

        draw_detections(frame, dets, 1.0, 1.0)
        fps = fps_update()
        if fps:
            cv2.putText(frame, f"FPS: {fps:.1f}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Infer: {inf_ms:.1f} ms, Faces: {len(dets)}",
                    (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Coral Face Detection (delegate)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if USE_PYCORAL:
        run_pycoral()
    else:
        run_tflite()
