import cv2, time, numpy as np, asyncio, websockets, json, threading
import mediapipe as mp
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect

from linear_actuator.linear_actuator_controller import (
    init_motor,
    cleanup_motor,
    moveUp, 
    moveDown
)

# ====== 설정 ======
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480
FDETECT_MODEL = 0
MIN_DET_CONF = 0.6
DEADBAND_PCT = 0.06  # range of center
EMA_ALPHA = 0.3
STABLE_FRAMES = 10
PRINT_EVERY = 0.15

WITH_FACE = 100
WITHOUT_FACE = 500

# 자동 종료 설정
AUTO_EXIT_STABLE_TIME = 3  # 3초간 안정화 유지하면 자동 종료
NO_DETECTION_TIMEOUT = 15   # 15초간 사용자 미감지 시 타임아웃

# EdgeTPU person detector 모델/라벨
EDGETPU_MODEL = "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
EDGETPU_LABELS = "coco_labels.txt"
PERSON_LABEL = "person"
PERSON_SCORE_TH = 0.4

# WebSocket 설정
HUB_WS_URL = "ws://localhost:8766"

# ===================

# 전역 제어 플래그
should_stop = False
ws_connection = None

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


def track_height():
    """높이 추적 메인 로직 (동기 함수)"""
    global should_stop
    
    init_motor()    # GPIO 세팅 및 리니어 액추에이터 모터 활성화

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
    stable_start_time = None
    last_detection_time = time.time()
    last_print_t = 0
    last_state = None
    frame_idx = 0

    try:
        while not should_stop:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            frame_idx += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_det.process(frame_rgb)

            state = None
            detection_found = False  # 이번 프레임에 감지 여부
            
            if result and result.detections:
                detection_found = True
                det = result.detections[0]
                box = det.location_data.relative_bounding_box
                y_center = box.ymin + box.height * 0.5  # 0~1

                ema_y = y_center if ema_y is None else EMA_ALPHA * y_center + (1 - EMA_ALPHA) * ema_y
                diff = ema_y - target_y
                if abs(diff) <= deadband:
                    state = "center"
                    stable_count += 1
                elif diff < 0:
                    state = "up"
                    stable_count = 0   # 얼굴이 중앙보다 위 → 카메라 내려야
                    moveUp(WITH_FACE)    # 액추에이터 위로 이동
                else:
                    state = "down"
                    stable_count = 0   # 얼굴이 중앙보다 아래 → 카메라 올려야
                    moveDown(WITH_FACE)     # 액추에이터 아래로 이동
            else:
                # 얼굴 없음 → EdgeTPU 사람 박스로 힌트
                person = detect_person_bbox(interpreter, labels, frame)
                stable_count = 0
                if person is None:
                    state = "hint_down"   # 화면에 사람 박스도 없으면 카메라가 너무 위일 확률 → 아래로 스캔
                    moveDown(WITHOUT_FACE) # 액추에이터 아래로 이동
                else:
                    detection_found = True  # 사람 박스는 감지됨
                    x0, y0, x1, y1 = person
                    y_center = (y0 + y1) * 0.5
                    # 경계 접촉이면 방향 확신 강화
                    if y0 <= 0.05:
                        state = "hint_up"     # 상단에 걸림 → 키 큼 → 카메라 위로
                        moveUp(WITHOUT_FACE)    # 액추에이터 위로 이동
                    elif y1 >= 0.95:
                        state = "hint_down"   # 하단에 걸림 → 키 작음 → 카메라 아래로
                        moveDown(WITHOUT_FACE) # 액추에이터 아래로 이동
                    else:
                        # 중앙 기준으로 간단 판정
                        state = "up" if y_center < target_y - deadband else "down"

            # 감지 시간 업데이트
            if detection_found:
                last_detection_time = time.time()
            
            # 타임아웃 체크 (30초간 미감지)
            if time.time() - last_detection_time > NO_DETECTION_TIMEOUT:
                print(f"⚠️ {NO_DETECTION_TIMEOUT}초간 사용자 미감지 → 높이 조절 타임아웃")
                break  # 루프 종료 (finally에서 타임아웃 메시지 전송)

            # 자동 종료 체크 (3초간 안정화)
            if state == "center" and stable_count >= STABLE_FRAMES:
                if stable_start_time is None:
                    stable_start_time = time.time()
                    print("✅ 안정화 감지 - 자동 종료 타이머 시작")
                elif time.time() - stable_start_time >= AUTO_EXIT_STABLE_TIME:
                    print(f"✅ {AUTO_EXIT_STABLE_TIME}초간 안정화 유지 → 높이 조절 완료!")
                    break  # 정상 완료
            else:
                stable_start_time = None  # 안정화 해제되면 타이머 리셋

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
            
            time.sleep(0.03)  # 약 30 FPS

    except KeyboardInterrupt:
        print("KeyboardInterrupt - 높이 조절 중단")
    finally:
        cap.release()
        face_det.close()
        # cv2.destroyAllWindows()
        cleanup_motor()         # GPIO 정리 및 리니어 액추에이터 모터 정리
        
        # 타임아웃인지 정상 완료인지 판단
        return "timeout" if (time.time() - last_detection_time > NO_DETECTION_TIMEOUT) else "complete"

async def ws_client():
    """WebSocket 클라이언트 - Hub와 통신"""
    global should_stop, ws_connection
    
    result_status = None
    
    try:
        async with websockets.connect(HUB_WS_URL) as ws:
            ws_connection = ws
            print("[Height Worker] Hub 연결됨 (포트 8766)")
            
            # 추적 시작을 별도 스레드에서 실행
            def run_track():
                nonlocal result_status
                result_status = track_height()
            
            track_thread = threading.Thread(target=run_track, daemon=True)
            track_thread.start()
            
            # WebSocket 메시지 수신 대기 (중단 명령용)
            try:
                async for raw in ws:
                    data = json.loads(raw)
                    msg_type = data.get("type")
                    
                    if msg_type == "HEIGHT_SET_OFF":
                        print("[Height Worker] 중단 명령 수신")
                        should_stop = True
                        break
            except websockets.exceptions.ConnectionClosed:
                print("[Height Worker] Hub 연결 끊김")
            
            # 추적 완료 대기
            track_thread.join(timeout=3)
            
            # 결과에 따라 다른 메시지 전송
            if should_stop:
                await ws.send(json.dumps({"type": "HEIGHT_SET_CANCEL"}))
                print("[Height Worker] 중단됨 → CANCELLED 전송")
            elif result_status == "timeout":
                await ws.send(json.dumps({"type": "HEIGHT_SET_TIMEOUT"}))
                print("[Height Worker] 타임아웃 → TIMEOUT 전송")
            else:
                await ws.send(json.dumps({"type": "HEIGHT_SET_END"}))
                print("[Height Worker] 정상 완료 → END 전송")
    
    except Exception as e:
        print(f"[Height Worker] 오류: {e}")
    finally:
        ws_connection = None


if __name__ == "__main__":
    asyncio.run(ws_client())