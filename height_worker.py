import cv2, time, numpy as np, asyncio, websockets, json, threading
import tflite_runtime.interpreter as tflite
import detect
from linear_actuator.linear_actuator_controller import (
    init_motor,
    cleanup_motor,
    moveUp, 
    moveDown,
    exceed_max_height,
    exceed_min_height 
)

# ====== 설정 ======
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480
MIN_DET_CONF = 0.5  # SSD confidence threshold
DEADBAND_PCT = 0.06  # range of center
EMA_ALPHA = 0.3
STABLE_FRAMES = 10
PRINT_EVERY = 0.15

WITH_FACE = 100
WITHOUT_FACE = 500

# 자동 종료 설정
AUTO_EXIT_STABLE_TIME = 3  # 3초간 안정화 유지하면 자동 종료
NO_DETECTION_TIMEOUT = 15   # 15초간 사용자 미감지 시 타임아웃

# SSD MobileNet V2 모델 경로
FACE_MODEL = "models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"
PERSON_MODEL = "models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
PERSON_SCORE_TH = 0.4

# WebSocket 설정
HUB_WS_URL = "ws://localhost:8766"

# ===================

# 전역 제어 플래그
should_stop = False
ws_connection = None


def detect_faces_ssd(interpreter, frame_bgr, score_threshold=0.5):
    """
    SSD MobileNet V2로 얼굴 감지
    Returns: [(xmin, ymin, xmax, ymax, score), ...] in normalized coordinates [0,1]
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return []
    
    H, W = frame_bgr.shape[:2]
    if H <= 0 or W <= 0:
        return []
    
    try:
        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # 모델 입력 크기로 리사이즈
        width, height = detect.input_size(interpreter)
        resized = cv2.resize(frame_rgb, (width, height))
        
        # 텐서 설정
        tensor = detect.input_tensor(interpreter)
        tensor.fill(0)
        tensor[:, :] = resized.copy()
        del tensor
        
        # 추론
        interpreter.invoke()
        
        # 결과 가져오기
        objs = detect.get_output(interpreter, score_threshold, (1.0, 1.0))
        
        # 절대 좌표를 정규화 좌표로 변환
        faces = []
        for obj in objs:
            bbox = obj.bbox
            # 입력 모델 크기 기준 좌표를 정규화
            xmin = bbox.xmin / width
            ymin = bbox.ymin / height
            xmax = bbox.xmax / width
            ymax = bbox.ymax / height
            
            # 범위 제한
            xmin = max(0.0, min(1.0, xmin))
            ymin = max(0.0, min(1.0, ymin))
            xmax = max(0.0, min(1.0, xmax))
            ymax = max(0.0, min(1.0, ymax))
            
            faces.append((xmin, ymin, xmax, ymax, obj.score))
        
        return faces
    
    except Exception as e:
        print(f"[FaceDetect] 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        return []


def detect_person_ssd(interpreter, frame_bgr, score_threshold=0.4):
    """
    SSD MobileNet V2 COCO로 사람 감지 (class_id=0이 person)
    Returns: (xmin, ymin, xmax, ymax) in normalized coordinates [0,1] or None
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    
    H, W = frame_bgr.shape[:2]
    if H <= 0 or W <= 0:
        return None
    
    try:
        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # 모델 입력 크기로 리사이즈
        width, height = detect.input_size(interpreter)
        resized = cv2.resize(frame_rgb, (width, height))
        
        # 텐서 설정
        tensor = detect.input_tensor(interpreter)
        tensor.fill(0)
        tensor[:, :] = resized.copy()
        del tensor
        
        # 추론
        interpreter.invoke()
        
        # 결과 가져오기
        objs = detect.get_output(interpreter, score_threshold, (1.0, 1.0))
        
        # class_id=0 (person) 중 가장 큰 것 찾기
        best = None
        best_area = -1.0
        
        for obj in objs:
            # COCO 모델에서 class_id=0이 person
            if obj.id != 0:
                continue
            
            bbox = obj.bbox
            # 입력 모델 크기 기준 좌표를 정규화
            xmin = max(0.0, min(1.0, bbox.xmin / width))
            ymin = max(0.0, min(1.0, bbox.ymin / height))
            xmax = max(0.0, min(1.0, bbox.xmax / width))
            ymax = max(0.0, min(1.0, bbox.ymax / height))
            
            area = (xmax - xmin) * (ymax - ymin)
            if area > best_area:
                best_area = area
                best = (xmin, ymin, xmax, ymax)
        
        return best
    
    except Exception as e:
        print(f"[PersonDetect] 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


def track_height():
    """높이 추적 메인 로직 (동기 함수)"""
    global should_stop
    
    try:
        init_motor()    # GPIO 세팅 및 리니어 액추에이터 모터 활성화
    except Exception as e:
        print(f"[Height] 모터 초기화 실패: {e}")
        return "error"

    # 카메라
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[Height] 카메라 열기 실패 (인덱스: {CAM_INDEX})")
        cleanup_motor()
        return "error"
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # 실제 설정된 해상도 확인
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Height] 카메라 해상도: {actual_w}x{actual_h}")

    # SSD MobileNet V2 interpreter 초기화
    try:
        # 얼굴 검출용 interpreter
        face_interpreter = tflite.Interpreter(
            model_path=FACE_MODEL,
            experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
        )
        face_interpreter.allocate_tensors()
        print(f"[Height] Face 모델 로드 완료 (EdgeTPU)")
        
        # 사람 검출용 interpreter
        person_interpreter = tflite.Interpreter(
            model_path=PERSON_MODEL,
            experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
        )
        person_interpreter.allocate_tensors()
        print(f"[Height] Person 모델 로드 완료 (EdgeTPU)")
        
    except Exception as e:
        print(f"[Height] 모델 로드 실패: {e}")
        cap.release()
        cleanup_motor()
        return "error"

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
            if not ok or frame is None:
                print("[Height] 카메라 읽기 실패")
                time.sleep(0.1)
                continue

            frame_idx += 1
            
            # 1단계: 얼굴 검출 시도
            faces = detect_faces_ssd(face_interpreter, frame, MIN_DET_CONF)

            state = None
            detection_found = False
            
            if faces:
                # 얼굴 검출 성공 - 가장 큰 얼굴 사용
                detection_found = True
                face = max(faces, key=lambda f: (f[2]-f[0])*(f[3]-f[1]))  # 가장 큰 얼굴
                xmin, ymin, xmax, ymax, score = face
                
                y_center = (ymin + ymax) * 0.5

                ema_y = y_center if ema_y is None else EMA_ALPHA * y_center + (1 - EMA_ALPHA) * ema_y
                diff = ema_y - target_y
                
                if abs(diff) <= deadband:
                    state = "center"
                    stable_count += 1
                elif diff < 0:
                    state = "up"
                    stable_count = 0
                    moveUp(WITH_FACE)
                    
                    if exceed_max_height():
                        print("🚫 최대 높이 도달 → 높이 조절 종료")
                        break
                else:
                    state = "down"
                    stable_count = 0
                    moveDown(WITH_FACE)
                    
                    if exceed_min_height():
                        print("🚫 최소 높이 도달 → 높이 조절 종료")
                        break
            else:
                # 2단계: 얼굴 없음 → 사람 검출로 힌트 제공
                person = detect_person_ssd(person_interpreter, frame, PERSON_SCORE_TH)
                stable_count = 0
                
                if person is None:
                    state = "hint_down"
                    moveDown(WITHOUT_FACE)
                    
                    if exceed_min_height():
                        print("🚫 최소 높이 도달 → 높이 조절 종료")
                        break
                else:
                    detection_found = True
                    x0, y0, x1, y1 = person
                    y_center = (y0 + y1) * 0.5
                    
                    if y0 <= 0.05:
                        state = "hint_up"
                        moveUp(WITHOUT_FACE)
                        
                        if exceed_max_height():
                            print("🚫 최대 높이 도달 → 높이 조절 종료")
                            break
                    elif y1 >= 0.95:
                        state = "hint_down"
                        moveDown(WITHOUT_FACE)
                        
                        if exceed_min_height():
                            print("🚫 최소 높이 도달 → 높이 조절 종료")
                            break
                    else:
                        state = "up" if y_center < target_y - deadband else "down"

            # 감지 시간 업데이트
            if detection_found:
                last_detection_time = time.time()
            
            # 타임아웃 체크
            if time.time() - last_detection_time > NO_DETECTION_TIMEOUT:
                print(f"⚠️ {NO_DETECTION_TIMEOUT}초간 사용자 미감지 → 높이 조절 타임아웃")
                break

            # 자동 종료 체크
            if state == "center" and stable_count >= STABLE_FRAMES:
                if stable_start_time is None:
                    stable_start_time = time.time()
                    print("✅ 안정화 감지 - 자동 종료 타이머 시작")
                elif time.time() - stable_start_time >= AUTO_EXIT_STABLE_TIME:
                    print(f"✅ {AUTO_EXIT_STABLE_TIME}초간 안정화 유지 → 높이 조절 완료!")
                    break
            else:
                stable_start_time = None

            # 상태 출력
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

    except KeyboardInterrupt:
        print("KeyboardInterrupt - 높이 조절 중단")
    except Exception as e:
        print(f"[Height] track_height 예외: {e}")
        import traceback
        traceback.print_exc()
        return "error"
    finally:
        cap.release()
        cleanup_motor()
        
        # 종료 이유 판단
        if exceed_max_height() or exceed_min_height():
            return "limit_reached"
        elif time.time() - last_detection_time > NO_DETECTION_TIMEOUT:
            return "timeout"
        else:
            return "complete"


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
                try:
                    result_status = track_height()
                except Exception as e:
                    print(f"[Height Worker] track_height() 예외: {e}")
                    import traceback
                    traceback.print_exc()
                    result_status = "error"
            
            track_thread = threading.Thread(target=run_track, daemon=True)
            track_thread.start()
            
            # 스레드가 시작될 시간 확보
            await asyncio.sleep(0.5)
            
            # 스레드가 즉시 종료되었는지 확인
            if not track_thread.is_alive():
                print("[Height Worker] ⚠️ track_height() 스레드가 즉시 종료됨")
                raise Exception("track_height failed to start")
            
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
            elif result_status == "limit_reached":
                await ws.send(json.dumps({"type": "HEIGHT_SET_END"}))
                print("[Height Worker] 한계 도달 → END 전송 (정상 종료)")
            elif result_status == "timeout":
                await ws.send(json.dumps({"type": "HEIGHT_SET_TIMEOUT"}))
                print("[Height Worker] 타임아웃 → TIMEOUT 전송")
            else:
                await ws.send(json.dumps({"type": "HEIGHT_SET_END"}))
                print("[Height Worker] 정상 완료 → END 전송")
    
    except Exception as e:
        print(f"[Height Worker] 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ws_connection = None


if __name__ == "__main__":
    print("[Height Worker] 프로세스 시작")
    try:
        asyncio.run(ws_client())
    except Exception as e:
        print(f"[Height Worker] 메인 예외: {e}")
        import traceback
        traceback.print_exc()
    print("[Height Worker] 프로세스 종료")