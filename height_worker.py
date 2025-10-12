import cv2, time, numpy as np, asyncio, websockets, json, threading
import tflite_runtime.interpreter as tflite
import detect
import os
import asyncio
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
MIN_DET_CONF = 0.4
PERSON_SCORE_TH = 0.3

DEADBAND_PCT = 0.06
EMA_ALPHA = 0.3
STABLE_FRAMES = 10
PRINT_EVERY = 0.15

WITH_FACE = 100
WITHOUT_FACE = 500

AUTO_EXIT_STABLE_TIME = 3
NO_DETECTION_TIMEOUT = 15

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_MODEL = os.path.join(SCRIPT_DIR, "models", "ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite")
PERSON_MODEL = os.path.join(SCRIPT_DIR, "models", "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite")

# HUB_WS_URL = "ws://localhost:8766"
HUB_WS_URL = "ws://127.0.0.1:8766"

# 디버그 모드
DEBUG = True

# ===================

should_stop = False
ws_connection = None


def debug_log(msg):
    """디버그 로그 출력"""
    if DEBUG:
        print(f"[DEBUG] {msg}")


def detect_faces_ssd(interpreter, frame_bgr, score_threshold=0.5):
    """SSD로 얼굴 감지"""
    if frame_bgr is None or frame_bgr.size == 0:
        debug_log("detect_faces_ssd: 빈 프레임")
        return []
    
    H, W = frame_bgr.shape[:2]
    if H <= 0 or W <= 0:
        debug_log(f"detect_faces_ssd: 잘못된 크기 {W}x{H}")
        return []
    
    try:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        width, height = detect.input_size(interpreter)
        resized = cv2.resize(frame_rgb, (width, height))
        
        tensor = detect.input_tensor(interpreter)
        tensor.fill(0)
        tensor[:, :] = resized.copy()
        del tensor
        
        interpreter.invoke()
        objs = detect.get_output(interpreter, score_threshold, (1.0, 1.0))
        
        debug_log(f"detect_faces_ssd: {len(objs)} 얼굴 검출")
        
        faces = []
        for obj in objs:
            bbox = obj.bbox
            xmin = max(0.0, min(1.0, bbox.xmin / width))
            ymin = max(0.0, min(1.0, bbox.ymin / height))
            xmax = max(0.0, min(1.0, bbox.xmax / width))
            ymax = max(0.0, min(1.0, bbox.ymax / height))
            faces.append((xmin, ymin, xmax, ymax, obj.score))
            debug_log(f"  Face: score={obj.score:.2f}, bbox=({xmin:.2f},{ymin:.2f},{xmax:.2f},{ymax:.2f})")
        
        return faces
    except Exception as e:
        print(f"[Face] 오류: {e}")
        import traceback
        traceback.print_exc()
        return []


def detect_person_ssd(interpreter, frame_bgr, score_threshold=0.4):
    """SSD로 사람 감지"""
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    
    H, W = frame_bgr.shape[:2]
    if H <= 0 or W <= 0:
        return None
    
    try:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        width, height = detect.input_size(interpreter)
        resized = cv2.resize(frame_rgb, (width, height))
        
        tensor = detect.input_tensor(interpreter)
        tensor.fill(0)
        tensor[:, :] = resized.copy()
        del tensor
        
        interpreter.invoke()
        objs = detect.get_output(interpreter, score_threshold, (1.0, 1.0))
        
        best = None
        best_area = -1.0
        person_count = 0
        
        for obj in objs:
            if obj.id != 0:
                continue
            
            person_count += 1
            bbox = obj.bbox
            xmin = max(0.0, min(1.0, bbox.xmin / width))
            ymin = max(0.0, min(1.0, bbox.ymin / height))
            xmax = max(0.0, min(1.0, bbox.xmax / width))
            ymax = max(0.0, min(1.0, bbox.ymax / height))
            
            area = (xmax - xmin) * (ymax - ymin)
            if area > best_area:
                best_area = area
                best = (xmin, ymin, xmax, ymax)
        
        debug_log(f"detect_person_ssd: {person_count} 사람 검출")
        if best:
            debug_log(f"  Best person: bbox=({best[0]:.2f},{best[1]:.2f},{best[2]:.2f},{best[3]:.2f})")
        
        return best
    except Exception as e:
        print(f"[Person] 오류: {e}")
        import traceback
        traceback.print_exc()
        return None


def track_height():
    """높이 추적 메인 로직"""
    global should_stop
    
    try:
        print("[Height] 모터 초기화 중...")
        init_motor()
        print("[Height] ✅ 모터 초기화 완료")
    except Exception as e:
        print(f"[Height] ❌ 모터 초기화 실패: {e}")
        return "error"

    # 카메라
    print("[Height] 카메라 초기화 중...")
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[Height] ❌ 카메라 열기 실패 (인덱스: {CAM_INDEX})")
        # 디버그용: 현재 비디오 디바이스 나열
        try:
            import glob
            print("[Height] /dev/video* =", glob.glob("/dev/video*"))
        except:
            pass
        cleanup_motor()
        return "error"
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Height] 카메라 해상도: {actual_w}x{actual_h}")

    # 카메라 워밍업
    print("[Height] 카메라 워밍업 중 (30 프레임)...")
    warmup_success = 0
    for i in range(30):
        ret, frame = cap.read()
        if ret and frame is not None:
            warmup_success += 1
        time.sleep(0.033)
    print(f"[Height] 워밍업 완료: {warmup_success}/30 프레임 성공")
    
    if warmup_success < 20:
        print("[Height] ⚠️ 카메라 워밍업 실패율 높음")

    # 모델 로드
    print("[Height] AI 모델 로드 중...")
    try:
        face_interpreter = tflite.Interpreter(
            model_path=FACE_MODEL,
            experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
        )
        face_interpreter.allocate_tensors()
        print("[Height] ✅ 얼굴 검출 모델 로드 완료 (EdgeTPU)")
        
        person_interpreter = tflite.Interpreter(
            model_path=PERSON_MODEL,
            experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
        )
        person_interpreter.allocate_tensors()
        print("[Height] ✅ 사람 검출 모델 로드 완료 (EdgeTPU)")
        
    except Exception as e:
        print(f"[Height] ❌ 모델 로드 실패: {e}")
        cap.release()
        cleanup_motor()
        return "error"

    print("[Height] 🚀 높이 조절 시작!")
    
    TARGET_OFFSET_PCT = 0.08  # 8% 위로 목표 이동 (원하는 만큼 조절)
    target_y = min(0.9, max(0.1, 0.5 - TARGET_OFFSET_PCT))
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
                print("[Height] ⚠️ 카메라 읽기 실패")
                time.sleep(0.1)
                continue

            frame_idx += 1
            
            # 🔍 처음 몇 프레임을 저장해서 확인
            if frame_idx <= 5:
                filename = f"/tmp/frame_{frame_idx}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[DEBUG] 프레임 저장: {filename}")
                
            # 디버그: 프레임 정보
            if frame_idx % 60 == 0:
                debug_log(f"Frame {frame_idx}: shape={frame.shape}, dtype={frame.dtype}")
            
            # 얼굴 검출
            faces = detect_faces_ssd(face_interpreter, frame, MIN_DET_CONF)

            state = None
            detection_found = False
            
            if faces:
                detection_found = True
                face = max(faces, key=lambda f: (f[2]-f[0])*(f[3]-f[1]))
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
                        print("🚫 최대 높이 도달 → 종료")
                        break
                else:
                    state = "down"
                    stable_count = 0
                    moveDown(WITH_FACE)
                    
                    if exceed_min_height():
                        print("🚫 최소 높이 도달 → 종료")
                        break
            else:
                # 사람 검출
                person = detect_person_ssd(person_interpreter, frame, PERSON_SCORE_TH)
                stable_count = 0
                
                if person is None:
                    state = "hint_down"
                    moveDown(WITHOUT_FACE)
                    
                    if exceed_min_height():
                        print("🚫 최소 높이 도달 → 종료")
                        break
                else:
                    detection_found = True
                    x0, y0, x1, y1 = person
                    
                    if y0 <= 0.05:
                        state = "hint_up"
                        moveUp(WITHOUT_FACE)
                        
                        if exceed_max_height():
                            print("🚫 최대 높이 도달 → 종료")
                            break
                    elif y1 >= 0.95:
                        state = "hint_down"
                        moveDown(WITHOUT_FACE)
                        
                        if exceed_min_height():
                            print("🚫 최소 높이 도달 → 종료")
                            break
                    else:
                        y_center = (y0 + y1) * 0.5
                        state = "up" if y_center < target_y - deadband else "down"

            # 감지 시간 업데이트
            if detection_found:
                last_detection_time = time.time()
            
            # 타임아웃 체크
            if time.time() - last_detection_time > NO_DETECTION_TIMEOUT:
                print(f"⏱️ {NO_DETECTION_TIMEOUT}초간 미감지 → 타임아웃")
                break

            # 자동 종료 체크
            if state == "center" and stable_count >= STABLE_FRAMES:
                if stable_start_time is None:
                    stable_start_time = time.time()
                    print("✅ 안정화 감지 - 타이머 시작")
                elif time.time() - stable_start_time >= AUTO_EXIT_STABLE_TIME:
                    print(f"🎉 {AUTO_EXIT_STABLE_TIME}초간 안정화 완료!")
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
                    print("No face, person near top → Go up")
                elif state == "hint_down":
                    print("No face, person missing/near bottom → Go down")
                else:
                    print("Searching…")
                last_print_t = now
                last_state = state

    except KeyboardInterrupt:
        print("KeyboardInterrupt - 중단")
    except Exception as e:
        print(f"[Height] 예외: {e}")
        import traceback
        traceback.print_exc()
        return "error"
    finally:
        cap.release()
        cleanup_motor()
        
        if exceed_max_height() or exceed_min_height():
            return "limit_reached"
        elif time.time() - last_detection_time > NO_DETECTION_TIMEOUT:
            return "timeout"
        else:
            return "complete"


async def ws_client():
    """WebSocket 클라이언트"""
    global should_stop, ws_connection
    
    result_status = None
    
    try:
        print(f"[Height Worker] Hub 연결 시도 중... ({HUB_WS_URL})")
        async with websockets.connect(HUB_WS_URL) as ws:
            ws_connection = ws
            print("[Height Worker] Hub 연결됨 (포트 8766)")
            
            def run_track():
                nonlocal result_status
                try:
                    result_status = track_height()
                    print(f"[Height Worker] track_height() 완료: {result_status}")
                except Exception as e:
                    print(f"[Height Worker] track_height() 예외: {e}")
                    import traceback
                    traceback.print_exc()
                    result_status = "error"
            
            track_thread = threading.Thread(target=run_track, daemon=True)
            track_thread.start()
            print("[Height Worker] track_thread 시작됨")
            
            await asyncio.sleep(1.0)
            
            if not track_thread.is_alive():
                print("[Height Worker] ⚠️ track_height() 즉시 종료")
                raise Exception("track_height failed to start")
            
            print("[Height Worker] 메인 루프 진입")
            try:
                while True:
                    # 1) 작업 스레드가 끝났는지 먼저 확인
                    if not track_thread.is_alive():
                        print("[Height Worker] 작업 완료 감지 → 결과 전송")
                        # 결과에 따라 서버로 통지
                        if should_stop:
                            await ws.send(json.dumps({"type": "HEIGHT_SET_CANCEL"}))
                            print("[Height Worker] CANCELLED 전송")
                        elif result_status == "limit_reached":
                            await ws.send(json.dumps({"type": "HEIGHT_SET_END"}))
                            print("[Height Worker] END 전송 (한계 도달)")
                        elif result_status == "timeout":
                            await ws.send(json.dumps({"type": "HEIGHT_SET_TIMEOUT"}))
                            print("[Height Worker] TIMEOUT 전송")
                        else:
                            await ws.send(json.dumps({"type": "HEIGHT_SET_END"}))
                            print("[Height Worker] END 전송 (정상 완료)")
                        break  # 루프 종료

                    # 2) 서버에서 오는 중단 명령 등 수신 (타임아웃으로 빠르게 폴링)
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=0.2)
                    except asyncio.TimeoutError:
                        continue  # 주기적으로 스레드 상태 재확인
                    except websockets.exceptions.ConnectionClosed:
                        print("[Height Worker] Hub 연결 끊김")
                        break

                    data = json.loads(raw)
                    msg_type = data.get("type")

                    if msg_type == "HEIGHT_SET_OFF":
                        print("[Height Worker] 중단 명령 수신")
                        should_stop = True
                        # track_thread는 내부에서 should_stop 보고 빠져나옴
                        # 여기서 바로 통지하지 말고 위의 완료 분기에서 일괄 전송
                        continue

            finally:
                print("[Height Worker] 스레드 종료 대기 중...")
                track_thread.join(timeout=3)
                print("[Height Worker] 스레드 종료 완료")
    
    except websockets.exceptions.WebSocketException as e:
        print(f"[Height Worker] ❌ WebSocket 연결 실패: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"[Height Worker] ❌ 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ws_connection = None
        print("[Height Worker] ws_client() 종료")                                
            
    #         track_thread.join(timeout=3)
            
    #         if should_stop:
    #             await ws.send(json.dumps({"type": "HEIGHT_SET_CANCEL"}))
    #             print("[Height Worker] CANCELLED 전송")
    #         elif result_status == "limit_reached":
    #             await ws.send(json.dumps({"type": "HEIGHT_SET_END"}))
    #             print("[Height Worker] END 전송 (한계 도달)")
    #         elif result_status == "timeout":
    #             await ws.send(json.dumps({"type": "HEIGHT_SET_TIMEOUT"}))
    #             print("[Height Worker] TIMEOUT 전송")
    #         else:
    #             await ws.send(json.dumps({"type": "HEIGHT_SET_END"}))
    #             print("[Height Worker] END 전송 (정상 완료)")
    
    # except Exception as e:
    #     print(f"[Height Worker] 오류: {e}")
    #     import traceback
    #     traceback.print_exc()
    # finally:
    #     ws_connection = None


if __name__ == "__main__":
    print("[Height Worker] 프로세스 시작")
    try:
        asyncio.run(ws_client())
    except Exception as e:
        print(f"[Height Worker] 메인 예외: {e}")
        import traceback
        traceback.print_exc()
    print("[Height Worker] 프로세스 종료")