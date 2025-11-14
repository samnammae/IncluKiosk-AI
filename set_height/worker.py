"""높이 조절 워커 - WebSocket 클라이언트 및 메인 로직"""
import cv2
import time
import asyncio
import websockets
import json
import threading
import tflite_runtime.interpreter as tflite

from . import config
from . import detection
from .linear_actuator.linear_actuator_controller import (
    init_motor,
    cleanup_motor,
    moveUp,
    moveDown,
    exceed_max_height,
    exceed_min_height
)

# 전역 변수
should_stop = False
ws_connection = None


def debug_log(msg):
    """디버그 로그 출력"""
    if config.DEBUG:
        print(f"[DEBUG] {msg}")


def track_height():
    """높이 추적 메인 로직"""
    global should_stop
    
    # 모터 초기화
    try:
        print("[Height] 모터 초기화 중...")
        init_motor()
        print("[Height] ✅ 모터 초기화 완료")
    except Exception as e:
        print(f"[Height] ❌ 모터 초기화 실패: {e}")
        return "error"

    # 카메라 초기화
    print("[Height] 카메라 초기화 중...")
    cap = cv2.VideoCapture(config.CAM_INDEX)
    if not cap.isOpened():
        print(f"[Height] ❌ 카메라 열기 실패 (인덱스: {config.CAM_INDEX})")
        try:
            import glob
            print("[Height] /dev/video* =", glob.glob("/dev/video*"))
        except:
            pass
        cleanup_motor()
        return "error"
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
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

    # AI 모델 로드
    print("[Height] AI 모델 로드 중...")
    try:
        face_interpreter = tflite.Interpreter(
            model_path=config.FACE_MODEL,
            experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
        )
        face_interpreter.allocate_tensors()
        print("[Height] ✅ 얼굴 검출 모델 로드 완료 (EdgeTPU)")
        
        person_interpreter = tflite.Interpreter(
            model_path=config.PERSON_MODEL,
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
    
    # 추적 변수 초기화
    target_y = min(0.9, max(0.1, 0.5 - config.TARGET_OFFSET_PCT))
    deadband = config.DEADBAND_PCT
    ema_y = None
    stable_count = 0
    stable_start_time = None
    last_detection_time = time.time()
    last_print_t = 0
    last_state = None
    
    limit_reached_without_face = False # 높이 조절 실패 여부를 위한 플래그

    try:
        while not should_stop:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[Height] ⚠️ 카메라 읽기 실패")
                time.sleep(0.1)
                continue

            # 얼굴 감지
            faces = detection.detect_faces(
                face_interpreter, 
                frame, 
                config.MIN_DET_CONF,
                debug=config.DEBUG
            )

            state = None
            detection_found = False
            
            if faces:
                # 얼굴이 감지된 경우
                detection_found = True
                # 가장 큰 얼굴 선택
                face = max(faces, key=lambda f: (f[2]-f[0])*(f[3]-f[1]))
                xmin, ymin, xmax, ymax, score = face
                
                # 얼굴 중심 계산
                y_center = (ymin + ymax) * 0.5
                
                # EMA 적용
                if ema_y is None:
                    ema_y = y_center
                else:
                    ema_y = config.EMA_ALPHA * y_center + (1 - config.EMA_ALPHA) * ema_y
                
                diff = ema_y - target_y
                if abs(diff) <= deadband: 
                    # 중앙 안정
                    state = "center"
                    stable_count += 1
                elif diff < 0:
                    # 위쪽에 있음 -> 내려야 함
                    state = "up"
                    stable_count = 0
                    # print(f"[MOVE UP TEST] diff={diff:.3f} → 얼굴이 목표보다 아래에 있음 (↑)")
                    moveUp(config.WITH_FACE)
                    
                    if exceed_max_height():
                        print("🚫 최대 높이 도달 → 종료")
                        break
                else:
                    # 아래쪽에 있음 -> 올라야 함
                    state = "down"
                    stable_count = 0
                    # print(f"[MOVE DOWN TEST] diff={diff:.3f} → 얼굴이 목표보다 위에 있음 (↓)")
                    moveDown(config.WITH_FACE)
                    
                    if exceed_min_height():
                        print("🚫 최소 높이 도달 → 종료")
                        break
            else:
                # 얼굴 없음 -> 사람 감지
                person = detection.detect_person(
                    person_interpreter,
                    frame,
                    config.PERSON_SCORE_TH,
                    debug=config.DEBUG
                )
                stable_count = 0
                
                if person is None:
                    # 사람도 없음 -> 내려감
                    state = "hint_down"
                    moveDown(config.WITHOUT_FACE)
                    
                    if exceed_min_height():
                        print("🚫 최소 높이 도달 → 종료 (얼굴 없음)")
                        limit_reached_without_face = True
                        break
                else:
                    # 사람은 있음
                    detection_found = True
                    x0, y0, x1, y1 = person
                    
                    if y0 <= 0.05:
                        # 사람이 화면 위쪽에 있음
                        state = "hint_up"
                        moveUp(config.WITHOUT_FACE)
                        
                        if exceed_max_height():
                            print("🚫 최대 높이 도달 → 종료 (얼굴 없음)")
                            limit_reached_without_face = True
                            break
                    elif y1 >= 0.95:
                        # 사람이 화면 아래쪽에 있음
                        state = "hint_down"
                        moveDown(config.WITHOUT_FACE)
                        
                        if exceed_min_height():
                            print("🚫 최소 높이 도달 → 종료 (얼굴 없음)")
                            limit_reached_without_face = True
                            break
                    else:
                        # 사람이 화면 중간에 있음
                        y_center = (y0 + y1) * 0.5
                        state = "up" if y_center < target_y - deadband else "down"

            # 감지 시간 업데이트
            if detection_found:
                last_detection_time = time.time()
            
            # 타임아웃 체크
            if time.time() - last_detection_time > config.NO_DETECTION_TIMEOUT:
                print(f"⏱️ {config.NO_DETECTION_TIMEOUT}초간 미감지 → 타임아웃")
                break

            # 자동 종료 체크
            if state == "center" and stable_count >= config.STABLE_FRAMES:
                if stable_start_time is None:
                    stable_start_time = time.time()
                    print("✅ 안정화 감지 - 타이머 시작")
                elif time.time() - stable_start_time >= config.AUTO_EXIT_STABLE_TIME:
                    print(f"🎉 {config.AUTO_EXIT_STABLE_TIME}초간 안정화 완료!")
                    break
            else:
                stable_start_time = None

            # # 상태 출력
            # now = time.time()
            # if now - last_print_t >= config.PRINT_EVERY:
            #     if state == "center":
            #         if stable_count >= config.STABLE_FRAMES:
            #             if last_state != "center":
            #                 print("Centered ✅ (stable)")
            #         else:
            #             print(f"Centered… ({stable_count}/{config.STABLE_FRAMES})")
            #     elif state == "up":
            #         print("Go down! (face/person above center)")
            #     elif state == "down":
            #         print("Go up! (face/person below center)")
            #     elif state == "hint_up":
            #         print("No face, person near top → Go up")
            #     elif state == "hint_down":
            #         print("No face, person missing/near bottom → Go down")
            #     else:
            #         print("Searching…")
            #     last_print_t = now
            #     last_state = state

    except KeyboardInterrupt:
        print("KeyboardInterrupt - 중단")
    except Exception as e:
        print(f"[Height] 예외: {e}")
        import traceback
        traceback.print_exc()
        return "error"
    finally:
        print("=== 정리 시작 ===")
        
        try:
            print("카메라 릴리즈 시작...")
            cap.release()
            print("카메라 릴리즈 완료!")
        except Exception as e:
            print(f"⚠️ 카메라 릴리즈 실패: {e}")
        
        try:
            print("모터 정리 시작...")
            cleanup_motor()
            print("✅ 모터 정리 완료")
        except Exception as e:
            print(f"⚠️ 모터 정리 실패: {e}")
            import traceback
            traceback.print_exc()
        
        # 반환값 결정
        if limit_reached_without_face:
            return "limit_reached_no_face" 
        if exceed_max_height() or exceed_min_height():
            return "limit_reached"
        elif time.time() - last_detection_time > config.NO_DETECTION_TIMEOUT:
            return "timeout"
        else:
            return "complete"


async def ws_client():
    """WebSocket 클라이언트"""
    global should_stop, ws_connection
    
    result_status = None
    
    try:
        print(f"[Height Worker] Hub 연결 시도 중... ({config.HUB_WS_URL})")
        async with websockets.connect(config.HUB_WS_URL) as ws:
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
            
            # 높이 추적 스레드 시작
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
                    # 작업 스레드 완료 확인
                    if not track_thread.is_alive():
                        print("[Height Worker] 작업 완료 감지 → 결과 전송")
                        
                        # 결과에 따라 서버로 통지
                        if should_stop:
                            # ALL_RESET 등으로 강제 종료된 경우 → 허브가 이미 알고 있으니 별도 통지 X
                            print("[Height Worker] 외부 명령으로 중단됨")
                            break
                        elif result_status == "limit_reached_no_face":
                            await ws.send(json.dumps({"type": "HEIGHT_SET_ERR"}))
                            print("[Height Worker] ERR 전송 (한계 도달 + 얼굴 없음)")
                        elif result_status == "limit_reached":
                            await ws.send(json.dumps({"type": "HEIGHT_SET_END"}))
                            print("[Height Worker] END 전송 (한계 도달)")
                        elif result_status == "timeout":
                            await ws.send(json.dumps({"type": "HEIGHT_SET_TIMEOUT"}))
                            print("[Height Worker] TIMEOUT 전송")
                        else:
                            await ws.send(json.dumps({"type": "HEIGHT_SET_END"}))
                            print("[Height Worker] END 전송 (정상 완료)")
                            
                        await asyncio.sleep(1.5)
                        print("[Height Worker] 메시지 전송 대기 완료")
                        break

                    # 서버 명령 수신 (0.2초 타임아웃)
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=0.2)
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        print("[Height Worker] Hub 연결 끊김")
                        break

                    data = json.loads(raw)
                    msg_type = data.get("type")

                    if msg_type == "HEIGHT_SET_OFF":
                        print("[Height Worker] 중단 명령 수신")
                        should_stop = True

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


def main():
    """메인 엔트리 포인트"""
    print("[Height Worker] 프로세스 시작")
    try:
        asyncio.run(ws_client())
    except Exception as e:
        print(f"[Height Worker] 메인 예외: {e}")
        import traceback
        traceback.print_exc()
    print("[Height Worker] 프로세스 종료")


if __name__ == "__main__":
    main()