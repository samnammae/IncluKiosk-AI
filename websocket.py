import asyncio
import websockets
import subprocess
import json
import sys
from pathlib import Path
import os
from functools import partial
import tts_stt

from linear_actuator.linear_actuator_controller import on_shutdown
import atexit

stt_fail_count = 0  # TTS 무응답(실패) 횟수 카운터

PYTHON = sys.executable
BASE_DIR = Path(__file__).resolve().parent
PIR_WORKER = str(BASE_DIR / "pir_sensor" / "pir_worker.py")
EYE_SCRIPT = str(BASE_DIR / "eye_tracking_worker.py")
HEIGHT_WORKER = str(BASE_DIR / "height_worker.py")

workers = {"PIR": None}
clients = set()  # 프론트엔드 클라이언트들

# 런처 핸들
eye_proc = None
height_proc = None
height_set_processing = False  # 처리 중 플래그

# === 내부 통신용 === #
internal_worker_ws = None  # eye_tracking_worker의 WebSocket 연결
frontend_ws = None    # 프론트엔드의 WebSocket 연결

USE_ACK = False  # 필요 없으면 False
atexit.register(on_shutdown)    # 최종 프로그램 종료시에 리니어액추에이터 높이 낮추기

async def send_json(ws, payload: dict):
    try:
        await ws.send(json.dumps(payload, ensure_ascii=False))
    except Exception:
        clients.discard(ws)

async def broadcast_json(payload: dict):
    dead = []
    raw = json.dumps(payload, ensure_ascii=False)
    for ws in list(clients):
        try:
            await ws.send(raw)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)

# === 라즈베리파이 내부 워커에게 메시지 전송 === #
async def send_to_internal_worker(payload: dict):
    if internal_worker_ws:
        try:
            await internal_worker_ws.send(json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            print(f"[Hub→Eye] 전송 실패: {e}")

# === front로 메시지 전송 === #
async def send_to_front(payload: dict):
    if frontend_ws:
        try:
            await frontend_ws.send(json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            print(f"[Hub→Front] 전송 실패: {e}")
            clients.discard(frontend_ws)
# === pir 센서 관련 ===
async def start_pir(websocket=None):
    if workers["PIR"] and (workers["PIR"].poll() is None):
        if USE_ACK and websocket:
            await send_json(websocket, {"type": "PIR_ON_ACK", "status": "already_running"})
        return
    proc = subprocess.Popen([PYTHON, PIR_WORKER])
    workers["PIR"] = proc
    if USE_ACK and websocket:
        await send_json(websocket, {"type": "PIR_ON_ACK", "status": "started"})

async def stop_pir(websocket=None):
    proc = workers.get("PIR")
    if not proc or (proc.poll() is not None):
        workers["PIR"] = None
        if USE_ACK and websocket:
            await send_json(websocket, {"type": "PIR_OFF_ACK", "status": "already_stopped"})
        return
    proc.terminate()
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, proc.wait, 3)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    finally:
        workers["PIR"] = None
        if USE_ACK and websocket:
            await send_json(websocket, {"type": "PIR_OFF_ACK", "status": "stopped"})

async def clear_pir_state_and_notify_frontend():
    workers["PIR"] = None
    await broadcast_json({"type": "PIR_OFF"})

# ====== 프로세스 런처 유틸 ======
def is_running(p):
    return (p is not None) and (p.poll() is None)

def stop_proc(p):
    if not is_running(p):
        return None
    try:
        p.terminate()
        try:
            p.wait(timeout=3)
        except Exception:
            p.kill()
    except Exception:
        pass
    return None

def start_eye():
    global eye_proc
    if is_running(eye_proc):
        return
    # sudo -E 필요
    eye_proc = subprocess.Popen(["sudo", "-E", "python", EYE_SCRIPT],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)

# === 높이 조절 관련 ===
async def start_height_worker():
    """높이 조절 워커 시작"""
    global height_proc, height_set_processing
    
    # 이미 처리 중이면 무시
    if height_set_processing:
        print("[Height] 이미 처리 중 (디바운싱)")
        return
    
    if is_running(height_proc):
        print("[Height] 이미 실행 중")
        return
    
    # 처리 시작 플래그 설정
    height_set_processing = True
    
    print("[Height] 워커 시작")
    height_proc = subprocess.Popen(
        ["sudo", "-E", "python", HEIGHT_WORKER],
        stdout=sys.stdout,  # 표준 출력으로 에러 확인
        stderr=sys.stderr   # 표준 에러로 에러 확인
    )
    print(f"[Height] 프로세스 PID: {height_proc.pid}")
    height_set_processing = False

# 높이 조절 워커 중단
async def stop_height_worker():
    """높이 조절 중단 (graceful shutdown)"""
    global height_proc, height_set_processing
    if not is_running(height_proc):
        print("[Height] 이미 중단됨")
        height_set_processing = False  # 플래그 리셋
        return
    
    print("[Height] 중단 명령 전송")
    # WebSocket으로 중단 명령 전송
    await send_to_internal_worker({"type": "HEIGHT_SET_OFF"})
    
    # 프로세스 종료 대기 (최대 5초)
    try:
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(None, height_proc.wait),
            timeout=5.0
        )
        print("[Height] 정상 종료됨")
    except asyncio.TimeoutError:
        print("[Height] 5초 대기 후 강제 종료")
        height_proc.terminate()
        try:
            height_proc.wait(timeout=2)
        except:
            height_proc.kill()
    finally:
        height_proc = None
        height_set_processing = False  # 🆕 플래그 리셋

# 대화 주문 핸들러
async def handle_chat_order_on(websocket=None):
    print("▣ ▣ ▣ CHAT_ORDER_ON 처리 로직 시작")
    
    # 1. 아이트래킹 정지
    global eye_proc
    eye_proc = stop_proc(eye_proc)
    await send_to_internal_worker({"type": "STOP_ALL"})
    
    # 2. 안내 TTS 재생
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, tts_stt.play_chat_guide_message)
        print("[TTS] 안내 종료 → TTS_OFF 전송")
        await send_to_front({"type": "TTS_OFF"})
    except Exception as e:
        print(f"[TTS] 안내 실패: {e}", file=sys.stderr)
        await send_to_front({"type": "TTS_ERROR", "message": f"Guide TTS failed: {e}"})

# ====== TTS/STT 핸들러 ======
async def handle_tts_on(websocket, data):
    """TTS_ON 메시지 처리"""
    text = data.get("message") or data.get("text") or ""
    if not str(text).strip():
        await send_json(websocket, {"type": "TTS_ERROR", "message": "Missing TTS text"})
        return
    
    lang = data.get("lang", "ko-KR")
    voice = data.get("voice", None)
    rate = float(data.get("speakingRate", 1.0))
    pitch = float(data.get("pitch", 0.0))
    enc = data.get("audioEncoding", "LINEAR16")
    
    loop = asyncio.get_running_loop()
    try:
        print(f"[TTS] 응답 시작: \"{text[:40]}...\" ({enc})")
        await loop.run_in_executor(
            None,
            partial(tts_stt.tts_play, text, lang, voice, rate, pitch, None, enc)
        )
        print("[TTS] 응답 종료 → TTS_OFF 전송")
        await send_json(websocket, {"type": "TTS_OFF"})
    except Exception as e:
        print(f"[TTS] 오류: {e}", file=sys.stderr)
        await send_json(websocket, {"type": "TTS_ERROR", "message": str(e)})


async def handle_stt_on(websocket, data, loop):
    """STT_ON 메시지 처리"""
    global stt_fail_count

    # 1. 파라미터 추출
    duration = int(data.get("duration", tts_stt.STT_MAX_DURATION))
    sample_rate = int(data.get("sampleRate", tts_stt.STT_SAMPLE_RATE))
    language_code = data.get("languageCode", "ko-KR")
    device_idx = data.get("deviceIndex")
    
    # 2. 장치 자동 선택
    if device_idx is None:
        device_idx = tts_stt.find_input_device_index()
        print(f"[STT] deviceIndex 자동 선택: {device_idx}")
    
    if device_idx is None:
        await send_json(websocket, {"type": "STT_ERROR", "message": "No input-capable device found"})
        return
    
    # 3. STT 실행
    def run_stt():
        return tts_stt.stt_once(
            mode="auto",
            duration=duration,
            sample_rate=sample_rate,
            language_code=language_code,
            device=device_idx,
            silence_sec=float(data.get("silence", tts_stt.STT_SILENCE_SEC)),
            max_duration=float(data.get("maxDuration", tts_stt.STT_MAX_DURATION)),
            calib_sec=float(data.get("calib", tts_stt.STT_CALIB_SEC)),
            sensitivity=float(data.get("sensitivity", tts_stt.STT_SENSITIVITY)),
            min_speech_sec=float(data.get("minSpeech", tts_stt.STT_MIN_SPEECH_SEC)),
            engine=data.get("engine", tts_stt.STT_ENGINE)
        )
    
    try:
        print(f"[STT] (auto) 녹음 시작: max {duration}s @ {sample_rate}Hz (device={device_idx})")
        transcript = await loop.run_in_executor(None, run_stt)
        
        if transcript and transcript.strip():
            # 성공!
            stt_fail_count = 0
            print(f"[STT] 성공 결과: \"{transcript}\"")
            await send_json(websocket, {"type": "STT_OFF", "message": transcript})
        else:
            # 실패 - 모든 처리는 handle_stt_failure에서
            await handle_stt_failure(websocket, loop, language_code)
    
    except Exception as e:
        print(f"[STT] 예외 발생: {e}", file=sys.stderr)
        await send_json(websocket, {"type": "STT_ERROR", "message": str(e)})


async def handle_stt_failure(websocket, loop, language_code="ko-KR"):
    """STT 실패 처리 (통합)"""
    global stt_fail_count
    stt_fail_count += 1
    print(f"[STT] 실패 처리 ({stt_fail_count}/2)")
    
    # 2회 실패 시 주문 취소
    if stt_fail_count >= 2:
        print("[ORDER] 무응답 2회 도달 → 주문 취소 플로우 실행")
        
        await send_to_front({"type": "ORDER_CANCEL"})
        
        try:
            await loop.run_in_executor(None, tts_stt.play_cancel_guide_message)
        except Exception as e:
            print(f"[TTS] 취소 안내 실패: {e}", file=sys.stderr)
        
        await send_to_front({"type": "CANCEL_END"})
        
        stt_fail_count = 0
    
    # 1회 실패 시 오류 안내
    else:
        # 1. STT_ERR 전송
        await send_to_front({"type": "STT_ERR"})
        
        # 2. 오류 안내 TTS 재생
        try:
            await loop.run_in_executor(
                None, 
                partial(tts_stt.play_error_guide_message, lang=language_code)
            )
        except Exception as e:
            print(f"[TTS] 오류 안내 실패: {e}", file=sys.stderr)
        
        # 3. ERR_END 전송
        await send_json(websocket, {"type": "ERR_END"})

async def handle_frontend(websocket):
    global eye_proc, frontend_ws, stt_fail_count
    print("클라이언트 연결됨")
    
    # ✅ 프론트 연결 저장
    frontend_ws = websocket
    clients.add(websocket)
    stt_fail_count = 0

    try:
        async for raw in websocket:
            print(f"받은 메시지: {raw}")
            try:
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError("JSON 객체가 아님")
            except Exception as e:
                await send_json(websocket, {"type": "ERROR", "message": f"Invalid JSON: {e}"})
                continue

            msg_type = data.get("type")
            msg_text = data.get("message")
            if not msg_type:
                await send_json(websocket, {"type": "ERROR", "message": "Missing 'type' field"})
                continue

            print(f"파싱된 type: {msg_type}")

            # === 잠금화면 / PIR ===
            if msg_type == "PIR_ON":
                await start_pir(websocket)

            elif msg_type == "PIR_DETECTED":
                print("▣ ▣ ▣ PIR_DETECTED(from pir-worker)")
                await send_to_front({"type": "PIR_DETECTED"})

            elif msg_type == "PIR_OFF":
                print("▣ ▣ ▣ PIR_OFF!!!")
                # 1. 내부 워커에게 종료 신호 전송
                await send_to_internal_worker({"type": "PIR_OFF"})
                
                # 2. 프로세스 종료 대기
                await asyncio.sleep(1.0)  # 워커가 정리할 시간 제공
                
                # 3. 프로세스 강제 종료 (필요시)
                await stop_pir(websocket)
                
                # 4. 프론트에게 완료 알림
                await send_to_front({"type": "PIR_END"})

            # === 높이조절 ===
            elif msg_type == "HEIGHT_SET_ON":
                print("▣ ▣ ▣ HEIGHT_SET_ON!!!")
                await start_height_worker()

            elif msg_type == "HEIGHT_SET_CANCEL":
                print("▣ ▣ ▣ HEIGHT_SET_CANCEL (사용자 중단)")
                await stop_height_worker()
                await send_to_front({"type": "HEIGHT_SET_CANCEL"})

            # === 조정/보정 ===
            elif msg_type == "EYE_CALIB_ON":
                print("▣ ▣ ▣ EYE_CALIB_ON!!!")
                # 1) 아이트래킹 프로세스 실행(필요 시만)
                start_eye()
                # 2) 프로세스가 WS 붙을 시간을 주기 위해 2초 대기
                await asyncio.sleep(2.0)
                # 3) 이제 보정 트리거 브로드캐스트
                await broadcast_json({"type": "EYE_CALIB_ON"})
                # === 새로 추가: eye_tracking_worker로 캘리브레이션 명령 전송 ===
                await send_to_internal_worker({"type": "EYE_CALIB_ON"})
                if USE_ACK:
                    await send_json(websocket, {"type": "EYE_CALIB_ON_ACK"})

            elif msg_type == "MODE_SELECT_ON":
                print("▣ ▣ ▣ MODE_SELECT_ON!!!")
                # 마우스 제어 '강제 ON'
                await broadcast_json({"type": "MOUSE_ON"})
                # === 새로 추가: eye_tracking_worker로 마우스 ON 명령 전송 ===
                await send_to_internal_worker({"type": "MOUSE_ON"})
                if USE_ACK:
                    await send_json(websocket, {"type": "MODE_SELECT_ON_ACK"})

            # === 모드 선택 → 대화/일반/눈 ===
            elif msg_type == "CHAT_ORDER_ON":
                print("▣ ▣ ▣ CHAT_ORDER_ON!!!")
                await handle_chat_order_on(websocket)

            elif msg_type == "NORMAL_ORDER_ON":
                print("▣ ▣ ▣ NORMAL_ORDER_ON!!!")
                # 아이트래킹 종료
                eye_proc = stop_proc(eye_proc)
                # === 새로 추가: eye_tracking_worker로 정지 명령 전송 ===
                await send_to_internal_worker({"type": "STOP_ALL"})
                if USE_ACK:
                    await send_json(websocket, {"type": "NORMAL_ORDER_ON_ACK"})

            elif msg_type == "EYE_ORDER_ON":
                print("▣ ▣ ▣ EYE_ORDER_ON!!!")
                # 주먹 인식 OFF + 마우스 제어 ON
                await broadcast_json({"type": "EYE_ORDER_ON"})
                await broadcast_json({"type": "MOUSE_ON"})
                # === 새로 추가: eye_tracking_worker로 명령 전송 ===
                await send_to_internal_worker({"type": "EYE_ORDER_ON"})
                await send_to_internal_worker({"type": "MOUSE_ON"})
                if USE_ACK:
                    await send_json(websocket, {"type": "EYE_ORDER_ON_ACK"})

            # === 대화주문 중 TTS/STT ===
            elif msg_type == "TTS_ON":
                print("▣ ▣ ▣ TTS_ON!!!")
                await handle_tts_on(websocket, data)

            elif msg_type == "STT_ON":
                print("▣ ▣ ▣ STT_ON!!!")
                loop = asyncio.get_running_loop()
                await handle_stt_on(websocket, data, loop)

            elif msg_type == "ALL_RESET":
                print("▣ ▣ ▣ ALL_RESET!!!")
                # 모든 종료
                eye_proc = stop_proc(eye_proc)
                await stop_pir()
                await stop_height_worker()
                # === 새로 추가: eye_tracking_worker로 정지 명령 전송 ===
                await send_to_internal_worker({"type": "STOP_ALL"})
                # PIR 시작
                await start_pir()
                if USE_ACK:
                    await send_json(websocket, {"type": "ALL_RESET_ACK"})
                else:
                    await send_json(websocket, {"type": "ALL_RESET"})

            else:
                await send_json(websocket, {"type": "ERROR", "message": f"Unknown type: {msg_type}"})

    except websockets.exceptions.ConnectionClosed:
        print("클라이언트 연결 끊김")
    finally:
        clients.discard(websocket)
        stt_fail_count = 0
        frontend_ws = None

# === 새로 추가: 라즈베리파이 내부 통신 핸들러 ===
async def handle_internal_worker(websocket):
    """eye_tracking_worker 및 height_worker의 WebSocket 연결 처리 (내부 통신용)"""
    global internal_worker_ws, height_proc, height_set_processing
    internal_worker_ws = websocket
    print("[Internal Worker] 연결됨 (포트 8766)")
    
    try:
        async for raw in websocket:
            data = json.loads(raw)
            msg_type = data.get("type")
            print(f"[Worker→Hub] 수신: {msg_type}")
            
            # PIR 감지
            if msg_type == "PIR_DETECTED":
                print("▣ ▣ ▣ PIR_DETECTED(from pir-worker)")
                await send_to_front({"type": "PIR_DETECTED"})
                
            # PIR 워커 정상 종료 (WebSocket 연결 끊김 감지)
            elif msg_type == "PIR_WORKER_EXIT":
                print("[Hub] PIR 워커 정상 종료")
                await send_to_front({"type": "PIR_END"})
                workers["PIR"] = None
            
            # 주먹 감지
            elif msg_type == "FIST_DETECTED":
                print("▣ ▣ ▣ FIST_DETECTED(from eye-worker)")
                await send_to_front({"type": "FIST_DETECTED"})
            
            # 높이 조절 완료
            elif msg_type == "HEIGHT_SET_END":
                print("[Hub] ✅ 높이 조절 정상 완료")
                await send_to_front({"type": "HEIGHT_SET_END"})
                height_proc = None  # 프로세스 핸들 정리
                height_set_processing = False  # 플래그 리셋
            
            # 높이 조절 타임아웃
            elif msg_type == "HEIGHT_SET_TIMEOUT":
                print("[Hub] ⚠️ 높이 조절 타임아웃 (30초간 미감지)")
                await send_to_front({"type": "HEIGHT_SET_CANCEL"})
                height_proc = None
                height_set_processing = False  # 플래그 리셋
            
            # 높이 조절 취소됨
            elif msg_type == "HEIGHT_SET_CANCEL":
                print("[Hub] 🚫 높이 조절 취소됨")
                await send_to_front({"type": "HEIGHT_SET_CANCEL"})
                height_proc = None
                height_set_processing = False  # 플래그 리셋
    
    except websockets.exceptions.ConnectionClosed:
        print("[Internal Worker] 연결 끊김")
        # PIR 워커가 비정상 종료된 경우도 처리
        if workers.get("PIR") and workers["PIR"].poll() is None:
            print("[Hub] PIR 워커 비정상 종료 감지")
            await send_to_front({"type": "PIR_END"})
            workers["PIR"] = None
    finally:
        internal_worker_ws = None

async def main():
    print("=" * 60)
    print("서버 시작됨")
    print("  - 프론트엔드: ws://0.0.0.0:8765")
    print("  - Internal Workers: ws://localhost:8766")
    print("=" * 60)
    
    # 프론트엔드용 서버 (포트 8765)
    frontend_server = websockets.serve(handle_frontend, "0.0.0.0", 8765)
    
    # eye_tracking_worker + height_worker용 내부 서버 (포트 8766)
    internal_server = websockets.serve(handle_internal_worker, "localhost", 8766)
    
    async with frontend_server, internal_server:
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())