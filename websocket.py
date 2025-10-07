import asyncio
import websockets
import subprocess
import json
import sys
from pathlib import Path
import os
from functools import partial

from tts_stt import (
    tts_play, 
    find_input_device_index,
    stt_once_with_error_handling,
    DEFAULT_CHAT_GUIDE,
    DEFAULT_ERROR_GUIDE,
    STT_SILENCE_SEC,
    STT_MAX_DURATION,
    STT_CALIB_SEC,
    STT_SENSITIVITY,
    STT_MIN_SPEECH_SEC,
    STT_ENGINE,
    STT_SAMPLE_RATE
)

from linear_actuator.linear_actuator_controller import on_shutdown
import atexit

PYTHON = sys.executable
BASE_DIR = Path(__file__).resolve().parent
PIR_WORKER = str(BASE_DIR / "pir_worker.py")
EYE_SCRIPT = str(BASE_DIR / "optimized_code.py")
TTS_STT_SCRIPT = str(BASE_DIR / "tts_stt.py")

workers = {"PIR": None}
clients = set()

# 런처 핸들
eye_proc = None
tts_proc = None

USE_ACK = False  # 필요 없으면 False
atexit.register(on_shutdown)    # 최종 프로그램 종료시에 리니어엑추에이터 높이 낮추기

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

def start_tts_stt():
    global tts_proc
    if is_running(tts_proc):
        return
    # 별도 프로세스로 실행 (요구 3,7 충족)
    tts_proc = subprocess.Popen([PYTHON, TTS_STT_SCRIPT],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)

# =================================

async def handle_client(websocket):
    global eye_proc, tts_proc
    print("클라이언트 연결됨")
    clients.add(websocket)
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

            elif msg_type == "PIR_OFF":
                await stop_pir(websocket)

            elif msg_type == "PIR_DETECTED":
                await broadcast_json({"type": "PIR_DETECTED"})

            # === 조정/보정 ===
            elif msg_type == "EYE_CALIB_ON":
                # 1) 아이트래킹 프로세스 실행(필요 시만)
                start_eye()
                # 2) 프로세스가 WS 붙을 시간을 주기 위해 2초 대기
                await asyncio.sleep(2.0)
                # 3) 이제 보정 트리거 브로드캐스트
                await broadcast_json({"type": "EYE_CALIB_ON"})
                if USE_ACK:
                    await send_json(websocket, {"type": "EYE_CALIB_ON_ACK"})

            elif msg_type == "MODE_SELECT_ON":
                # 마우스 제어 '강제 ON'
                await broadcast_json({"type": "MOUSE_ON"})
                if USE_ACK:
                    await send_json(websocket, {"type": "MODE_SELECT_ON_ACK"})

            # === 모드 선택 → 대화/일반/눈 ===
            elif msg_type == "CHAT_ORDER_ON":
                # 1) 실행 중이던 optimized_code.py 종료
                eye_proc = stop_proc(eye_proc)
                # 2) tts_stt.py 실행
                start_tts_stt()
                if USE_ACK:
                    await send_json(websocket, {"type": "CHAT_ORDER_ON_ACK"})
                # (선택) 프론트에 상태 알림
                await broadcast_json({"type": "TTS_ON_STARTED"})

            elif msg_type == "NORMAL_ORDER_ON":
                # 아이트래킹 종료
                eye_proc = stop_proc(eye_proc)
                if USE_ACK:
                    await send_json(websocket, {"type": "NORMAL_ORDER_ON_ACK"})

            elif msg_type == "EYE_ORDER_ON":
                # 주먹 인식 OFF + 마우스 제어 ON
                await broadcast_json({"type": "EYE_ORDER_ON"})
                await broadcast_json({"type": "MOUSE_ON"})
                if USE_ACK:
                    await send_json(websocket, {"type": "EYE_ORDER_ON_ACK"})

            # === 대화주문 중 TTS/STT ===
            elif msg_type == "TTS_ON":
                text  = data.get("message") or data.get("text") or ""
                if not str(text).strip():
                    await send_json(websocket, {"type": "TTS_ERROR", "message": "Missing TTS text"})
                    continue
                lang  = data.get("lang", "ko-KR")
                voice = data.get("voice", None)
                rate  = float(data.get("speakingRate", 1.0))
                pitch = float(data.get("pitch", 0.0))
                enc   = data.get("audioEncoding", "LINEAR16")

                loop = asyncio.get_running_loop()
                try:
                    print(f"[TTS] 응답 시작: \"{text[:40]}...\" ({enc})")
                    await loop.run_in_executor(None, partial(tts_play, text, lang, voice, rate, pitch, None, enc))
                except Exception as e:
                    print(f"[TTS] 오류: {e}", file=sys.stderr)
                    await send_json(websocket, {"type": "TTS_ERROR", "message": str(e)})
                else:
                    print("[TTS] 응답 종료 → TTS_OFF 전송")
                    await send_json(websocket, {"type": "TTS_OFF"})

            elif msg_type == "STT_ON":
                duration = int(data.get("duration", STT_MAX_DURATION))
                sample_rate = int(data.get("sampleRate", STT_SAMPLE_RATE))
                language_code = data.get("languageCode", "ko-KR")
                device_idx = data.get("deviceIndex", None)

                if device_idx is None:
                    device_idx = find_input_device_index()
                    print(f"[STT] deviceIndex 자동 선택: {device_idx}")

                if device_idx is None:
                    await send_json(websocket, {"type": "STT_ERROR", "message": "No input-capable device found"})
                    continue

                async def notify_stt_error():
                    print("[WebSocket] STT_ERR 전송")
                    await send_json(websocket, {"type": "STT_ERR"})

                async def notify_error_end():
                    print("[WebSocket] ERR_END 전송")
                    await send_json(websocket, {"type": "ERR_END"})

                def run_stt_with_error():
                    def sync_callback(coro):
                        future = asyncio.run_coroutine_threadsafe(coro, loop)
                        future.result(timeout=5)
                    return stt_once_with_error_handling(
                        on_error_callback=lambda: sync_callback(notify_stt_error()),
                        on_error_end_callback=lambda: sync_callback(notify_error_end()),
                        error_guide_text=DEFAULT_ERROR_GUIDE,
                        error_guide_lang=language_code,
                        mode="auto",
                        duration=duration,
                        sample_rate=sample_rate,
                        language_code=language_code,
                        device=device_idx,
                        silence_sec=float(data.get("silence", STT_SILENCE_SEC)),
                        max_duration=float(data.get("maxDuration", STT_MAX_DURATION)),
                        calib_sec=float(data.get("calib", STT_CALIB_SEC)),
                        sensitivity=float(data.get("sensitivity", STT_SENSITIVITY)),
                        min_speech_sec=float(data.get("minSpeech", STT_MIN_SPEECH_SEC)),
                        engine=data.get("engine", STT_ENGINE)
                    )

                loop = asyncio.get_running_loop()
                try:
                    print(f"[STT] (auto) 녹음 시작: max {duration}s @ {sample_rate}Hz (device={device_idx})")
                    transcript, success = await loop.run_in_executor(None, run_stt_with_error)
                    if success:
                        print(f"[STT] 성공 결과: \"{transcript}\"")
                        await send_json(websocket, {"type": "STT_OFF", "message": transcript})
                    else:
                        print("[STT] 실패 처리 완료 (프론트 대기 중)")
                except Exception as e:
                    print(f"[STT] 예외 발생: {e}", file=sys.stderr)
                    await send_json(websocket, {"type": "STT_ERROR", "message": str(e)})

            elif msg_type == "ALL_RESET":
                # 모든 종료
                eye_proc = stop_proc(eye_proc)
                tts_proc = stop_proc(tts_proc)
                await stop_pir()
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

async def main():
    print("서버 시작됨 (0.0.0.0:8765)")
    async with websockets.serve(handle_client, "0.0.0.0", 8765):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
