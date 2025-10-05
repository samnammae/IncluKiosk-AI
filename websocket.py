import asyncio
import websockets
import subprocess
import json
import sys
from pathlib import Path
import os
from functools import partial

from tts_stt import tts_play, stt_once, tts_guide_default, find_input_device_index

from linear_actuator.linear_actuator_controller import on_shutdown
import atexit

PYTHON = sys.executable
BASE_DIR = Path(__file__).resolve().parent
PIR_WORKER = str(BASE_DIR / "pir_worker.py")

workers = {"PIR": None}
clients = set()

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

async def start_pir(websocket):
    if workers["PIR"] and (workers["PIR"].poll() is None):
        if USE_ACK:
            await send_json(websocket, {"type": "PIR_ON_ACK", "status": "already_running"})
        return
    proc = subprocess.Popen([PYTHON, PIR_WORKER])
    workers["PIR"] = proc
    if USE_ACK:
        await send_json(websocket, {"type": "PIR_ON_ACK", "status": "started"})

async def stop_pir(websocket):
    proc = workers.get("PIR")
    if not proc or (proc.poll() is not None):
        workers["PIR"] = None
        if USE_ACK:
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
        if USE_ACK:
            await send_json(websocket, {"type": "PIR_OFF_ACK", "status": "stopped"})

async def clear_pir_state_and_notify_frontend():
    workers["PIR"] = None
    await broadcast_json({"type": "PIR_OFF"})

DEFAULT_CHAT_GUIDE = "안녕하세요. 음성으로 주문을 도와드릴게요."

async def handle_client(websocket):
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
                await broadcast_json({"type": "EYE_CALIB_ON"})

            elif msg_type == "MODE_SELECT_ON":
                subprocess.Popen([PYTHON, "-c", "print('MODE_SELECT stub')"])
                if USE_ACK:
                    await send_json(websocket, {"type": "MODE_SELECT_ON_ACK"})

            # === 모드 선택 → 대화/일반/눈 ===
            elif msg_type == "CHAT_ORDER_ON":
                subprocess.Popen([PYTHON, "-c", "print('STOP eye/fist workers stub')"])
                if USE_ACK:
                    await send_json(websocket, {"type": "CHAT_ORDER_ON_ACK"})

                guide_text = msg_text if (isinstance(msg_text, str) and msg_text.strip()) else DEFAULT_CHAT_GUIDE
                loop = asyncio.get_running_loop()
                try:
                    print(f"[TTS] 안내 시작: \"{guide_text}\" (WAV)")
                    # WAV로 합성해서 aplay 사용 (mpg123 미설치 환경 회피)
                    await loop.run_in_executor(None, partial(tts_play, guide_text, "ko-KR", None, 1.0, 0.0, None, "LINEAR16"))
                except Exception as e:
                    print(f"[TTS] 안내 실패: {e}", file=sys.stderr)
                    await send_json(websocket, {"type": "TTS_ERROR", "message": f"Guide TTS failed: {e}"})
                else:
                    print("[TTS] 안내 종료 → TTS_OFF 전송")
                    await send_json(websocket, {"type": "TTS_OFF"})

            elif msg_type == "NORMAL_ORDER_ON":
                subprocess.Popen([PYTHON, "-c", "print('NORMAL_ORDER stub')"])
                if USE_ACK:
                    await send_json(websocket, {"type": "NORMAL_ORDER_ON_ACK"})

            elif msg_type == "EYE_ORDER_ON":
                subprocess.Popen([PYTHON, "-c", "print('EYE_ORDER stub')"])
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
                enc   = data.get("audioEncoding", "LINEAR16")  # WAV 기본

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
                duration = int(data.get("duration", 60))         # auto모드에서 '최대' 시간
                sample_rate = int(data.get("sampleRate", 16000))
                language_code = data.get("languageCode", "ko-KR")
                device_idx = data.get("deviceIndex", None)

                # 장치 자동 탐색(Respeaker 키워드 매칭)
                if device_idx is None:
                    device_idx = find_input_device_index()
                    print(f"[STT] deviceIndex 자동 선택: {device_idx}")

                if device_idx is None:
                    await send_json(websocket, {"type": "STT_ERROR", "message": "No input-capable device found"})
                    continue

                loop = asyncio.get_running_loop()
                try:
                    print(f"[STT] (auto) 녹음 시작: max {duration}s @ {sample_rate}Hz (device={device_idx})")
                    transcript = await loop.run_in_executor(
                        None,
                        partial(
                            stt_once,
                            mode="auto",
                            duration=duration,
                            sample_rate=sample_rate,
                            language_code=language_code,
                            device=device_idx,
                            silence_sec=float(data.get("silence", 2.0)),
                            max_duration=float(data.get("maxDuration", duration)),
                            calib_sec=float(data.get("calib", 0.4)),
                            sensitivity=float(data.get("sensitivity", 2.0)),
                        ),
                    )
                    print(f"[STT] 결과: \"{transcript}\"")
                    await send_json(websocket, {"type": "STT_OFF", "message": transcript})
                except Exception as e:
                    print(f"[STT] 오류: {e}", file=sys.stderr)
                    await send_json(websocket, {"type": "STT_ERROR", "message": str(e)})

            elif msg_type == "ALL_RESET":
                await stop_pir(websocket)
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
