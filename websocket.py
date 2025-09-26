import asyncio
import websockets
import subprocess
import json
import sys
from pathlib import Path
import os
from functools import partial

from tts_stt import tts_play, stt_once

# ===== 워커 경로/파이썬 경로 =====
PYTHON = sys.executable
BASE_DIR = Path(__file__).resolve().parent
PIR_WORKER = str(BASE_DIR / "pir_worker.py")

# ===== 전역: 실행 중인 프로세스/클라이언트 관리 =====
workers = {
    "PIR": None,
}

clients = set()  # 연결된 모든 클라이언트 소켓


# ===== JSON 유틸 =====
async def send_json(ws, payload: dict):
    try:
        await ws.send(json.dumps(payload, ensure_ascii=False))
    except Exception:
        # 송신 실패 시 클라이언트 제거
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


# ===== PIR 제어 =====
async def start_pir(websocket):
    if workers["PIR"] and (workers["PIR"].poll() is None):
        await send_json(websocket, {"type": "PIR_ON_ACK", "status": "already_running"})
        return

    # 필요 시 환경변수 전달 가능: env={**os.environ, "PIR_PIN":"18"}
    proc = subprocess.Popen([PYTHON, PIR_WORKER])
    workers["PIR"] = proc
    await send_json(websocket, {"type": "PIR_ON_ACK", "status": "started"})

async def stop_pir(websocket):
    """외부에서 PIR을 끄라고 했을 때: 실행 중이면 정상 종료 시도"""
    proc = workers.get("PIR")
    if not proc or (proc.poll() is not None):
        workers["PIR"] = None
        await send_json(websocket, {"type": "PIR_OFF_ACK", "status": "already_stopped"})
        return

    proc.terminate()  # SIGTERM → 워커가 cleanup 후 종료
    try:
        # 블로킹 wait를 스레드풀로 넘겨서 3초 대기
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, proc.wait, 3)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    finally:
        workers["PIR"] = None
        await send_json(websocket, {"type": "PIR_OFF_ACK", "status": "stopped"})

async def clear_pir_state_and_notify_frontend():
    workers["PIR"] = None
    await broadcast_json({"type": "PIR_OFF"})


# ===== 메인 핸들러 =====
DEFAULT_CHAT_GUIDE = "안녕하세요. 음성으로 주문을 도와드릴게요. 무엇을 드시고 싶으신가요?"

async def handle_client(websocket):
    print("클라이언트 연결됨")
    clients.add(websocket)
    try:
        async for raw in websocket:
            print(f"받은 메시지: {raw}")

            # ---- 입력 JSON 파싱 ----
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

            # ===== 잠금화면 / PIR =====
            if msg_type == "PIR_ON":
                # CASE 1
                await start_pir(websocket)

            elif msg_type == "PIR_OFF":
                # CASE 2-2: 프론트가 직접 중지
                await stop_pir(websocket)

            elif msg_type == "PIR_DETECTED":
                # CASE 2-1: 라즈베리(워커) → 프론트: PIR 감지됨
                await broadcast_json({"type": "PIR_DETECTED"})

            # ===== 조정/보정 =====
            elif msg_type == "EYE_CALIB_ON":
                # CASE 2-3: (라즈베리 → 프론트) 눈보정 화면으로
                await broadcast_json({"type": "EYE_CALIB_ON"})

            elif msg_type == "MODE_SELECT_ON":
                # CASE 2-4, CASE 3: 모드선택 진입
                # (아이트래킹/주먹감지 준비는 별도 워커로 확장 가능)
                subprocess.Popen([PYTHON, "-c", "print('MODE_SELECT stub')"])
                await send_json(websocket, {"type": "MODE_SELECT_ON_ACK"})

            # ===== 모드 선택 → 대화/일반/눈 =====
            elif msg_type == "CHAT_ORDER_ON":
                # CASE 4-1/4-2: 대화주문 진입
                # 1) 감지 코드 정지 (필요 시 워커 종료)
                subprocess.Popen([PYTHON, "-c", "print('STOP eye/fist workers stub')"])
                await send_json(websocket, {"type": "CHAT_ORDER_ON_ACK"})

                # 2) 안내 TTS 자동 재생 (message가 없으면 기본 멘트)
                guide_text = msg_text if (isinstance(msg_text, str) and msg_text.strip()) else DEFAULT_CHAT_GUIDE
                loop = asyncio.get_running_loop()
                try:
                    await loop.run_in_executor(None, partial(tts_play, guide_text, "ko-KR", None))
                except Exception as e:
                    await send_json(websocket, {"type": "TTS_ERROR", "message": f"Guide TTS failed: {e}"})
                else:
                    # 재생 종료 통지 → 프론트는 여기서 STT_ON 시작
                    await send_json(websocket, {"type": "TTS_OFF"})

            elif msg_type == "NORMAL_ORDER_ON":
                # CASE 4-3
                subprocess.Popen([PYTHON, "-c", "print('NORMAL_ORDER stub')"])
                await send_json(websocket, {"type": "NORMAL_ORDER_ON_ACK"})

            elif msg_type == "EYE_ORDER_ON":
                # CASE 4-4
                subprocess.Popen([PYTHON, "-c", "print('EYE_ORDER stub')"])
                await send_json(websocket, {"type": "EYE_ORDER_ON_ACK"})

            # ===== 대화주문 중 TTS/STT =====
            elif msg_type == "TTS_ON":
                # 대화 중간 응답 읽기 (명세상 주문 시작이 아님)
                # message(텍스트) 외 lang/voice/rate/pitch/enc 확장 필드도 허용
                text  = data.get("message") or data.get("text") or ""
                lang  = data.get("lang", "ko-KR")
                voice = data.get("voice", None)  # 예: "ko-KR-Wavenet-B"
                rate  = float(data.get("speakingRate", 1.0))
                pitch = float(data.get("pitch", 0.0))
                enc   = data.get("audioEncoding", "MP3")

                loop = asyncio.get_running_loop()
                try:
                    await loop.run_in_executor(
                        None, partial(tts_play, text, lang, voice, rate, pitch, None, enc)
                    )
                except Exception as e:
                    await send_json(websocket, {"type": "TTS_ERROR", "message": str(e)})
                else:
                    await send_json(websocket, {"type": "TTS_OFF"})

            elif msg_type == "STT_ON":
                # 단발 녹음 → 인식 → STT_OFF로 결과 전달
                duration = int(data.get("duration", 5))
                sample_rate = int(data.get("sampleRate", 16000))
                language_code = data.get("languageCode", "ko-KR")
                device_idx = data.get("deviceIndex", None)

                loop = asyncio.get_running_loop()
                try:
                    transcript = await loop.run_in_executor(
                        None,
                        partial(stt_once, duration=duration, sample_rate=sample_rate,
                                language_code=language_code, device=device_idx)
                    )
                    await send_json(websocket, {"type": "STT_OFF", "message": transcript})
                except Exception as e:
                    await send_json(websocket, {"type": "STT_ERROR", "message": str(e)})

            # ===== 전체 리셋 =====
            elif msg_type == "ALL_RESET":
                # CASE 5: 주문 완료 후 전체 리셋
                await stop_pir(websocket)
                # 이 외에 돌고 있는 워커들 정리 필요 시 여기에 추가
                await send_json(websocket, {"type": "ALL_RESET_ACK"})

            else:
                await send_json(websocket, {"type": "ERROR", "message": f"Unknown type: {msg_type}"})

    except websockets.exceptions.ConnectionClosed:
        print("클라이언트 연결 끊김")
    finally:
        clients.discard(websocket)


async def main():
    print("서버 시작됨 (0.0.0.0:8765)")
    print("Ctrl+C로 종료")
    async with websockets.serve(handle_client, "0.0.0.0", 8765):
        await asyncio.Future()  # 서버 계속 실행

if __name__ == "__main__":
    asyncio.run(main())
