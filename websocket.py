import asyncio
import websockets
import subprocess
import json
import sys
from pathlib import Path

# ===== 워커 경로/파이썬 경로 =====
PYTHON = sys.executable
BASE_DIR = Path(__file__).resolve().parent
PIR_WORKER = str(BASE_DIR / "pir_worker.py")

# ===== 전역: 실행 중인 프로세스들 관리 =====
workers = {
    "PIR": None,
}

async def start_pir(websocket):
    if workers["PIR"] and (workers["PIR"].poll() is None):
        await websocket.send("이미 PIR 실행 중")
        return

    # 필요 시 환경변수 전달 가능: env={**os.environ, "PIR_PIN":"18"}
    proc = subprocess.Popen([PYTHON, PIR_WORKER])
    workers["PIR"] = proc
    await websocket.send("PIR 워커 시작")

async def stop_pir(websocket):
    """외부에서 PIR을 끄라고 했을 때: 실행 중이면 정상 종료 시도"""
    proc = workers.get("PIR")
    if not proc or (proc.poll() is not None):
        workers["PIR"] = None
        await websocket.send("PIR 이미 중지됨")
        return

    proc.terminate()  # SIGTERM → 워커가 cleanup 후 종료
    try:
        await asyncio.get_event_loop().run_in_executor(None, proc.wait, 3)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    finally:
        workers["PIR"] = None
        await websocket.send("PIR 워커 중지")

async def clear_pir_state(websocket):
    """워커가 스스로 종료되며 보내온 이벤트 처리: 시그널 없이 상태만 OFF로."""
    workers["PIR"] = None
    await websocket.send("PIR 워커 종료 이벤트 수신 → 상태: OFF")

async def handle_client(websocket):
    print("클라이언트 연결됨")
    try:
        async for message in websocket:
            print(f"받은 메시지: {message}")

            data = None
            try:
                data = json.loads(message)
                msg_type = data.get("type")
            except (json.JSONDecodeError, AttributeError):
                msg_type = message

            print(f"파싱된 type: {msg_type}")

            if msg_type == "PIR_ON":
                await start_pir(websocket)

            elif msg_type == "PIR_OFF":
                # 워커가 스스로 보낸 PIR_OFF면 상태만 정리 (연결 유지)
                if isinstance(data, dict) and data.get("source") == "worker":
                    await clear_pir_state(websocket)
                else:
                    await stop_pir(websocket)

            elif msg_type == "MODE_SELECT_ON":
                subprocess.Popen([PYTHON, "-c", "print('MODE_SELECT stub')"])
                await websocket.send(f"서버에서 답장: {msg_type}")

            elif msg_type == "CHAT_ORDER_ON":
                subprocess.Popen([PYTHON, "-c", "print('CHAT_ORDER stub')"])
                await websocket.send(f"서버에서 답장: {msg_type}")

            elif msg_type == "NORMAL_ORDER_ON":
                subprocess.Popen([PYTHON, "-c", "print('NORMAL_ORDER stub')"])
                await websocket.send(f"서버에서 답장: {msg_type}")

            elif msg_type == "EYE_ORDER_ON":
                subprocess.Popen([PYTHON, "-c", "print('EYE_ORDER stub')"])
                await websocket.send(f"서버에서 답장: {msg_type}")

            elif msg_type == "ALL_RESET":
                await stop_pir(websocket)

            else:
                await websocket.send(f"알 수 없는 type: {msg_type}")

    except websockets.exceptions.ConnectionClosed:
        print("클라이언트 연결 끊김")

async def main():
    print("서버 시작됨 (0.0.0.0:8765)")
    print("Ctrl+C로 종료")
    async with websockets.serve(handle_client, "0.0.0.0", 8765):
        await asyncio.Future()  # 서버 계속 실행

if __name__ == "__main__":
    asyncio.run(main())