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
OBJECT_EYE = str(BASE_DIR / "object_eyecontrol.py")
GAZE_TRACKING = str(BASE_DIR / "gaze_tracking.py")

# ===== 전역: 실행 중인 프로세스/클라이언트 관리 =====
workers = {
    "PIR": None,
}

clients = set()  # 연결된 모든 클라이언트 소켓

# ===== 유틸리티 =====
async def broadcast(payload: str):
    """연결된 모든 클라이언트에 메시지 전파 (에러난 소켓은 제거)."""
    dead = []
    for ws in list(clients):
        try:
            await ws.send(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)

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
        # 블로킹 wait를 쓰레드로 넘겨서 3초 대기
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, proc.wait, 3)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    finally:
        workers["PIR"] = None
        await websocket.send("PIR 워커 중지")

async def clear_pir_state_and_notify_frontend():
    workers["PIR"] = None
    await broadcast("PIR_OFF")

async def handle_client(websocket):
    print("클라이언트 연결됨")
    clients.add(websocket)
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
                # CASE 1: 프론트 → 서버 → PIR 워커 시작
                await start_pir(websocket)

            elif msg_type == "PIR_OFF":
                # CASE 2-1/2-2
                #  - 워커(source=worker)가 보낸 PIR_OFF: 프론트에 그대로 전달 + 내부상태정리
                #  - 프론트가 보낸 PIR_OFF: PIR 워커만 중지
                if isinstance(data, dict) and data.get("source") == "worker":
                    await clear_pir_state_and_notify_frontend()
                else:
                    await stop_pir(websocket)

            elif msg_type == "MODE_SELECT_ON":
                subprocess.Popen([PYTHON, GAZE_TRACKING])
                await websocket.send("MODE_SELECT_ON_ACK")

            elif msg_type == "CHAT_ORDER_ON":
                # CASE 4-1/4-2
                subprocess.Popen([PYTHON, "-c", "print('CHAT_ORDER stub')"])
                await websocket.send("CHAT_ORDER_ON_ACK")

            elif msg_type == "NORMAL_ORDER_ON":
                # CASE 4-3
                subprocess.Popen([PYTHON, "-c", "print('NORMAL_ORDER stub')"])
                await websocket.send("NORMAL_ORDER_ON_ACK")

            elif msg_type == "EYE_ORDER_ON":
                # CASE 4-4
                subprocess.Popen([PYTHON, "-c", "print('EYE_ORDER stub')"])
                await websocket.send("EYE_ORDER_ON_ACK")

            elif msg_type == "ALL_RESET":
                # CASE 5: 주문 완료 후 전체 리셋
                await stop_pir(websocket)
                await websocket.send("ALL_RESET_ACK")

            else:
                await websocket.send(f"알 수 없는 type: {msg_type}")

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