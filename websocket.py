import asyncio
import websockets
import subprocess
import json
import sys
from pathlib import Path

# ===== 워커 경로/파이썬 경로 =====
PYTHON = sys.executable  # 현재 파이썬 그대로 사용 (가상환경이면 그 환경)
BASE_DIR = Path(__file__).resolve().parent
PIR_WORKER = str(BASE_DIR / "pir_worker.py")

# ===== 전역: 실행 중인 프로세스들 관리 =====
workers = {
    "PIR": None,
    # 필요하면 MODE, CHAT 등 다른 워커도 키 추가
}

async def start_pir(websocket):
    if workers["PIR"] and (workers["PIR"].poll() is None):
        await websocket.send("이미 PIR 실행 중")
        return

    # 환경변수로 핀 지정 등 가능: env={"PIR_PIN":"7", ...}
    proc = subprocess.Popen([PYTHON, PIR_WORKER])
    workers["PIR"] = proc
    await websocket.send("PIR 워커 시작")

async def stop_pir(websocket):
    proc = workers.get("PIR")
    if not proc or (proc.poll() is not None):
        await websocket.send("PIR 이미 중지됨")
        workers["PIR"] = None
        return

    proc.terminate()  # SIGTERM 보냄 → 워커가 GPIO cleanup 후 종료
    try:
        # 블로킹 방지를 위해 짧게만 기다림
        await asyncio.get_event_loop().run_in_executor(None, proc.wait, 3)
    except Exception:
        # 여전히 살아있다면 강제 종료
        try:
            proc.kill()
        except Exception:
            pass
    finally:
        workers["PIR"] = None
        await websocket.send("PIR 워커 중지")

async def handle_client(websocket):
    print("클라이언트 연결됨")
    try:
        async for message in websocket:
            print(f"받은 메시지: {message}")

            try:
                data = json.loads(message)
                msg_type = data.get("type")
            except json.JSONDecodeError:
                msg_type = message

            print(f"파싱된 type: {msg_type}")

            if msg_type == "PIR_ON":
                await start_pir(websocket)

            elif msg_type == "PIR_OFF":
                await stop_pir(websocket)

            elif msg_type == "MODE_SELECT_ON":
                process = subprocess.Popen(["python", "test.py"])
                await websocket.send(f"서버에서 답장: {msg_type}")

            elif msg_type == "CHAT_ORDER_ON":
                process = subprocess.Popen(["python", "test.py"])
                await websocket.send(f"서버에서 답장: {msg_type}")

            elif msg_type == "NORMAL_ORDER_ON":
                process = subprocess.Popen(["python", "test.py"])
                await websocket.send(f"서버에서 답장: {msg_type}")

            elif msg_type == "EYE_ORDER_ON":
                process = subprocess.Popen(["python", "test.py"])
                await websocket.send(f"서버에서 답장: {msg_type}")

            elif msg_type == "ALL_RESET":
                await stop_pir(websocket)
                
            else: 
                await websocket.send(f"알 수 없는 type: {msg_type}")
            
    except websockets.exceptions.ConnectionClosed:
        print("클라이언트 연결 끊김")

async def main():
    print("서버 시작됨")
    print("Ctrl+C로 종료")
    
    # 서버 시작
    async with websockets.serve(handle_client, "0.0.0.0", 8765):
        await asyncio.Future()  # 서버 계속 실행

if __name__ == "__main__":
    asyncio.run(main())
