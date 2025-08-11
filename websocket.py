import asyncio
import websockets
import subprocess
import json

async def handle_client(websocket):
    print("클라이언트 연결됨")
    
    try:
        async for message in websocket:
            print(f"받은 메시지: {message}")
            
            if message == "PIR_ON":
                print("running test.py ...")
                process = subprocess.Popen(["python", "test.py"])
                await websocket.send(f"서버에서 답장: run test.py")
                
            else: 
                # 받은 메시지를 그대로 다시 보내기
                await websocket.send(f"서버에서 답장: {message}")
            
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
