import asyncio
import websockets
import subprocess
import json
import sys
import signal
import logging
from pathlib import Path

# ===== 로깅 설정 =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== 워커 경로/파이썬 경로 =====
PYTHON = sys.executable
BASE_DIR = Path(__file__).resolve().parent
PIR_WORKER = str(BASE_DIR / "pir_worker.py")
EYE_WORKER = str(BASE_DIR / "object_eyecontrol.py")

# ===== 설정 =====
PROCESS_TIMEOUT = 5  # 프로세스 종료 대기 시간 (초)

# ===== 전역: 실행 중인 프로세스/클라이언트 관리 =====
workers = {"PIR": None, "EYE": None}
clients = set()  # 연결된 모든 클라이언트 소켓
_shutdown_event = asyncio.Event()

def _handle_signal(signum, frame):
    """시그널 핸들러"""
    logger.info(f"Signal {signum} received, shutting down...")
    _shutdown_event.set()

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)

# ===== 유틸리티 =====
async def broadcast(payload: str):
    """모든 연결된 클라이언트에게 메시지 브로드캐스트"""
    if not clients:
        logger.warning("브로드캐스트할 클라이언트가 없음")
        return
    
    dead_clients = []
    for ws in list(clients):
        try:
            await ws.send(payload)
            logger.info(f"브로드캐스트 성공: {payload}")
        except websockets.exceptions.ConnectionClosed:
            logger.warning("클라이언트 연결 끊김 - 제거 예정")
            dead_clients.append(ws)
        except Exception as e:
            logger.error(f"브로드캐스트 실패: {e}")
            dead_clients.append(ws)
    
    # 죽은 클라이언트 제거
    for ws in dead_clients:
        clients.discard(ws)

def is_process_running(proc):
    """프로세스가 실행 중인지 확인"""
    return proc is not None and proc.poll() is None

async def terminate_process_safely(proc, process_name="Process"):
    """프로세스를 안전하게 종료"""
    if not proc or proc.poll() is not None:
        return True
    
    try:
        logger.info(f"{process_name} 종료 시작...")
        proc.terminate()
        
        # 비동기로 프로세스 종료 대기
        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, proc.wait), 
                timeout=PROCESS_TIMEOUT
            )
            logger.info(f"{process_name} 정상 종료됨")
            return True
        except asyncio.TimeoutError:
            logger.warning(f"{process_name} 강제 종료 시도...")
            proc.kill()
            await loop.run_in_executor(None, proc.wait)
            logger.info(f"{process_name} 강제 종료됨")
            return True
            
    except Exception as e:
        logger.error(f"{process_name} 종료 중 오류: {e}")
        return False

# ----- PIR -----
async def start_pir(websocket):
    """PIR 워커 시작"""
    if is_process_running(workers["PIR"]):
        await websocket.send("이미 PIR 실행 중")
        logger.warning("PIR 워커 이미 실행 중")
        return
    
    try:
        proc = subprocess.Popen(
            [PYTHON, PIR_WORKER],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        workers["PIR"] = proc
        await websocket.send("PIR 워커 시작")
        logger.info(f"PIR 워커 시작: PID {proc.pid}")
    except Exception as e:
        error_msg = f"PIR 워커 시작 실패: {e}"
        await websocket.send(error_msg)
        logger.error(error_msg)

async def stop_pir(websocket=None):
    """PIR 워커 정지"""
    proc = workers.get("PIR")
    if not is_process_running(proc):
        workers["PIR"] = None
        message = "PIR 이미 중지됨"
        if websocket:
            await websocket.send(message)
        logger.info(message)
        return
    
    success = await terminate_process_safely(proc, "PIR 워커")
    workers["PIR"] = None
    
    message = "PIR 워커 중지" if success else "PIR 워커 중지 완료 (일부 오류 발생)"
    if websocket:
        await websocket.send(message)
    logger.info(message)

async def clear_pir_state_and_notify_frontend():
    """PIR 상태 정리 및 프론트엔드 알림"""
    # PIR 워커가 감지 후 자동 종료했으므로 상태만 정리
    workers["PIR"] = None
    await broadcast("PIR_OFF")
    logger.info("PIR 상태 정리 및 프론트엔드 알림 완료")

# ----- EYE (아이트래킹/주먹감지) -----
async def start_eye(mode: str, websocket):
    """
    아이트래킹 워커 시작
    mode: 'mode_select' -> 눈+주먹,  'eye_only' -> 눈만
    """
    # 이미 실행 중이면 기존 것을 먼저 종료
    if is_process_running(workers["EYE"]):
        logger.info(f"EYE 워커 재시작: {mode} 모드로 변경")
        await stop_eye()
        # 프로세스가 완전히 종료될 때까지 잠시 대기
        await asyncio.sleep(0.5)
    
    try:
        args = [PYTHON, EYE_WORKER, "--mode", mode, "--server", "ws://localhost:8765"]
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        workers["EYE"] = proc
        message = f"EYE 워커 시작({mode})"
        await websocket.send(message)
        logger.info(f"{message}: PID {proc.pid}")
    except Exception as e:
        error_msg = f"EYE 워커 시작 실패: {e}"
        await websocket.send(error_msg)
        logger.error(error_msg)

async def stop_eye(websocket=None):
    """아이트래킹 워커 정지"""
    proc = workers.get("EYE")
    if not is_process_running(proc):
        workers["EYE"] = None
        message = "EYE 이미 중지됨"
        if websocket:
            await websocket.send(message)
        logger.info(message)
        return
    
    success = await terminate_process_safely(proc, "EYE 워커")
    workers["EYE"] = None
    
    message = "EYE 워커 중지" if success else "EYE 워커 중지 완료 (일부 오류 발생)"
    if websocket:
        await websocket.send(message)
    logger.info(message)

# ----- 핸들러 -----
async def handle_client(websocket, path=None):
    """클라이언트 연결 핸들러"""
    client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
    logger.info(f"클라이언트 연결됨: {client_info}")
    clients.add(websocket)
    
    try:
        async for message in websocket:
            logger.info(f"받은 메시지: {message}")

            # JSON/문자열 파싱
            data = None
            try:
                data = json.loads(message)
                msg_type = data.get("type")
            except (json.JSONDecodeError, AttributeError):
                msg_type = message.strip()

            logger.info(f"파싱된 type: {msg_type}")

            # 메시지 타입별 처리
            if msg_type == "PIR_ON":
                # CASE 1: PIR 센서 시작
                await start_pir(websocket)

            elif msg_type == "PIR_OFF":
                # CASE 2-1/2-2: PIR 센서 정지
                if isinstance(data, dict) and data.get("source") == "worker":
                    # PIR 워커가 감지→종료를 알림: 상태정리 + 프론트로 브로드캐스트
                    await clear_pir_state_and_notify_frontend()
                else:
                    # 프론트가 직접 PIR 중지 요청
                    await stop_pir(websocket)

            elif msg_type == "MODE_SELECT_ON":
                # CASE 3: 아이트래킹+주먹감지 시작
                await start_eye("mode_select", websocket)
                await websocket.send("MODE_SELECT_ON_ACK")

            elif msg_type == "CHAT_ORDER_ON":
                # CASE 4-1/4-2: 대화주문 진입
                if isinstance(data, dict) and data.get("source") == "worker":
                    # 워커(주먹감지)에서 온 경우: 프론트에 브로드캐스트
                    await broadcast("CHAT_ORDER_ON")
                # 아이트래킹/주먹감지 정지
                await stop_eye()
                await websocket.send("CHAT_ORDER_ON_ACK")

            elif msg_type == "NORMAL_ORDER_ON":
                # CASE 4-3: 일반주문
                await stop_eye()
                await websocket.send("NORMAL_ORDER_ON_ACK")

            elif msg_type == "EYE_ORDER_ON":
                # CASE 4-4: 아이트래킹 주문 (커서 시각화, 주먹감지 중지)
                await start_eye("eye_only", websocket)
                await websocket.send("EYE_ORDER_ON_ACK")

            elif msg_type == "ALL_RESET":
                # CASE 5: 전체 리셋
                await stop_pir()
                await stop_eye()
                await websocket.send("ALL_RESET_ACK")
                logger.info("전체 리셋 완료")

            else:
                error_msg = f"알 수 없는 type: {msg_type}"
                await websocket.send(error_msg)
                logger.warning(error_msg)

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"클라이언트 연결 끊김: {client_info}")
    except Exception as e:
        logger.error(f"클라이언트 처리 중 오류: {e}")
    finally:
        clients.discard(websocket)
        logger.info(f"클라이언트 정리됨: {client_info}")

async def cleanup_on_shutdown():
    """종료 시 정리 작업"""
    logger.info("서버 종료 중... 모든 워커 정리")
    await stop_pir()
    await stop_eye()
    logger.info("정리 작업 완료")

async def main():
    """메인 서버 실행"""
    logger.info("WebSocket 서버 시작 (0.0.0.0:8765)")
    logger.info("Ctrl+C로 종료")
    
    try:
        # WebSocket 서버 시작
        server = await websockets.serve(handle_client, "0.0.0.0", 8765)
        
        # 종료 시그널 대기
        await _shutdown_event.wait()
        
    except Exception as e:
        logger.error(f"서버 실행 중 오류: {e}")
    finally:
        await cleanup_on_shutdown()
        logger.info("서버 종료됨")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("사용자에 의해 종료됨")
    except Exception as e:
        logger.error(f"실행 중 오류: {e}")