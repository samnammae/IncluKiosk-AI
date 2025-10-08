import os
import signal
import asyncio
import json
import RPi.GPIO as GPIO
import websockets
import sys
import traceback

PIR_PIN = int(os.environ.get("PIR_PIN", "18"))      # BCM 번호 (물리 12)
SLEEP_SEC = float(os.environ.get("PIR_SLEEP", "0.15"))
SERVER_URI = os.environ.get("SERVER_URI", "ws://localhost:8766")
ACTIVE_HIGH = os.environ.get("ACTIVE_HIGH", "1") == "1"  # 감지=HIGH
WARMUP_SEC = float(os.environ.get("PIR_WARMUP", "2.0"))  # 안정 대기
CONFIRM_COUNT = int(os.environ.get("PIR_CONFIRM_COUNT", "2"))  # 연속 확인
PULL_MODE = os.environ.get("PIR_PULL", "DOWN").upper()   # DOWN|UP|NONE

_running = True

def _handle_stop(signum, frame):
    global _running
    _running = False

signal.signal(signal.SIGTERM, _handle_stop)
signal.signal(signal.SIGINT, _handle_stop)

def _resolve_pull_mode():
    if PULL_MODE == "DOWN":
        return GPIO.PUD_DOWN
    elif PULL_MODE == "UP":
        return GPIO.PUD_UP
    else:
        return None
    
async def main():
    print(
        f"[PIR] Worker start | PIN={PIR_PIN}, ACTIVE_HIGH={ACTIVE_HIGH}, "
        f"PULL={PULL_MODE}, WARMUP={WARMUP_SEC}s, CONFIRM_COUNT={CONFIRM_COUNT}, "
        f"URI={SERVER_URI}",
        flush=True
    )

    try:
        # ===== GPIO 초기화 =====
        GPIO.setmode(GPIO.BCM)
        pull = _resolve_pull_mode()
        if pull is None:
            GPIO.setup(PIR_PIN, GPIO.IN)
        else:
            GPIO.setup(PIR_PIN, GPIO.IN, pull_up_down=pull)

        # 워밍업 대기
        if WARMUP_SEC > 0:
            await asyncio.sleep(WARMUP_SEC)

        # ===== 내부 허브(8766)에 단일 연결 유지 =====
        async with websockets.connect(SERVER_URI) as ws:

            # 허브에서 오는 명령(PIR_OFF) 수신 태스크
            async def recv_cmd():
                global _running
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                        if msg.get("type") == "PIR_OFF":
                            print("[PIR] PIR_OFF received → stopping worker", flush=True)
                            _running = False
                    except Exception:
                        pass

            asyncio.create_task(recv_cmd())  # ← 함수 바깥에서 태스크 생성

            last = None
            confirm = 0

            while _running:
                val = GPIO.input(PIR_PIN)

                if val != last:
                    if (val == 1 and ACTIVE_HIGH) or (val == 0 and not ACTIVE_HIGH):
                        print("[PIR] Motion detected (edge/state change)", flush=True)
                    else:
                        print("[PIR] No motion", flush=True)
                    last = val

                # 감지 판단 + 디바운스
                motion_now = (val == 1) if ACTIVE_HIGH else (val == 0)
                if motion_now:
                    confirm += 1
                    await asyncio.sleep(0.05)
                else:
                    confirm = 0

                # ===== 확정 감지 시 서버로 통지 (종료 금지) =====
                if confirm >= max(1, CONFIRM_COUNT):
                    await ws.send(json.dumps({
                        "type": "PIR_DETECTED",
                        "source": "worker"
                    }))
                    print("[PIR] Motion confirmed → sent PIR_DETECTED", flush=True)
                    confirm = 0  # 계속 대기 (break/exit 금지)

                await asyncio.sleep(SLEEP_SEC)

    except Exception as e:
        print("[PIR] Worker exception:", e, flush=True)
        traceback.print_exc(file=sys.stdout)
    finally:
        GPIO.cleanup()
        print("[PIR] Exit", flush=True)

if __name__ == "__main__":
    asyncio.run(main())