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
SERVER_URI = os.environ.get("SERVER_URI", "ws://localhost:8765")
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

async def send_pir_detected():
    """서버에 'worker' 소스로 PIR_DETECTED 전송 (서버는 프론트로 브로드캐스트)."""
    try:
        async with websockets.connect(SERVER_URI) as ws:
            await ws.send(json.dumps({"type": "PIR_DETECTED", "source": "worker"}))
        print(f"[PIR] Sent PIR_DETECTED(source=worker) to {SERVER_URI}", flush=True)
    except Exception as e:
        print(f"[PIR] Failed to send PIR_DETECTED: {e}", flush=True)

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

            # 확정 시 서버 통지 후 워커 종료 (서버는 계속 RUN)
            if confirm >= max(1, CONFIRM_COUNT):
                print("[PIR] Motion confirmed → notify server and exit", flush=True)
                await send_pir_detected()
                break

            await asyncio.sleep(SLEEP_SEC)

    except Exception as e:
        print("[PIR] Worker exception:", e, flush=True)
        traceback.print_exc(file=sys.stdout)
    finally:
        GPIO.cleanup()
        print("[PIR] Exit", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
