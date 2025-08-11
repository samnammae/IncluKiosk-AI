import os
import signal
import asyncio
import RPi.GPIO as GPIO
import websockets

PIR_PIN = int(os.environ.get("PIR_PIN", "18"))      # BCM 번호 (물리 12)
SLEEP_SEC = float(os.environ.get("PIR_SLEEP", "0.15"))
SERVER_URI = os.environ.get("SERVER_URI", "ws://localhost:8765")
ACTIVE_HIGH = os.environ.get("ACTIVE_HIGH", "1") == "1"  # 감지=HIGH가 일반적
WARMUP_SEC = float(os.environ.get("PIR_WARMUP", "2.0"))  # 전원 투입/시작 직후 안정 대기
CONFIRM_COUNT = int(os.environ.get("PIR_CONFIRM_COUNT", "2"))  # 연속 확인 횟수(노이즈 억제)
# PULL: "DOWN" | "UP" | "NONE"
PULL_MODE = os.environ.get("PIR_PULL", "DOWN").upper()

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
        return None  # 내부 풀업/풀다운 미적용

async def send_pir_off():
    try:
        async with websockets.connect(SERVER_URI) as ws:
            await ws.send("PIR_OFF")  # 필요 시 JSON으로 전송 가능
        print(f"[PIR] Sent PIR_OFF to {SERVER_URI}", flush=True)
    except Exception as e:
        print(f"[PIR] Failed to send PIR_OFF: {e}", flush=True)

async def main():
    print(
        f"[PIR] Worker start | PIN={PIR_PIN}, ACTIVE_HIGH={ACTIVE_HIGH}, "
        f"PULL={PULL_MODE}, WARMUP={WARMUP_SEC}s, CONFIRM_COUNT={CONFIRM_COUNT}, "
        f"URI={SERVER_URI}",
        flush=True
    )

    # ===== GPIO 초기화 =====
    GPIO.setmode(GPIO.BCM)
    pull = _resolve_pull_mode()
    if pull is None:
        GPIO.setup(PIR_PIN, GPIO.IN)
    else:
        GPIO.setup(PIR_PIN, GPIO.IN, pull_up_down=pull)

    try:
        # 워밍업 대기 (모듈 초기 HIGH/노이즈 억제)
        if WARMUP_SEC > 0:
            await asyncio.sleep(WARMUP_SEC)

        last = None
        confirm = 0

        while _running:
            val = GPIO.input(PIR_PIN)

            # 상태 변화 로그(도배 방지)
            if val != last:
                if (val == 1 and ACTIVE_HIGH) or (val == 0 and not ACTIVE_HIGH):
                    print("[PIR] Motion detected (edge/state change)", flush=True)
                else:
                    print("[PIR] No motion", flush=True)
                last = val

            # 감지 조건 계산
            motion_now = (val == 1) if ACTIVE_HIGH else (val == 0)
            if motion_now:
                confirm += 1
                # 아주 짧은 디바운스 간격
                await asyncio.sleep(0.05)
            else:
                confirm = 0

            # 연속 확인 횟수 충족 시 감지 확정
            if confirm >= max(1, CONFIRM_COUNT):
                print("[PIR] Motion confirmed → notify server and exit", flush=True)
                await send_pir_off()
                break

            await asyncio.sleep(SLEEP_SEC)

    finally:
        GPIO.cleanup()
        print("[PIR] Exit", flush=True)

if __name__ == "__main__":
    asyncio.run(main())