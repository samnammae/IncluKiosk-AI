#!/usr/bin/env python3
import sys
import time
import signal
import select
import termios
import tty
import RPi.GPIO as GPIO

# ===== 기본 설정값 (환경변수 없이 코드 내 직접 지정) =====
PIR_PIN = 18           # BCM 번호 (물리 12)
ACTIVE_HIGH = True     # 센서 감지 시 HIGH 신호면 True, LOW면 False
PULL_MODE = "DOWN"     # "DOWN", "UP", 또는 "NONE"
SLEEP_SEC = 0.15       # 감지 주기 (초)
WARMUP_SEC = 2.0       # 센서 안정화 대기 시간 (초)
CONFIRM_COUNT = 2      # 연속 감지 횟수 (디바운스)

_running = True


# ===== 종료 핸들러 =====
def _handle_stop(signum, frame):
    global _running
    _running = False

signal.signal(signal.SIGINT, _handle_stop)
signal.signal(signal.SIGTERM, _handle_stop)


# ===== 풀업/풀다운 설정 =====
def _resolve_pull_mode():
    if PULL_MODE == "DOWN":
        return GPIO.PUD_DOWN
    elif PULL_MODE == "UP":
        return GPIO.PUD_UP
    else:
        return None


# ===== 비차단 키입력 클래스 =====
class NonBlockingStdin:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def getch(self):
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            ch = sys.stdin.read(1)
            return ch
        return None


# ===== 메인 루프 =====
def main():
    print(
        f"[PIR_TEST] Start | PIN={PIR_PIN}, ACTIVE_HIGH={ACTIVE_HIGH}, "
        f"PULL={PULL_MODE}, WARMUP={WARMUP_SEC}s, CONFIRM_COUNT={CONFIRM_COUNT}, "
        f"SLEEP={SLEEP_SEC}s",
        flush=True
    )

    # GPIO 초기화
    GPIO.setmode(GPIO.BCM)
    pull = _resolve_pull_mode()
    if pull is None:
        GPIO.setup(PIR_PIN, GPIO.IN)
    else:
        GPIO.setup(PIR_PIN, GPIO.IN, pull_up_down=pull)

    # 워밍업
    if WARMUP_SEC > 0:
        print(f"[PIR_TEST] Warming up for {WARMUP_SEC:.1f}s...")
        time.sleep(WARMUP_SEC)

    last_val = None
    confirm = 0

    try:
        with NonBlockingStdin() as nbs:
            print("[PIR_TEST] Running... (press 'q' or ESC to quit)")
            while _running:
                val = GPIO.input(PIR_PIN)
                motion_now = (val == 1) if ACTIVE_HIGH else (val == 0)

                # 디바운스 처리
                if motion_now:
                    confirm = min(confirm + 1, CONFIRM_COUNT)
                else:
                    confirm = 0

                if confirm >= CONFIRM_COUNT:
                    msg = "[PIR_TEST] Motion detected"
                else:
                    msg = "[PIR_TEST] No motion"

                # 상태 변경 시 엣지 로그
                if val != last_val:
                    edge = "HIGH" if val == 1 else "LOW"
                    print(f"[PIR_TEST] Edge change → {edge}", flush=True)
                    last_val = val

                print(msg, flush=True)

                # 종료 키 체크
                ch = nbs.getch()
                if ch is not None:
                    if ch.lower() == "q" or ord(ch) == 27:  # ESC
                        print("[PIR_TEST] Quit key pressed")
                        break

                time.sleep(SLEEP_SEC)

    except Exception as e:
        print(f"[PIR_TEST] Exception: {e}")
    finally:
        GPIO.cleanup()
        print("[PIR_TEST] Exit")


if __name__ == "__main__":
    main()
