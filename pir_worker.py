import RPi.GPIO as GPIO
import time
import signal
import sys
import os

# ===== 설정 =====
PIR_PIN = int(os.environ.get("PIR_PIN", "7"))  # 필요하면 환경변수로 바꿔 쓰기
SLEEP_SEC = float(os.environ.get("PIR_SLEEP", "0.2"))

running = True

def handle_stop(signum, frame):
    global running
    running = False

signal.signal(signal.SIGTERM, handle_stop)
signal.signal(signal.SIGINT, handle_stop)

def main():
    GPIO.setmode(GPIO.BCM)
    # 센서에 따라 풀업/풀다운이 다를 수 있음.
    # 일반적인 HC-SR501류 PIR은 출력이 High/Low로 드라이브되므로 내부 풀업/다운 없이도 동작.
    GPIO.setup(PIR_PIN, GPIO.IN)

    print(f"[PIR] 시작 (GPIO {PIR_PIN})", flush=True)

    try:
        last = None
        while running:
            val = GPIO.input(PIR_PIN)  # HIGH=감지, LOW=없음
            if val != last:
                if val:
                    print("[PIR] 움직임 감지(HIGH)", flush=True)
                else:
                    print("[PIR] 움직임 없음(LOW)", flush=True)
                last = val
            time.sleep(SLEEP_SEC)
    finally:
        GPIO.cleanup()
        print("[PIR] 종료", flush=True)

if __name__ == "__main__":
    # 루트 권한 필요할 수 있음: sudo로 실행 권장
    main()
