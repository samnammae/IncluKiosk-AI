"""
리니어 엑추에이터 구동 및 안전 제어 통합 모듈
- 기본 제어 (moveUp / moveDown)
- 현재 높이 추적 (CUR_HEIGHT_STEP)
- 안전 이동 및 한계 검증
- 종료 시 원점 복귀
"""

import RPi.GPIO as GPIO
from time import sleep
from pathlib import Path

# === GPIO 핀 설정 ===
PUL = 17   # Pulse(전력 공급)
DIR = 27   # Direction
ENA = 22   # Enable(모터 활성화)

GPIO.setmode(GPIO.BCM)
GPIO.setup(PUL, GPIO.OUT)
GPIO.setup(DIR, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)

# === 기본 설정 ===
STEP_DELAY = 0.0005     # 한 스텝당 지연 (속도 제어)
DEFAULT_STEP = 10       # 기본 이동 스텝 수 (미세 조정용)
# MAX_HEIGHT_STEP = 5000  # 액추에이터 최고 높이 (임시, 실제 측정 필요)
CUR_HEIGHT_STEP = 0     # 액추에이터 현재 높이 (전역 상태)
HEIGHT_FILE = Path(__file__).with_name("current_height.txt") # 현재 높이를 담은 파일
HEIGHT_MIN = 0          # 액추에이터 최소 높이
HEIGHT_MAX = 28000      # 액추에이터 최고 높이

def _read_height_from_file() -> int:
    """
    current_height.txt에서 높이를 읽어온다.
    - 정수 아니거나 범위를 벗어나면 None을 반환(안전 차단)
    """
    try:
        v = int(HEIGHT_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        return None
    if not (HEIGHT_MIN <= v <= HEIGHT_MAX):
        return None
    return v

def _write_height_to_file(value: int):
    """현재 높이 파일에 기록"""
    try:
        HEIGHT_FILE.write_text(str(value), encoding="utf-8")
    except Exception as e:
        print(f"[ACTUATOR] ⚠️ 높이 파일 기록 실패: {e}")

# === 모터 준비 ===
def init_motor():
    """모터를 활성화하여 동작 준비"""
    GPIO.output(ENA, GPIO.HIGH)
    print("[ACTUATOR] ✅ Ready: Motor Enabled")

# === 모터 정리 ===
def cleanup_motor():
    """GPIO 및 모터 비활성화 (높이조절 프로세스 종료 시)"""
    GPIO.output(ENA, GPIO.LOW)
    GPIO.cleanup()
    print("[ACTUATOR] 🧹 GPIO cleaned up and motor disabled")

# === 한계 확인 ===
def exceed_max_height() -> bool:
    """
    파일의 현재 높이를 읽어 상한 초과 여부를 판단.
    - 파일 읽기 실패/이상치면 True(이동 차단)
    - 정상 읽기 시 전역 CUR_HEIGHT_STEP과 동기화
    """
    global CUR_HEIGHT_STEP
    v = _read_height_from_file()
    if v is None:
        print("[ACTUATOR] 🚫 current_height.txt 오류(없음/비정상/범위외). 이동 차단")
        return True
    CUR_HEIGHT_STEP = v
    return CUR_HEIGHT_STEP >= HEIGHT_MAX  # 0~28000 범위에서 28000이면 더 이상 UP 불가

def exceed_min_height() -> bool:
    """
    파일의 현재 높이를 읽어 하한 초과(=바닥) 여부를 판단.
    - 파일 읽기 실패/이상치면 True(이동 차단)
    - 정상 읽기 시 전역 CUR_HEIGHT_STEP과 동기화
    """
    global CUR_HEIGHT_STEP
    v = _read_height_from_file()
    if v is None:
        print("[ACTUATOR] 🚫 current_height.txt 오류(없음/비정상/범위외). 이동 차단")
        return True
    CUR_HEIGHT_STEP = v
    return CUR_HEIGHT_STEP <= HEIGHT_MIN  # 0이면 더 이상 DOWN 불가

# === 위로 이동 ===
def moveUp(steps: int = DEFAULT_STEP):
    """
    액추에이터를 위로 (확장)
    - 얼굴이 화면 중앙보다 아래에 있을 때 호출
    - 자동으로 최대 높이 검증 수행
    """
    global CUR_HEIGHT_STEP

    if exceed_max_height():
        print("[ACTUATOR] 🚫 최대 높이 도달, 이동 중단")
        return

    GPIO.output(DIR, GPIO.LOW)   # 위쪽 방향 (edited: HIGH > LOW)
    GPIO.output(ENA, GPIO.HIGH)   # 모터 활성화
    print(f"[ACTUATOR] ↑ Move UP {steps} steps (Current: {CUR_HEIGHT_STEP}/{HEIGHT_MAX})")

    for _ in range(steps):
        if exceed_max_height():
            print("[ACTUATOR] 🚫 최대 높이 도달, 이동 중단")
            break

        GPIO.output(PUL, GPIO.HIGH)
        sleep(STEP_DELAY)
        GPIO.output(PUL, GPIO.LOW)
        sleep(STEP_DELAY)

        CUR_HEIGHT_STEP += 1
        _write_height_to_file(CUR_HEIGHT_STEP)

    print(f"[ACTUATOR] ↑ Current step: {CUR_HEIGHT_STEP}/{HEIGHT_MAX}")

# === 아래로 이동 ===
def moveDown(steps: int = DEFAULT_STEP):
    """
    액추에이터를 아래로 (수축)
    - 얼굴이 화면 중앙보다 위에 있을 때 호출
    - 자동으로 최소 높이 검증 수행
    """
    global CUR_HEIGHT_STEP
    if exceed_min_height():
        print("[ACTUATOR] 🚫 최소 높이 도달, 이동 중단")
        return

    GPIO.output(DIR, GPIO.HIGH)   # 아래쪽 방향 (edited: LOW > HIGH)
    GPIO.output(ENA, GPIO.HIGH)  # 모터 활성화
    print(f"[ACTUATOR] ↓ Move DOWN {steps} steps (Current: {CUR_HEIGHT_STEP}/{HEIGHT_MAX})")

    for _ in range(steps):
        if exceed_min_height():
            print("[ACTUATOR] 🚫 최소 높이 도달, 이동 중단")
            break

        GPIO.output(PUL, GPIO.HIGH)
        sleep(STEP_DELAY)
        GPIO.output(PUL, GPIO.LOW)
        sleep(STEP_DELAY)

        CUR_HEIGHT_STEP -= 1
        CUR_HEIGHT_STEP = max(HEIGHT_MIN, CUR_HEIGHT_STEP)
        _write_height_to_file(CUR_HEIGHT_STEP)

    CUR_HEIGHT_STEP = max(0, CUR_HEIGHT_STEP)
    print(f"[ACTUATOR] ↓ Current step: {CUR_HEIGHT_STEP}/{HEIGHT_MAX}")

# === 프로그램 종료 시 원점 복귀 ===
def return_to_start():
    """
    프로그램 종료 시 현재 위치(CUR_HEIGHT_STEP)만큼 하강 → 기계 원점 복귀
    """
    global CUR_HEIGHT_STEP

    if CUR_HEIGHT_STEP <= 0:
        print("[ACTUATOR] Already at home position (0 step)")
        return

    print(f"[ACTUATOR] 🏁 Returning to home: {CUR_HEIGHT_STEP} steps down...")
    GPIO.output(ENA, GPIO.HIGH)
    GPIO.output(DIR, GPIO.LOW)

    for i in range(CUR_HEIGHT_STEP):
        GPIO.output(PUL, GPIO.HIGH)
        sleep(STEP_DELAY)
        GPIO.output(PUL, GPIO.LOW)
        sleep(STEP_DELAY)
        _write_height_to_file(CUR_HEIGHT_STEP - i - 1)

        if i % 100 == 0 and i > 0:
            print(f"   ↓ 진행: {i}/{CUR_HEIGHT_STEP}")

    CUR_HEIGHT_STEP = 0
    _write_height_to_file(CUR_HEIGHT_STEP)
    GPIO.output(ENA, GPIO.LOW)
    print("[ACTUATOR] ✅ Returned to home position (step=0)")

# === 종료 훅 ===
def on_shutdown():
    """
    프로그램 종료 시 자동으로 원점 복귀 수행
    """
    print("[SYSTEM] 🔻 Returning actuator to 0...")
    return_to_start()
    print("[SYSTEM] ✅ Actuator returned to home position.")
