"""
작동 확인 및 MAX_STEP 찾기 위한 테스트 코드 
"""
from time import sleep
import RPi.GPIO as GPIO

# ===== GPIO 핀 설정 =====
PUL = 17
DIR = 27
ENA = 22

GPIO.setmode(GPIO.BCM)
GPIO.setup(PUL, GPIO.OUT)
GPIO.setup(DIR, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)

print('Linear Actuator Control Initialized')

# ===== 동작 설정 =====
delay = 0.0004
test_steps = 500
cycle_delay = 1.0

# ===== 기본 함수 =====
def move_up(steps=test_steps):
    """액추에이터 위로 (확장)"""
    print(f"⬆️ 위로 {steps} 스텝 이동")
    GPIO.output(ENA, GPIO.HIGH)
    GPIO.output(DIR, GPIO.LOW)   # 확장 방향
    sleep(0.2)

    for step in range(steps):
        GPIO.output(PUL, GPIO.HIGH)
        sleep(delay)
        GPIO.output(PUL, GPIO.LOW)
        sleep(delay)
        if step % 100 == 0 and step > 0:
            print(f"   ↑ 진행 {step}/{steps}")

    GPIO.output(ENA, GPIO.LOW)
    sleep(0.5)

def move_down(steps=test_steps):
    """액추에이터 아래로 (수축)"""
    print(f"⬇️ 아래로 {steps} 스텝 이동")
    GPIO.output(ENA, GPIO.HIGH)
    GPIO.output(DIR, GPIO.HIGH)  # 수축 방향
    sleep(0.2)

    for step in range(steps):
        GPIO.output(PUL, GPIO.HIGH)
        sleep(delay)
        GPIO.output(PUL, GPIO.LOW)
        sleep(delay)
        if step % 100 == 0 and step > 0:
            print(f"   ↓ 진행 {step}/{steps}")

    GPIO.output(ENA, GPIO.LOW)
    sleep(0.5)

def test_cycle():
    """위/아래 500 스텝 테스트"""
    print("\n=== 500 Step Up/Down 테스트 시작 ===")
    move_up(500)
    move_down(500)
    print("=== 테스트 완료 ===")

# ===== Max Step 측정 + 하강 선택 =====
def find_max_step():
    print("\n=== Max Step 측정 시작 ===")
    total_steps = 0
    GPIO.output(ENA, GPIO.HIGH)
    GPIO.output(DIR, GPIO.LOW)
    sleep(0.5)
    print("더 올리고 싶은 스텝만큼 숫자를 입력해주세요(ex-1000,500,100)/측정을 멈추고 싶으시면 n을 입력하세요.")
    try:
        while True:
            user_input = input("몇스텝 올릴까요?(ex-1000,500,100) 또는 측정을 그만할까요?(n) ").strip().lower()
            if user_input == "n":
                print(f"🧩 총 {total_steps} 스텝까지 이동 가능 (Max Step 추정치)")
                break

            if not user_input.isdigit():
                print("❌ 숫자만 입력하거나 n을 입력해주세요.")
                continue

            step_val = int(user_input)
            for _ in range(step_val):
                GPIO.output(PUL, GPIO.HIGH)
                sleep(delay)
                GPIO.output(PUL, GPIO.LOW)
                sleep(delay)
            total_steps += step_val
            print(f"⬆️ 누적 이동: {total_steps} 스텝")



        # 측정 후 하강 여부 확인
        ans = input("내릴까요? (y/n): ").strip().lower()
        if ans == "y":
            try:
                down_steps = int(input("몇 스텝 내릴까요?: "))
                print(f"⬇️ {down_steps} 스텝 하강 중...")
                move_down(down_steps)
                print("하강 완료 ✅")
            except ValueError:
                print("❌ 잘못된 입력입니다. 숫자만 입력하세요.")
        else:
            print("하강 생략됨.")

    except KeyboardInterrupt:
        print("\n사용자 중단됨")
        print(f"마지막 총 누적 스텝:{total_steps}")

    finally:
        GPIO.output(ENA, GPIO.LOW)
        GPIO.cleanup()
        print("GPIO 정리 완료")

# ===== 메인 =====
if __name__ == "__main__":
    try:
        print("\n=== 리니어 액추에이터 테스트 모드 ===")
        print("1. 위/아래 500스텝 테스트")
        print("2. Max Step 측정 (측정 후 하강 여부 선택)")
        choice = input("선택 (1/2): ").strip()

        if choice == "1":
            test_cycle()
        elif choice == "2":
            find_max_step()
        else:
            print("잘못된 선택입니다.")

    except KeyboardInterrupt:
        print("\n사용자 중단됨")

    finally:
        GPIO.output(ENA, GPIO.LOW)
        GPIO.cleanup()
        print("프로그램 종료")
