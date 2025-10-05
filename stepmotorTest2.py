# Linear Actuator Control - Much slower than regular stepper motor
from time import sleep
import RPi.GPIO as GPIO

PUL = 17  # Stepper Drive Pulses
DIR = 27  # Controller Direction Bit
ENA = 22  # Controller Enable Bit

GPIO.setmode(GPIO.BCM)
GPIO.setup(PUL, GPIO.OUT)
GPIO.setup(DIR, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)

print('Linear Actuator Control Initialized')

# 리니어 액추에이터용 설정 (훨씬 느리게)
delay = 0.0005            # 10ms (원래: 0.0000001) - 100,000배 느리게!
steps_extend = 500      # 확장할 스텝 수 (원래: 5000)
steps_retract = 5000     # 수축할 스텝 수 (원래: 5000)
cycles = 3              # 테스트 사이클

print(f'Delay: {delay}s per step')
print(f'Steps: {steps_extend} per direction')

def extend_actuator():
    """액추에이터 확장"""
    print("=== 액추에이터 확장 시작 ===")
    GPIO.output(ENA, GPIO.HIGH)
    print('모터 활성화')
    
    sleep(1)  # 안정화 대기
    
    GPIO.output(DIR, GPIO.LOW)  # 확장 방향 (필요시 HIGH로 변경)
    print('확장 방향 설정')
    
    sleep(0.5)
    
    print('확장 중...')
    for step in range(steps_extend):
        GPIO.output(PUL, GPIO.HIGH)
        sleep(delay)
        GPIO.output(PUL, GPIO.LOW)
        sleep(delay)
        
        # 진행 상황 표시 (매 100스텝마다)
        if step % 100 == 0 and step > 0:
            print(f'확장 진행: {step}/{steps_extend} 스텝')
    
    print('확장 완료!')
    GPIO.output(ENA, GPIO.LOW)
    print('모터 비활성화')
    sleep(2)  # 다음 동작 전 대기

def retract_actuator():
    """액추에이터 수축"""
    print("=== 액추에이터 수축 시작 ===")
    GPIO.output(ENA, GPIO.HIGH)
    print('모터 활성화')
    
    sleep(1)
    
    GPIO.output(DIR, GPIO.HIGH)  # 수축 방향 (필요시 LOW로 변경)
    print('수축 방향 설정')
    
    sleep(0.5)
    
    print('수축 중...')
    for step in range(steps_retract):
        GPIO.output(PUL, GPIO.HIGH)
        sleep(delay)
        GPIO.output(PUL, GPIO.LOW)
        sleep(delay)
        
        if step % 100 == 0 and step > 0:
            print(f'수축 진행: {step}/{steps_retract} 스텝')
    
    print('수축 완료!')
    GPIO.output(ENA, GPIO.LOW)
    print('모터 비활성화')
    sleep(2)

def slow_test():
    """매우 느린 속도로 테스트"""
    print("=== 느린 속도 테스트 ===")
    test_delay = 0.0005  # 50ms - 매우 느리게
    test_steps = 5000
    
    GPIO.output(ENA, GPIO.HIGH)
    GPIO.output(DIR, GPIO.LOW)
    
    print(f'매우 느린 속도로 {test_steps}스텝 실행 (딜레이: {test_delay}s)')
    
    for i in range(test_steps):
        GPIO.output(PUL, GPIO.HIGH)
        sleep(test_delay)
        GPIO.output(PUL, GPIO.LOW)
        sleep(test_delay)
        
        if i % 20 == 0:
            print(f'진행: {i}/{test_steps}')
    
    GPIO.output(ENA, GPIO.LOW)
    print('느린 테스트 완료')

try:
    # 먼저 매우 느린 속도로 테스트
    # slow_test()
    # sleep(3)
    
    # 정상 속도로 사이클 실행
    for cycle in range(cycles):
        print(f'\n--- 사이클 {cycle + 1}/{cycles} ---')
        # extend_actuator()
        retract_actuator()
        print(f'사이클 {cycle + 1} 완료')
        
except KeyboardInterrupt:
    print("\n사용자 중단")
    
finally:
    GPIO.output(ENA, GPIO.LOW)  # 모터 비활성화
    GPIO.cleanup()
    print('프로그램 종료')
