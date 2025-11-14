import os
import pyautogui

# =========================
# 화면/마우스 관련 전역 설정
# =========================

# 모니터 스크린 크기 (px)
MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()
# MONITOR_WIDTH = 1080      # 라즈베리파이 가로
# MONITOR_HEIGHT = 1920     # 라즈베리파이 세로
CENTER_X = MONITOR_WIDTH // 2
CENTER_Y = MONITOR_HEIGHT // 2

# 사용자와 모니터 사이 거리 (cm)
USER_MONITOR_DISTANCE = 60.0
# 턱-이마 기본설정값 (cm)
DEFAULT_FACE_LENGTH = 15.0

# 화면 맵핑 좌우/상하 각도 범위
YAW_SENSITIVITY = 8
PITCH_SENSITIVITY = 16

# 모니터 실제 크기 (cm)
MONITOR_WIDTH_CM = 19.36    # 라즈베리파이 가로
MONITOR_HEIGHT_CM = 34.42   # 라즈베리파이 세로
# MONITOR_WIDTH_CM =35.5    # 가로
# MONITOR_HEIGHT_CM = 22.5   # 세로

CAMERA_INDEX = 0

FIST_MIN_HAND_SIZE = 50 # 최소 손 감지 크기
THUMB_THRESHOLD = 1.7 # 엄지 감지 완화 비율 (1.0=엄격, 1.3=권장, 1.5=관대)
FIST_HOLD_TIME = 1.5 # 주먹 유지 시간