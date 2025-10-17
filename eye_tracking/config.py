"""
Eye Tracking Configuration
모든 설정값과 상수를 관리
"""
from pathlib import Path
import pyautogui

# =====================
# Directory & File Paths
# =====================
BASE_DIR = Path(__file__).resolve().parent.parent

# WebSocket
WS_URL = "ws://localhost:8766"

# Logging
LOG_FILE = "/tmp/eye_tracking_optimized.log"

# Screen Position File
# SCREEN_POSITION_FILE = str(BASE_DIR / "screen_position.txt")
SCREEN_POSITION_FILE = str(Path(__file__).resolve().parent / "screen_position.txt")

# =====================
# Performance Settings
# =====================
# FaceMesh 처리 주기 (프레임 단위)
FACEMESH_EVERY = 3

# Hand detection 주기 (프레임 단위)
HAND_EVERY = 4

# Face detection 주기 (프레임 단위)
DETECT_EVERY = 4

# Face TTL (초)
FACE_TTL = 3.0

# ROI 마진
ROI_MARGIN = 0.25

# Hand ROI Scale
HAND_ROI_SCALE = 1.5

# Fullframe fallback 허용
ALLOW_FALLBACK_FULLFRAME = True

# =====================
# Camera Settings
# =====================
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_BUFFER_SIZE = 1

# =====================
# Monitor Settings
# =====================
# 사용자와 모니터 사이 거리 (cm)
USER_MONITOR_DISTANCE = 60.0

# 모니터 실제 크기 (cm)
MONITOR_WIDTH_CM = 34.42
MONITOR_HEIGHT_CM = 19.36

# 화면 해상도 (픽셀)
# MONITOR_WIDTH_PX, MONITOR_HEIGHT_PX = pyautogui.size()
MONITOR_WIDTH_PX = 1920 
MONITOR_HEIGHT_PX = 1080 
CENTER_X = MONITOR_WIDTH_PX // 2
CENTER_Y = MONITOR_HEIGHT_PX // 2

# =====================
# Click Controller Settings
# =====================
CLICK_PREPARE_TIME = 0.4    # 준비 단계 시간 (초)
CLICK_PROGRESS_TIME = 0.8   # 진행 단계 시간 (초)
CLICK_TIME = 1.2            # 클릭 실행 시간 (초)
CLICK_RADIUS = 40           # 클릭 영역 반경 (픽셀)
CLICK_COOLDOWN = 0.5        # 클릭 후 쿨다운 (초)

# =====================
# Touch Settings
# =====================
TOUCH_HOLDOFF = 0.5  # 터치가 끝난 뒤 대기 시간 (초)

# =====================
# Hand Detection Settings
# =====================
FIST_COOLDOWN = 0.6  # 주먹 감지 쿨다운 (초)

# =====================
# Gaze Tracking Settings
# =====================
# Smoothing buffer 길이
FILTER_LENGTH = 8

# Gaze vector 길이 (디버깅용)
GAZE_LENGTH = 350

# Eye sphere base radius
BASE_RADIUS = 20

# Gaze angle ranges (degrees)
YAW_DEGREES = 15
PITCH_DEGREES = 5

# =====================
# MediaPipe Settings
# =====================
# Face Detection
FACE_DETECTION_MODEL = 0  # 0: 짧은 거리 모델
FACE_DETECTION_CONFIDENCE = 0.5

# Face Mesh
FACEMESH_MAX_FACES = 1
FACEMESH_MIN_DETECTION_CONFIDENCE = 0.3
FACEMESH_MIN_TRACKING_CONFIDENCE = 0.3

# Hands
HANDS_MODEL_COMPLEXITY = 0
HANDS_MAX_NUM = 1
HANDS_MIN_DETECTION_CONFIDENCE = 0.4
HANDS_MIN_TRACKING_CONFIDENCE = 0.4

# =====================
# Landmark Indices
# =====================
# Nose landmarks for coordinate box
NOSE_INDICES = [
    4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241, 
    461, 125, 354, 218, 438, 195, 167, 393, 165, 391,
    3, 248
]

# Iris landmarks
LEFT_IRIS_IDX = 468
RIGHT_IRIS_IDX = 473

# Face landmarks for height calculation
CHIN_IDX = 152
FOREHEAD_IDX = 10

# =====================
# Debug Settings
# =====================
DEBUG = False