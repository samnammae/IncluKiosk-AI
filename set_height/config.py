"""높이 조절 시스템 설정"""
import os

# ====== 카메라 설정 ======
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480

# ====== 감지 설정 ======
MIN_DET_CONF = 0.4      # 최소 감지 신뢰도
PERSON_SCORE_TH = 0.3   # 사람 감지 임계값

# ====== 높이 조절 설정 ======
TARGET_OFFSET_PCT = 0.08    # 목표 위치 오프셋 (8% 위로)
DEADBAND_PCT = 0.06         # 데드밴드 (중앙 안정 범위)
EMA_ALPHA = 0.3             # 지수 이동 평균 알파값
STABLE_FRAMES = 10          # 안정화 판단 프레임 수

# ====== 모터 속도 설정 ======
WITH_FACE = 100         # 얼굴 감지 시 모터 속도
WITHOUT_FACE = 500      # 얼굴 미감지 시 모터 속도

# ====== 타임아웃 설정 ======
AUTO_EXIT_STABLE_TIME = 3   # 안정화 후 자동 종료 시간 (초)
NO_DETECTION_TIMEOUT = 15   # 감지 없을 때 타임아웃 (초)
PRINT_EVERY = 0.15          # 상태 출력 주기 (초)

# ====== 모델 경로 ======
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

FACE_MODEL = os.path.join(
    MODELS_DIR, 
    "ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"
)
PERSON_MODEL = os.path.join(
    MODELS_DIR,
    "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
)

# ====== WebSocket 설정 ======
HUB_WS_URL = "ws://127.0.0.1:8766"

# ====== 디버그 설정 ======
DEBUG = True