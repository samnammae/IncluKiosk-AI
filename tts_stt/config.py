""" TTS/STT 설정 상수 """
import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent

# 기본 안내 멘트
DEFAULT_CHAT_GUIDE = "안녕하세요. 음성으로 주문을 도와드릴게요."
DEFAULT_ERROR_GUIDE = "죄송합니다, 말씀을 정확히 인식하지 못했습니다. 다시 한번 말씀해 주시겠어요?"
DEFAULT_CANCEL_GUIDE = "인식되는 음성이 없어 주문이 취소되었습니다."
DEFAULT_PRE_SOUND = str(PACKAGE_DIR / "start_recording.mp3")

# STT 설정
STT_SILENCE_SEC = 1.2
STT_MAX_DURATION = 60.0
STT_CALIB_SEC = 0.4
STT_SENSITIVITY = 2.0
STT_MIN_SPEECH_SEC = 0.2
STT_ENGINE = "google"
STT_SAMPLE_RATE = 16000
STT_PRE_SOUND_PAUSE = 0.1
STT_INITIAL_SILENCE_TIMEOUT = 5.0

# 장치 선호 키워드
PREFERRED_DEVICE_KEYWORDS = ("usb", "mic", "seeed", "respeaker")