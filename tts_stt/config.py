""" TTS/STT 설정 상수 """
import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = PACKAGE_DIR / "audio"

# 기본 안내 멘트
DEFAULT_CHAT_GUIDE = "안녕하세요. 음성으로 주문을 도와드릴게요."
DEFAULT_ERROR_GUIDE = "죄송합니다, 말씀을 정확히 인식하지 못했습니다. 다시 한번 말씀해 주시겠어요?"
DEFAULT_CANCEL_GUIDE = "인식되는 음성이 없어 주문이 취소되었습니다."
DEFAULT_PRE_SOUND = str(PACKAGE_DIR / "start_recording.mp3")

# 화면별 안내 멘트
HEIGHT_GUIDE_SOUND = "높이 조절 중입니다."
CALIB_GUIDE_SOUND = "눈 보정을 실행 중입니다. 화면 중앙을 바라봐주세요."
MODE_GUIDE_SOUND = "안녕하세요. 음성 주문을 원하시면 주먹을 쥐어주세요."

# 미리 추출된 오디오 파일 경로
CHAT_GUIDE_AUDIO = str(AUDIO_DIR / "chat_guide.wav")
ERROR_GUIDE_AUDIO = str(AUDIO_DIR / "error_guide.wav")
CANCEL_GUIDE_AUDIO = str(AUDIO_DIR / "cancel_guide.wav")
HEIGHT_GUIDE_AUDIO = str(AUDIO_DIR / "height_guide.wav")
CALIB_GUIDE_AUDIO = str(AUDIO_DIR / "calib_guide.wav")
MODE_GUIDE_AUDIO = str(AUDIO_DIR / "mode_guide.wav")

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

# 미리 추출된 오디오 파일 사용 여부 (True: 파일 사용, False: 실시간 TTS)
USE_PREGENERATED_AUDIO = True