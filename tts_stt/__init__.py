""" TTS/STT 패키지 - 주요 함수와 상수를 export """

# 설정
from .config import (
    DEFAULT_CHAT_GUIDE,
    DEFAULT_ERROR_GUIDE,
    DEFAULT_CANCEL_GUIDE,
    HEIGHT_GUIDE_SOUND,
    CALIB_GUIDE_SOUND,
    MODE_GUIDE_SOUND,
    STT_SILENCE_SEC,
    STT_MAX_DURATION,
    STT_CALIB_SEC,
    STT_SENSITIVITY,
    STT_MIN_SPEECH_SEC,
    STT_ENGINE,
    STT_SAMPLE_RATE,
)

# 오디오 유틸
from .audio_utils import (
    find_input_device_index,
    list_input_devices,
)

# TTS 함수
from .tts_functions import tts_play

# STT 함수
from .stt_functions import stt_once

# 안내 메시지
from .guide_messages import (
    play_chat_guide_message,
    play_error_guide_message,
    play_cancel_guide_message,
    play_height_guide_message,
    play_calib_guide_message,
    play_mode_guide_message,
)

__all__ = [
    # 설정
    "DEFAULT_CHAT_GUIDE",
    "DEFAULT_ERROR_GUIDE",
    "DEFAULT_CANCEL_GUIDE",
    "HEIGHT_GUIDE_SOUND",
    "CALIB_GUIDE_SOUND",
    "MODE_GUIDE_SOUND",
    "STT_SILENCE_SEC",
    "STT_MAX_DURATION",
    "STT_CALIB_SEC",
    "STT_SENSITIVITY",
    "STT_MIN_SPEECH_SEC",
    "STT_ENGINE",
    "STT_SAMPLE_RATE",
    # 오디오
    "find_input_device_index",
    "list_input_devices",
    # TTS
    "tts_play",
    # STT
    "stt_once",
    # 안내 메시지
    "play_chat_guide_message",
    "play_error_guide_message",
    "play_cancel_guide_message",
    "play_height_guide_message",
    "play_calib_guide_message",
    "play_mode_guide_message",
]