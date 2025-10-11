""" 안내 메시지 재생 함수들 """
from typing import Optional

from .config import DEFAULT_CHAT_GUIDE, DEFAULT_ERROR_GUIDE, DEFAULT_CANCEL_GUIDE
from .tts_functions import tts_play


def play_chat_guide_message(
    text: str = DEFAULT_CHAT_GUIDE,
    lang: str = "ko-KR",
    voice: Optional[str] = None,
    speaking_rate: float = 1.0,
    pitch: float = 0.0,
    audio_encoding: str = "LINEAR16"
) -> str:
    """주문 시작 안내 메시지 재생"""
    print(f"[TTS] 주문 안내 시작: \"{text}\" ({audio_encoding})")
    result = tts_play(
        text=text,
        lang=lang,
        voice=voice,
        speaking_rate=speaking_rate,
        pitch=pitch,
        out_path=None,
        audio_encoding=audio_encoding
    )
    print("[TTS] 주문 안내 종료")
    return result


def play_error_guide_message(
    text: str = DEFAULT_ERROR_GUIDE,
    lang: str = "ko-KR",
    voice: Optional[str] = None,
    speaking_rate: float = 1.0,
    pitch: float = 0.0,
    audio_encoding: str = "LINEAR16"
) -> str:
    """음성 인식 오류 안내 메시지 재생"""
    print(f"[TTS] 오류 안내 시작: \"{text}\" ({audio_encoding})")
    result = tts_play(
        text=text,
        lang=lang,
        voice=voice,
        speaking_rate=speaking_rate,
        pitch=pitch,
        out_path=None,
        audio_encoding=audio_encoding
    )
    print("[TTS] 오류 안내 종료")
    return result


def play_cancel_guide_message(
    text: str = DEFAULT_CANCEL_GUIDE,
    lang: str = "ko-KR",
    voice: Optional[str] = None,
    speaking_rate: float = 1.0,
    pitch: float = 0.0,
    audio_encoding: str = "LINEAR16"
) -> str:
    """주문 취소 안내 메시지 재생"""
    print(f"[TTS] 취소 안내 시작: \"{text}\" ({audio_encoding})")
    result = tts_play(
        text=text,
        lang=lang,
        voice=voice,
        speaking_rate=speaking_rate,
        pitch=pitch,
        out_path=None,
        audio_encoding=audio_encoding
    )
    print("[TTS] 취소 안내 종료")
    return result