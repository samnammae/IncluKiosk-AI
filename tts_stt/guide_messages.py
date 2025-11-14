""" 안내 메시지 재생 함수들 """
import os
from typing import Optional

from .config import (
    DEFAULT_CHAT_GUIDE,
    DEFAULT_ERROR_GUIDE,
    DEFAULT_CANCEL_GUIDE,
    HEIGHT_GUIDE_SOUND,
    CALIB_GUIDE_SOUND,
    MODE_GUIDE_SOUND,
    CHAT_GUIDE_AUDIO,
    ERROR_GUIDE_AUDIO,
    CANCEL_GUIDE_AUDIO,
    HEIGHT_GUIDE_AUDIO,
    CALIB_GUIDE_AUDIO,
    MODE_GUIDE_AUDIO,
    USE_PREGENERATED_AUDIO
)
from .tts_functions import tts_play
from .audio_utils import play_audio_file


def _play_message(
    audio_file: str,
    fallback_text: str,
    lang: str = "ko-KR",
    voice: Optional[str] = None,
    speaking_rate: float = 1.0,
    pitch: float = 0.0,
    audio_encoding: str = "LINEAR16",
    use_pregenerated: bool = USE_PREGENERATED_AUDIO
) -> str:
    """
    안내 메시지 재생 (미리 추출된 파일 또는 실시간 TTS)
    
    Args:
        audio_file: 미리 추출된 오디오 파일 경로
        fallback_text: 파일이 없을 경우 사용할 텍스트
        use_pregenerated: True면 파일 사용, False면 실시간 TTS
    
    Returns:
        재생된 오디오 파일 경로
    """
    # 미리 추출된 파일 사용 모드이고 파일이 존재하면
    if use_pregenerated and os.path.isfile(audio_file):
        print(f"[TTS] 미리 추출된 파일 재생: {audio_file}")
        play_audio_file(audio_file, False)
        return audio_file
    else:
        # 파일이 없거나 실시간 TTS 모드면 Google TTS API 사용
        if use_pregenerated:
            print(f"[TTS] 경고: 미리 추출된 파일을 찾을 수 없습니다: {audio_file}")
            print(f"[TTS] 실시간 TTS로 대체합니다.")
        
        return tts_play(
            text=fallback_text,
            lang=lang,
            voice=voice,
            speaking_rate=speaking_rate,
            pitch=pitch,
            out_path=None,
            audio_encoding=audio_encoding
        )


def play_chat_guide_message(
    text: Optional[str] = None,
    lang: str = "ko-KR",
    voice: Optional[str] = None,
    speaking_rate: float = 1.0,
    pitch: float = 0.0,
    audio_encoding: str = "LINEAR16",
    use_pregenerated: bool = USE_PREGENERATED_AUDIO
) -> str:
    """주문 시작 안내 메시지 재생"""
    print(f"[TTS] 주문 안내 시작 ({audio_encoding})")
    result = _play_message(
        audio_file=CHAT_GUIDE_AUDIO,
        fallback_text=text or DEFAULT_CHAT_GUIDE,
        lang=lang,
        voice=voice,
        speaking_rate=speaking_rate,
        pitch=pitch,
        audio_encoding=audio_encoding,
        use_pregenerated=use_pregenerated
    )
    print("[TTS] 주문 안내 종료")
    return result


def play_error_guide_message(
    text: Optional[str] = None,
    lang: str = "ko-KR",
    voice: Optional[str] = None,
    speaking_rate: float = 1.0,
    pitch: float = 0.0,
    audio_encoding: str = "LINEAR16",
    use_pregenerated: bool = USE_PREGENERATED_AUDIO
) -> str:
    """음성 인식 오류 안내 메시지 재생"""
    print(f"[TTS] 오류 안내 시작 ({audio_encoding})")
    result = _play_message(
        audio_file=ERROR_GUIDE_AUDIO,
        fallback_text=text or DEFAULT_ERROR_GUIDE,
        lang=lang,
        voice=voice,
        speaking_rate=speaking_rate,
        pitch=pitch,
        audio_encoding=audio_encoding,
        use_pregenerated=use_pregenerated
    )
    print("[TTS] 오류 안내 종료")
    return result


def play_cancel_guide_message(
    text: Optional[str] = None,
    lang: str = "ko-KR",
    voice: Optional[str] = None,
    speaking_rate: float = 1.0,
    pitch: float = 0.0,
    audio_encoding: str = "LINEAR16",
    use_pregenerated: bool = USE_PREGENERATED_AUDIO
) -> str:
    """주문 취소 안내 메시지 재생"""
    print(f"[TTS] 취소 안내 시작 ({audio_encoding})")
    result = _play_message(
        audio_file=CANCEL_GUIDE_AUDIO,
        fallback_text=text or DEFAULT_CANCEL_GUIDE,
        lang=lang,
        voice=voice,
        speaking_rate=speaking_rate,
        pitch=pitch,
        audio_encoding=audio_encoding,
        use_pregenerated=use_pregenerated
    )
    print("[TTS] 취소 안내 종료")
    return result


def play_height_guide_message(
    text: Optional[str] = None,
    lang: str = "ko-KR",
    voice: Optional[str] = None,
    speaking_rate: float = 1.0,
    pitch: float = 0.0,
    audio_encoding: str = "LINEAR16",
    use_pregenerated: bool = USE_PREGENERATED_AUDIO
) -> str:
    """높이 조절 안내 메시지 재생"""
    print(f"[TTS] 높이 안내 시작 ({audio_encoding})")
    result = _play_message(
        audio_file=HEIGHT_GUIDE_AUDIO,
        fallback_text=text or HEIGHT_GUIDE_SOUND,
        lang=lang,
        voice=voice,
        speaking_rate=speaking_rate,
        pitch=pitch,
        audio_encoding=audio_encoding,
        use_pregenerated=use_pregenerated
    )
    print("[TTS] 높이 안내 종료")
    return result


def play_calib_guide_message(
    text: Optional[str] = None,
    lang: str = "ko-KR",
    voice: Optional[str] = None,
    speaking_rate: float = 1.0,
    pitch: float = 0.0,
    audio_encoding: str = "LINEAR16",
    use_pregenerated: bool = USE_PREGENERATED_AUDIO
) -> str:
    """눈 보정 안내 메시지 재생"""
    print(f"[TTS] 보정 안내 시작 ({audio_encoding})")
    result = _play_message(
        audio_file=CALIB_GUIDE_AUDIO,
        fallback_text=text or CALIB_GUIDE_SOUND,
        lang=lang,
        voice=voice,
        speaking_rate=speaking_rate,
        pitch=pitch,
        audio_encoding=audio_encoding,
        use_pregenerated=use_pregenerated
    )
    print("[TTS] 보정 안내 종료")
    return result


def play_mode_guide_message(
    text: Optional[str] = None,
    lang: str = "ko-KR",
    voice: Optional[str] = None,
    speaking_rate: float = 1.0,
    pitch: float = 0.0,
    audio_encoding: str = "LINEAR16",
    use_pregenerated: bool = USE_PREGENERATED_AUDIO
) -> str:
    """모드 선택 안내 메시지 재생"""
    print(f"[TTS] 모드 안내 시작 ({audio_encoding})")
    result = _play_message(
        audio_file=MODE_GUIDE_AUDIO,
        fallback_text=text or MODE_GUIDE_SOUND,
        lang=lang,
        voice=voice,
        speaking_rate=speaking_rate,
        pitch=pitch,
        audio_encoding=audio_encoding,
        use_pregenerated=use_pregenerated
    )
    print("[TTS] 모드 안내 종료")
    return result