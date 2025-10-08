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


def stt_once_with_error_handling(
    on_error_callback=None,
    on_error_end_callback=None,
    error_guide_text: str = DEFAULT_ERROR_GUIDE,
    error_guide_lang: str = "ko-KR",
    error_guide_voice: Optional[str] = None,
    **stt_kwargs
) -> tuple[str, bool]:
    """
    STT 실행 + 오류 시 자동 안내
    
    Returns:
        (text, success): 인식된 텍스트, 성공 여부
    """
    from .stt_functions import stt_once
    
    try:
        text = stt_once(**stt_kwargs)
        
        if not text or not text.strip():
            print("[STT] 음성 인식 실패 → 오류 처리 시작")
            
            # 1. 오류 발생 알림
            if on_error_callback:
                try:
                    on_error_callback()
                    print("[STT] STT_ERR 콜백 실행 완료")
                except Exception as e:
                    print(f"[STT] STT_ERR 콜백 실패: {e}")
            
            # 2. 오류 안내 음성 재생
            try:
                print(f"[STT] 오류 안내 TTS 재생: \"{error_guide_text}\"")
                play_error_guide_message(
                    text=error_guide_text,
                    lang=error_guide_lang,
                    voice=error_guide_voice
                )
                print("[STT] 오류 안내 TTS 재생 완료")
            except Exception as e:
                print(f"[STT] 오류 안내 TTS 실패: {e}")
            
            # 3. 오류 안내 종료 알림
            if on_error_end_callback:
                try:
                    on_error_end_callback()
                    print("[STT] ERR_END 콜백 실행 완료")
                except Exception as e:
                    print(f"[STT] ERR_END 콜백 실패: {e}")
            
            return "", False
        
        return text, True
        
    except Exception as e:
        print(f"[STT] 예외 발생: {e}")
        
        # 예외 발생 시에도 동일한 오류 처리
        if on_error_callback:
            try:
                on_error_callback()
            except Exception:
                pass
        
        try:
            play_error_guide_message(
                text=error_guide_text,
                lang=error_guide_lang,
                voice=error_guide_voice
            )
        except Exception:
            pass
        
        if on_error_end_callback:
            try:
                on_error_end_callback()
            except Exception:
                pass
        
        return "", False