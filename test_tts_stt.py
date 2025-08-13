import os
import traceback
import pyaudio
import wave
from google.cloud import texttospeech
from google.cloud import speech

# 서비스 계정 키 경로 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/pi/IncluKiosk/gcp-tts-key.json"

def list_input_devices():
    """디버깅용: 입력 가능한 PyAudio 장치 목록 출력"""
    p = pyaudio.PyAudio()
    print("\n[Input-capable devices]")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if int(info.get("maxInputChannels", 0)) > 0:
            print(f"  {i}: {info.get('name')} (in={info.get('maxInputChannels')}, rate={info.get('defaultSampleRate')})")
    p.terminate()

def find_input_device_index(prefer_keywords=("respeaker", "usb", "mic", "seeed")):
    """
    PyAudio 입력 장치 인덱스를 자동 탐색.
    prefer_keywords에 해당하는 단어가 '이름'에 포함된 입력 장치를 우선 선택.
    매칭이 없으면, 입력 가능한 장치 중 첫 번째를 반환.
    """
    p = pyaudio.PyAudio()
    target = None
    prefer = [k.lower() for k in prefer_keywords]
    try:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            name = str(info.get("name", "")).lower()
            max_in = int(info.get("maxInputChannels", 0))
            if max_in > 0:
                if any(k in name for k in prefer):
                    target = i
                    break
                if target is None:
                    target = i
    finally:
        p.terminate()
    return target

def record_audio(filename="recorded_audio.wav", duration=5, sample_rate=16000):
    """마이크에서 오디오를 녹음하는 함수 (입력 장치 자동 선택)"""
    chunk = 1024
    fmt = pyaudio.paInt16
    channels = 1

    audio = pyaudio.PyAudio()
    device_index = find_input_device_index()

    try:
        if device_index is None:
            print("❗ 입력 가능한 장치를 찾지 못했습니다. 장치 목록을 확인하세요.")
            list_input_devices()
            raise RuntimeError("No input-capable device found")

        # 선택된 장치 정보 출력(디버깅)
        dev_info = audio.get_device_info_by_index(device_index)
        print(f"🎛️  Using input device index={device_index}, name='{dev_info.get('name')}'")

        # 마이크 스트림 열기
        stream = audio.open(format=fmt,
                            channels=channels,
                            rate=sample_rate,
                            input=True,
                            input_device_index=device_index,
                            frames_per_buffer=chunk)

        print(f"🎤 {duration}초 동안 말씀해 주세요...")
        frames = []

        # 오디오 데이터 수집 (오버플로우 시 예외 무시)
        for _ in range(0, int(sample_rate / chunk * duration)):
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)

        print("🔴 녹음 완료!")

        # 스트림 종료
        stream.stop_stream()
        stream.close()

        # WAV 파일로 저장
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(fmt))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))

    except Exception:
        # 에러 시 장치 목록을 보여주어 선택에 도움
        list_input_devices()
        raise
    finally:
        audio.terminate()

    return filename

def text_to_speech():
    """TTS 기능"""
    try:
        # 사용자 입력 받기
        text = input("📝 읽어줄 문장을 입력하세요: ").strip()
        if not text:
            raise ValueError("입력된 문장이 없습니다.")

        # TTS 클라이언트 초기화
        client = texttospeech.TextToSpeechClient()

        # 요청 구성
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="ko-KR", name="ko-KR-Wavenet-B"
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        # TTS 요청
        print("🔄 TTS 요청 중...")
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        print("✅ TTS 응답 완료.")

        # 음성 저장
        output_path = "output.mp3"
        with open(output_path, "wb") as out:
            out.write(response.audio_content)
        print(f"💾 음성 파일 저장 완료: {output_path}")

        # 음성 재생
        print("🔊 음성 재생 중...")
        os.system(f"mpg123 {output_path}")

    except Exception as e:
        print("\n❗TTS 에러 발생:")
        print(e)
        traceback.print_exc()

def speech_to_text():
    """STT 기능"""
    try:
        # 녹음 시간 설정
        duration = input("🎙️  녹음 시간을 입력하세요 (초, 기본값 5초): ").strip()
        duration = int(duration) if duration.isdigit() else 5

        # 오디오 녹음
        audio_file = record_audio(duration=duration)

        # STT 클라이언트 초기화
        client = speech.SpeechClient()

        # 오디오 파일 읽기
        with open(audio_file, "rb") as audio_content:
            content = audio_content.read()

        # 음성 인식 설정 (녹음 설정과 일치)
        audio_msg = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="ko-KR",
        )

        # STT 요청
        print("🔄 STT 요청 중...")
        response = client.recognize(config=config, audio=audio_msg)

        # 결과 출력
        if response.results:
            print("✅ STT 완료!")
            print("📄 인식된 텍스트:")
            for result in response.results:
                print(f"   '{result.alternatives[0].transcript}'")
                print(f"   신뢰도: {result.alternatives[0].confidence:.2%}")
        else:
            print("❌ 음성을 인식하지 못했습니다. 다시 시도해 주세요.")

        # 임시 파일 삭제
        if os.path.exists(audio_file):
            os.remove(audio_file)

    except Exception as e:
        print("\n❗STT 에러 발생:")
        print(e)
        traceback.print_exc()

def main():
    """메인 함수"""
    print("=" * 50)
    print("🎙️  Google Cloud TTS & STT 프로그램")
    print("=" * 50)

    while True:
        print("\n📋 메뉴를 선택하세요:")
        print("1️⃣  TTS (Text-to-Speech) - 텍스트를 음성으로")
        print("2️⃣  STT (Speech-to-Text) - 음성을 텍스트로")
        print("3️⃣  종료")

        choice = input("\n선택 (1/2/3): ").strip()

        if choice == "1":
            print("\n🔤 TTS 모드를 선택했습니다.")
            text_to_speech()

        elif choice == "2":
            print("\n🎤 STT 모드를 선택했습니다.")
            speech_to_text()

        elif choice == "3":
            print("👋 프로그램을 종료합니다.")
            break

        else:
            print("❌ 잘못된 선택입니다. 1, 2, 3 중에서 선택해 주세요.")

        print("\n" + "─" * 50)

if __name__ == "__main__":
    main()
