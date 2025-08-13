import os
import traceback
import pyaudio
import wave
from google.cloud import texttospeech
from google.cloud import speech

# 서비스 계정 키 경로 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/pi/IncluKiosk/gcp-tts-key.json"

def record_audio(filename="recorded_audio.wav", duration=5, sample_rate=16000):
    """마이크에서 오디오를 녹음하는 함수"""
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    
    audio = pyaudio.PyAudio()
    
    try:
        # 마이크 스트림 열기
        stream = audio.open(format=format,
                          channels=channels,
                          rate=sample_rate,
                          input=True,
                          frames_per_buffer=chunk)
        
        print(f"🎤 {duration}초 동안 말씀해 주세요...")
        frames = []
        
        # 오디오 데이터 수집
        for i in range(0, int(sample_rate / chunk * duration)):
            data = stream.read(chunk)
            frames.append(data)
        
        print("🔴 녹음 완료!")
        
        # 스트림 종료
        stream.stop_stream()
        stream.close()
        
        # WAV 파일로 저장
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(format))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))
            
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
        
        # 음성 인식 설정
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="ko-KR",
        )
        
        # STT 요청
        print("🔄 STT 요청 중...")
        response = client.recognize(config=config, audio=audio)
        
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