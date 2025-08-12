import os
import signal
import asyncio
import json
import sys
import traceback
import logging
from typing import Optional

try:
    import RPi.GPIO as GPIO
except ImportError:
    print("[PIR] WARNING: RPi.GPIO not available - running in mock mode", flush=True)
    GPIO = None

import websockets

# ===== 로깅 설정 =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [PIR] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== 환경변수 설정 =====
def get_env_int(key: str, default: int) -> int:
    """환경변수를 정수로 가져오기 (검증 포함)"""
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        logger.warning(f"Invalid {key} value, using default: {default}")
        return default

def get_env_float(key: str, default: float) -> float:
    """환경변수를 실수로 가져오기 (검증 포함)"""
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        logger.warning(f"Invalid {key} value, using default: {default}")
        return default

def get_env_bool(key: str, default: bool) -> bool:
    """환경변수를 불린으로 가져오기"""
    value = os.environ.get(key, "1" if default else "0")
    return value.lower() in ("1", "true", "yes", "on")

# 설정값들
PIR_PIN = get_env_int("PIR_PIN", 18)
SLEEP_SEC = get_env_float("PIR_SLEEP", 0.15)
SERVER_URI = os.environ.get("SERVER_URI", "ws://localhost:8765")
ACTIVE_HIGH = get_env_bool("ACTIVE_HIGH", True)
WARMUP_SEC = get_env_float("PIR_WARMUP", 2.0)
CONFIRM_COUNT = get_env_int("PIR_CONFIRM_COUNT", 2)
PULL_MODE = os.environ.get("PIR_PULL", "DOWN").upper()

# 설정값 검증
if PIR_PIN < 0 or PIR_PIN > 27:
    logger.error(f"Invalid PIR_PIN: {PIR_PIN} (must be 0-27)")
    sys.exit(1)

if SLEEP_SEC <= 0:
    logger.error(f"Invalid PIR_SLEEP: {SLEEP_SEC} (must be > 0)")
    sys.exit(1)

if CONFIRM_COUNT <= 0:
    CONFIRM_COUNT = 1

if PULL_MODE not in ["DOWN", "UP", "NONE"]:
    logger.warning(f"Invalid PIR_PULL: {PULL_MODE}, using DOWN")
    PULL_MODE = "DOWN"

# ===== 전역 상태 =====
_running = True

def _handle_stop(signum, frame):
    """시그널 핸들러"""
    global _running
    _running = False
    logger.info(f"Signal {signum} received, stopping PIR worker...")

signal.signal(signal.SIGTERM, _handle_stop)
signal.signal(signal.SIGINT, _handle_stop)

def _resolve_pull_mode() -> Optional[int]:
    """Pull 모드를 GPIO 상수로 변환"""
    if not GPIO:
        return None
    
    if PULL_MODE == "DOWN":
        return GPIO.PUD_DOWN
    elif PULL_MODE == "UP":
        return GPIO.PUD_UP
    else:
        return None

async def send_pir_off():
    """서버에 PIR_OFF 메시지 전송"""
    max_retries = 3
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            async with websockets.connect(SERVER_URI) as ws:
                payload = json.dumps({"type": "PIR_OFF", "source": "worker"})
                await ws.send(payload)
            logger.info(f"PIR_OFF 메시지 전송 성공 (attempt {attempt + 1})")
            return True
        except Exception as e:
            logger.warning(f"PIR_OFF 전송 실패 (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                logger.error("PIR_OFF 전송 완전 실패")
                return False

def setup_gpio() -> bool:
    """GPIO 초기화"""
    if not GPIO:
        logger.warning("GPIO not available - mock mode")
        return True
    
    try:
        GPIO.setmode(GPIO.BCM)
        pull = _resolve_pull_mode()
        
        if pull is None:
            GPIO.setup(PIR_PIN, GPIO.IN)
            logger.info(f"GPIO {PIR_PIN} setup as INPUT (no pull resistor)")
        else:
            GPIO.setup(PIR_PIN, GPIO.IN, pull_up_down=pull)
            logger.info(f"GPIO {PIR_PIN} setup as INPUT with {PULL_MODE} pull resistor")
        
        return True
    except Exception as e:
        logger.error(f"GPIO setup failed: {e}")
        return False

def cleanup_gpio():
    """GPIO 정리"""
    if GPIO:
        try:
            GPIO.cleanup()
            logger.info("GPIO cleanup completed")
        except Exception as e:
            logger.warning(f"GPIO cleanup warning: {e}")

def read_pir_value() -> int:
    """PIR 센서값 읽기 (Mock 모드 지원)"""
    if not GPIO:
        # Mock mode: 5초 후 감지 시뮬레이션
        import time
        time.sleep(5)
        return 1 if ACTIVE_HIGH else 0
    
    try:
        return GPIO.input(PIR_PIN)
    except Exception as e:
        logger.error(f"PIR read error: {e}")
        return 0

async def monitor_pir():
    """PIR 센서 모니터링 메인 루프"""
    logger.info("PIR 센서 모니터링 시작")
    
    # 워밍업 대기
    if WARMUP_SEC > 0:
        logger.info(f"PIR 센서 워밍업 중... ({WARMUP_SEC}초)")
        await asyncio.sleep(WARMUP_SEC)
    
    last_value = None
    confirm_count = 0
    no_motion_count = 0
    
    while _running:
        try:
            # PIR 센서값 읽기
            current_value = read_pir_value()
            
            # 상태 변화 로깅
            if current_value != last_value:
                motion_detected = (current_value == 1 and ACTIVE_HIGH) or (current_value == 0 and not ACTIVE_HIGH)
                if motion_detected:
                    logger.info("Motion detected (상태 변화)")
                else:
                    logger.info("No motion (상태 변화)")
                last_value = current_value
            
            # 감지 판단 및 디바운싱
            motion_now = (current_value == 1) if ACTIVE_HIGH else (current_value == 0)
            
            if motion_now:
                confirm_count += 1
                no_motion_count = 0
                if confirm_count == 1:
                    logger.info(f"Motion 감지 시작... (확인 필요: {CONFIRM_COUNT}회)")
                # 빠른 확인을 위해 짧은 대기
                await asyncio.sleep(0.05)
            else:
                if confirm_count > 0:
                    logger.debug(f"Motion 중단 (확인 카운트 리셋: {confirm_count} -> 0)")
                confirm_count = 0
                no_motion_count += 1
            
            # 감지 확정 시 서버 통지 후 종료
            if confirm_count >= max(1, CONFIRM_COUNT):
                logger.info(f"Motion 확정! (확인 횟수: {confirm_count}) - 서버 통지 후 종료")
                success = await send_pir_off()
                if success:
                    logger.info("PIR 워커 정상 종료")
                else:
                    logger.warning("PIR 워커 종료 (통신 오류 발생)")
                break
            
            # 정상 대기
            await asyncio.sleep(SLEEP_SEC)
            
        except asyncio.CancelledError:
            logger.info("PIR 모니터링 취소됨")
            break
        except Exception as e:
            logger.error(f"PIR 모니터링 오류: {e}")
            await asyncio.sleep(1.0)  # 오류 시 대기 후 재시도

async def main():
    """메인 함수"""
    logger.info(
        f"PIR Worker 시작 | PIN={PIR_PIN}, ACTIVE_HIGH={ACTIVE_HIGH}, "
        f"PULL={PULL_MODE}, WARMUP={WARMUP_SEC}s, CONFIRM_COUNT={CONFIRM_COUNT}, "
        f"URI={SERVER_URI}"
    )

    # GPIO 초기화
    if not setup_gpio():
        logger.error("GPIO 초기화 실패 - 종료")
        sys.exit(1)
    
    try:
        # PIR 센서 모니터링 시작
        await monitor_pir()
    except KeyboardInterrupt:
        logger.info("사용자 인터럽트로 종료")
    except Exception as e:
        logger.error(f"PIR Worker 실행 중 오류: {e}")
        traceback.print_exc()
    finally:
        cleanup_gpio()
        logger.info("PIR Worker 종료")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"PIR Worker 시작 실패: {e}")
        sys.exit(1)