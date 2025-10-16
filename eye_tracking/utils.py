"""
Utility Functions
로깅, 파일 처리 등 유틸리티 함수
"""
import logging
import sys
from pathlib import Path
from . import config


def setup_logging():
    """로깅 설정"""
    try:
        logging.basicConfig(
            filename=config.LOG_FILE,
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(message)s"
        )
    except PermissionError:
        # 권한 문제 시 콘솔에만 출력
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(message)s"
        )
        print(f"⚠ 로그 파일 생성 실패, 콘솔 출력만 사용: {config.LOG_FILE}")


def dbg(msg):
    """디버그 메시지 출력 (콘솔 + 파일)"""
    print(msg, flush=True)
    logging.debug(msg)


def print_boot_info():
    """부팅 정보 출력"""
    import os
    dbg("=== [BOOT] eye_tracking_worker.py start ===")
    dbg(f"WS_URL={config.WS_URL}")
    dbg(f"PWD={os.getcwd()}, USER={os.getenv('USER')}, DISPLAY={os.getenv('DISPLAY')}")