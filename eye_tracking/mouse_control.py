"""
Mouse Control Module
마우스 제어 관련 함수
"""
import threading
import time
import pyautogui
from . import config


class MouseController:
    """마우스 이동 제어"""
    
    def __init__(self):
        self.target = [config.CENTER_X, config.CENTER_Y]
        self.lock = threading.Lock()
        self.enabled = False
        self.thread = None
        self.running = False
        
        # Touch 상태
        self.touch_active = False
        self.last_touch_end = 0.0
    
    def start(self):
        """마우스 제어 스레드 시작"""
        if self.thread is not None and self.thread.is_alive():
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._mouse_mover, daemon=True)
        self.thread.start()
    
    def stop(self):
        """마우스 제어 스레드 중지"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
    
    def set_target(self, x, y):
        """마우스 목표 위치 설정"""
        with self.lock:
            self.target[0] = x
            self.target[1] = y
    
    def set_enabled(self, enabled):
        """마우스 제어 활성화/비활성화"""
        self.enabled = enabled
    
    def set_touch_active(self, active):
        """터치 상태 설정"""
        self.touch_active = active
        if not active:
            self.last_touch_end = time.time()
    
    def _mouse_mover(self):
        """마우스 이동 스레드 (백그라운드)"""
        last_xy = (0, 0)
        
        while self.running:
            # print("🟠[Mouse] running mouse over thread")
            # 터치 중이거나 홀드오프 기간이면 대기
            if self.touch_active or (time.time() - self.last_touch_end < config.TOUCH_HOLDOFF):
                print("🟠[Mouse] Touch or HoldOFF -> WAIT!")
                time.sleep(0.01)
                continue
            
            if self.enabled:
                with self.lock:
                    xy = (self.target[0], self.target[1])
                
                if xy != last_xy:
                    last_xy = xy
                    try:
                        pyautogui.moveTo(xy[0], xy[1])
                    except Exception:
                        pass
            
            time.sleep(0.01)


def write_screen_position(x, y):
    """화면 위치를 파일에 기록"""
    try:
        with open(config.SCREEN_POSITION_FILE, 'w') as f:
            f.write(f"{x},{y}\n")
    except Exception:
        pass