"""
Progressive Dwell Click Controller
시선이 일정 시간 한 곳에 머물면 클릭을 실행하는 컨트롤러
"""

import numpy as np
import time

class ClickController:
    """Progressive Dwell Click 로직 관리"""
    
    def __init__(self, 
                 prepare_time=0.4,    # 준비 단계 (파란색)
                 progress_time=0.8,   # 진행 단계 시작 (초록색)
                 click_time=1.2,      # 클릭 실행
                 radius=40,           # 허용 반경 (픽셀)
                 cooldown=0.5):       # 클릭 후 대기 시간
        
        self.prepare_time = prepare_time
        self.progress_time = progress_time
        self.click_time = click_time
        self.radius = radius
        self.cooldown = cooldown
        
        self.dwell_start_time = None
        self.dwell_center = None
        self.last_click_time = 0.0
        self.enabled = False
        self.click_count = 0
    
    def update(self, current_pos, current_time):
        """
        매 프레임 호출하여 클릭 상태 업데이트
        
        Args:
            current_pos: (x, y) 튜플 또는 None
            current_time: 현재 시간 (time.time())
        
        Returns:
            dict: {
                'state': 'idle' | 'prepare' | 'progress' | 'click',
                'progress': 0.0~1.0,
                'should_click': bool,
                'center': (x, y) or None,
                'elapsed': float
            }
        """
        if not self.enabled:
            self._reset()
            return self._idle_state()
        
        # 쿨다운 체크
        if current_time - self.last_click_time < self.cooldown:
            self._reset()
            return self._idle_state()
        
        # 시선 좌표 없음
        if current_pos is None:
            self._reset()
            return self._idle_state()
        
        x, y = current_pos
        
        # 첫 시작
        if self.dwell_start_time is None:
            self.dwell_start_time = current_time
            self.dwell_center = (x, y)
            return self._idle_state()
        
        # 거리 체크 (반경 벗어나면 리셋)
        dist = np.sqrt((x - self.dwell_center[0])**2 + 
                       (y - self.dwell_center[1])**2)
        
        if dist > self.radius:
            self._reset()
            return self._idle_state()
        
        # 경과 시간
        elapsed = current_time - self.dwell_start_time
        
        # 클릭 실행
        if elapsed >= self.click_time:
            self.last_click_time = current_time
            self.click_count += 1
            self._reset()
            return {
                'state': 'click',
                'progress': 1.0,
                'should_click': True,
                'center': self.dwell_center,
                'elapsed': elapsed
            }
        
        # 진행 단계 (초록색)
        elif elapsed >= self.progress_time:
            progress = (elapsed - self.progress_time) / (self.click_time - self.progress_time)
            return {
                'state': 'progress',
                'progress': 0.5 + progress * 0.5,
                'should_click': False,
                'center': self.dwell_center,
                'elapsed': elapsed
            }
        
        # 준비 단계 (파란색)
        elif elapsed >= self.prepare_time:
            progress = (elapsed - self.prepare_time) / (self.progress_time - self.prepare_time)
            return {
                'state': 'prepare',
                'progress': progress * 0.5,
                'should_click': False,
                'center': self.dwell_center,
                'elapsed': elapsed
            }
        
        # 대기 중
        else:
            return self._idle_state()
    
    def _reset(self):
        """내부 상태 리셋"""
        self.dwell_start_time = None
        self.dwell_center = None
    
    def _idle_state(self):
        """아이들 상태 반환"""
        return {
            'state': 'idle',
            'progress': 0.0,
            'should_click': False,
            'center': None,
            'elapsed': 0.0
        }
    
    def set_enabled(self, enabled):
        """클릭 기능 활성화/비활성화"""
        self.enabled = enabled
        if not enabled:
            self._reset()
    
    def get_click_count(self):
        """총 클릭 횟수 반환"""
        return self.click_count