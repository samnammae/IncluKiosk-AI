"""
Click Controller
Progressive Dwell Click 로직을 관리
"""
import numpy as np
import cv2


class ClickController:
    """Progressive Dwell Click 로직 관리"""
    
    def __init__(self, 
                 prepare_time=0.4,
                 progress_time=0.8,
                 click_time=1.2,
                 radius=40,
                 cooldown=0.5):
        
        self.prepare_time = prepare_time
        self.progress_time = progress_time
        self.click_time = click_time
        self.radius = radius
        self.cooldown = cooldown
        
        self.dwell_start_time = None
        self.dwell_center = None
        self.last_click_time = 0.0
        self.enabled = False  # 기본값: 비활성화
        self.click_count = 0
    
    def update(self, current_pos, current_time):
        """클릭 상태 업데이트"""
        if not self.enabled:
            self._reset()
            return self._idle_state()
        
        if current_time - self.last_click_time < self.cooldown:
            self._reset()
            return self._idle_state()
        
        if current_pos is None:
            self._reset()
            return self._idle_state()
        
        x, y = current_pos
        
        if self.dwell_start_time is None:
            self.dwell_start_time = current_time
            self.dwell_center = (x, y)
            return self._idle_state()
        
        dist = np.sqrt((x - self.dwell_center[0])**2 + 
                       (y - self.dwell_center[1])**2)
        
        if dist > self.radius:
            self._reset()
            return self._idle_state()
        
        elapsed = current_time - self.dwell_start_time
        
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
        
        elif elapsed >= self.progress_time:
            progress = (elapsed - self.progress_time) / (self.click_time - self.progress_time)
            return {
                'state': 'progress',
                'progress': 0.5 + progress * 0.5,
                'should_click': False,
                'center': self.dwell_center,
                'elapsed': elapsed
            }
        
        elif elapsed >= self.prepare_time:
            progress = (elapsed - self.prepare_time) / (self.progress_time - self.prepare_time)
            return {
                'state': 'prepare',
                'progress': progress * 0.5,
                'should_click': False,
                'center': self.dwell_center,
                'elapsed': elapsed
            }
        
        else:
            return self._idle_state()
    
    def _reset(self):
        """내부 상태 리셋"""
        self.dwell_start_time = None
        self.dwell_center = None
    
    def _idle_state(self):
        """대기 상태 반환"""
        return {
            'state': 'idle',
            'progress': 0.0,
            'should_click': False,
            'center': None,
            'elapsed': 0.0
        }
    
    def set_enabled(self, enabled):
        """클릭 활성화/비활성화"""
        self.enabled = enabled
        if not enabled:
            self._reset()


def draw_click_feedback(frame, click_state):
    """프레임에 클릭 진행 상황 표시"""
    state = click_state['state']
    
    if state == 'idle':
        return
    
    center = click_state['center']
    if center is None:
        return
    
    cx, cy = center
    progress = click_state['progress']
    
    radius_base = 30
    radius_outer = 35
    
    if state == 'prepare':
        color = (255, 200, 0)
        thickness = 3
        
        cv2.circle(frame, (cx, cy), radius_outer, color, 2)
        
        angle = int(360 * progress)
        if angle > 0:
            cv2.ellipse(frame, (cx, cy), (radius_base, radius_base),
                       -90, 0, angle, color, thickness)
        
        cv2.circle(frame, (cx, cy), 3, color, -1)
        
        cv2.putText(frame, "Preparing...", (cx - 50, cy - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    elif state == 'progress':
        color = (0, 255, 0)
        thickness = 4
        
        cv2.circle(frame, (cx, cy), radius_outer, color, 2)
        
        angle = int(360 * progress)
        cv2.ellipse(frame, (cx, cy), (radius_base, radius_base),
                   -90, 0, angle, color, thickness)
        
        cv2.circle(frame, (cx, cy), 5, color, -1)
        
        remaining = 1.2 - click_state['elapsed']
        cv2.putText(frame, f"Click in {remaining:.1f}s", (cx - 60, cy - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    elif state == 'click':
        color = (0, 0, 255)
        cv2.circle(frame, (cx, cy), radius_outer + 5, color, 5)
        cv2.circle(frame, (cx, cy), 8, color, -1)
        
        cv2.putText(frame, "CLICK!", (cx - 30, cy - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)