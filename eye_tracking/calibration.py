"""
Calibration Module
시선 캘리브레이션 관련 기능
"""
import numpy as np
from . import config
from .gaze_tracking import compute_scale, create_monitor_plane
import time


class CalibrationState:
    """캘리브레이션 상태 관리"""
    
    def __init__(self):
        # Eye sphere tracking
        self.left_sphere_locked = False
        self.left_sphere_local_offset = None
        self.left_calibration_nose_scale = None
        
        self.right_sphere_locked = False
        self.right_sphere_local_offset = None
        self.right_calibration_nose_scale = None
        
        # Monitor plane
        self.monitor_corners = None
        self.monitor_center_w = None
        self.monitor_normal_w = None
        self.units_per_cm = None
        
        # Offset calibration
        self.offset_yaw = 0
        self.offset_pitch = 0
    
    def is_calibrated(self):
        """캘리브레이션 완료 여부"""
        return self.left_sphere_locked and self.right_sphere_locked
    
    def reset(self):
        """캘리브레이션 초기화"""
        self.left_sphere_locked = False
        self.left_sphere_local_offset = None
        self.left_calibration_nose_scale = None
        
        self.right_sphere_locked = False
        self.right_sphere_local_offset = None
        self.right_calibration_nose_scale = None
        
        self.monitor_corners = None
        self.monitor_center_w = None
        self.monitor_normal_w = None
        self.units_per_cm = None


def perform_eye_calibration(calib_state, head_center, R_final, nose_points_3d, 
                            iris_3d_left, iris_3d_right, face_landmarks=None, w=None, h=None):
    """
    눈 위치 캘리브레이션 수행
    
    Args:
        calib_state: CalibrationState 객체
        head_center: 머리 중심점
        R_final: 회전 행렬
        nose_points_3d: 코 랜드마크 3D 점들
        iris_3d_left: 왼쪽 홍채 3D 좌표
        iris_3d_right: 오른쪽 홍채 3D 좌표
        face_landmarks: 얼굴 랜드마크 (Optional, monitor plane 생성 시 필요)
        w, h: 프레임 크기 (Optional, monitor plane 생성 시 필요)
    """    
    current_nose_scale = compute_scale(nose_points_3d)
    camera_dir_world = np.array([0, 0, 1], dtype=float)
    camera_dir_local = R_final.T @ camera_dir_world
    
    # Left eye
    calib_state.left_sphere_local_offset = R_final.T @ (iris_3d_left - head_center)
    calib_state.left_sphere_local_offset += config.BASE_RADIUS * camera_dir_local
    calib_state.left_calibration_nose_scale = current_nose_scale
    calib_state.left_sphere_locked = True
    
    # Right eye
    calib_state.right_sphere_local_offset = R_final.T @ (iris_3d_right - head_center)
    calib_state.right_sphere_local_offset += config.BASE_RADIUS * camera_dir_local
    calib_state.right_calibration_nose_scale = current_nose_scale
    calib_state.right_sphere_locked = True
    
    # Monitor plane (only if face_landmarks provided)
    if face_landmarks is not None and w is not None and h is not None:
        sphere_world_l_calib = head_center + R_final @ calib_state.left_sphere_local_offset
        sphere_world_r_calib = head_center + R_final @ calib_state.right_sphere_local_offset
        
        left_dir = iris_3d_left - sphere_world_l_calib
        right_dir = iris_3d_right - sphere_world_r_calib
        
        if np.linalg.norm(left_dir) > 1e-9:
            left_dir /= np.linalg.norm(left_dir)
        if np.linalg.norm(right_dir) > 1e-9:
            right_dir /= np.linalg.norm(right_dir)
        
        forward_hint = (left_dir + right_dir) * 0.5
        if np.linalg.norm(forward_hint) > 1e-9:
            forward_hint /= np.linalg.norm(forward_hint)
        else:
            forward_hint = None
        
        gaze_origin = (sphere_world_l_calib + sphere_world_r_calib) / 2
        gaze_dir = forward_hint
        
        corners, center_w, normal_w, upc = create_monitor_plane(
            head_center, R_final, face_landmarks, w, h,
            forward_hint=forward_hint,
            gaze_origin=gaze_origin,
            gaze_dir=gaze_dir
        )
        
        calib_state.monitor_corners = corners
        calib_state.monitor_center_w = center_w
        calib_state.monitor_normal_w = normal_w
        calib_state.units_per_cm = upc


def perform_screen_calibration(calib_state, head_center, R_final, nose_points_3d,
                               iris_3d_left, iris_3d_right):
    """
    화면 중앙 캘리브레이션 수행 (오프셋 조정)
    
    Returns:
        offset_yaw, offset_pitch
    """
    from .gaze_tracking import convert_gaze_to_screen_coordinates
    
    current_nose_scale = compute_scale(nose_points_3d)
    
    # Left eye sphere
    scale_ratio_l = current_nose_scale / calib_state.left_calibration_nose_scale \
                    if calib_state.left_calibration_nose_scale else 1.0
    sphere_world_l = head_center + R_final @ (calib_state.left_sphere_local_offset * scale_ratio_l)
    
    # Right eye sphere
    scale_ratio_r = current_nose_scale / calib_state.right_calibration_nose_scale \
                    if calib_state.right_calibration_nose_scale else 1.0
    sphere_world_r = head_center + R_final @ (calib_state.right_sphere_local_offset * scale_ratio_r)
    
    # Gaze directions
    left_gaze_dir = iris_3d_left - sphere_world_l
    right_gaze_dir = iris_3d_right - sphere_world_r
    
    if np.linalg.norm(left_gaze_dir) > 1e-9:
        left_gaze_dir /= np.linalg.norm(left_gaze_dir)
    if np.linalg.norm(right_gaze_dir) > 1e-9:
        right_gaze_dir /= np.linalg.norm(right_gaze_dir)
    
    current_combined_direction = (left_gaze_dir + right_gaze_dir) / 2.0
    if np.linalg.norm(current_combined_direction) > 1e-9:
        current_combined_direction /= np.linalg.norm(current_combined_direction)
    
    # 현재 각도 계산 (오프셋 없이)
    _, _, raw_yaw, raw_pitch = convert_gaze_to_screen_coordinates(
        current_combined_direction, 0, 0
    )
    
    # 오프셋 계산 (화면 중앙을 가리키도록)
    calib_state.offset_yaw = -raw_yaw
    calib_state.offset_pitch = -raw_pitch
    
    return calib_state.offset_yaw, calib_state.offset_pitch


def perform_full_calibration(calib_state, head_center, R_final, nose_points_3d,
                            iris_3d_left, iris_3d_right, face_landmarks, w, h):
    """
    화면 중앙을 볼 때 한 번에 캘리브레이션 수행 (c + s 통합)
    
    사용자가 화면 중앙을 보고 있을 때 이 함수를 호출하면
    눈 위치 고정 + 화면 중앙 보정을 한 번에 처리
    
    Args:
        calib_state: CalibrationState 객체
        head_center: 머리 중심점
        R_final: 회전 행렬
        nose_points_3d: 코 랜드마크 3D 점들
        iris_3d_left: 왼쪽 홍채 3D 좌표
        iris_3d_right: 오른쪽 홍채 3D 좌표
        face_landmarks: 얼굴 랜드마크
        w, h: 프레임 크기
        
    Returns:
        True: 성공, False: 실패
    """
    import pyautogui
    # 1단계: 눈 위치 캘리브레이션 (c 키와 동일)
    perform_eye_calibration(
        calib_state, head_center, R_final, nose_points_3d,
        iris_3d_left, iris_3d_right, face_landmarks, w, h
    )
    
    # 2단계: 화면 중앙 보정 (s 키와 동일)
    # 이제 is_calibrated()가 True이므로 실행 가능
    offset_yaw, offset_pitch = perform_screen_calibration(
        calib_state, head_center, R_final, nose_points_3d,
        iris_3d_left, iris_3d_right
    )
    
    # ✅ 3단계: 마우스를 화면 중앙으로 즉시 이동
    print(f"🟡[eye_worker WS] [Calibration] 마우스를 화면 중앙({config.CENTER_X}, {config.CENTER_Y})으로 이동")
    pyautogui.moveTo(config.CENTER_X, config.CENTER_Y, duration=0)  # duration=0: 즉시 이동
    time.sleep(5)
    print(f"🟡[eye_worker WS] [Calibration] 센터에서 5초간 머무름")
    
    
    
    return True


def compute_gaze_vectors(calib_state, head_center, R_final, nose_points_3d,
                        iris_3d_left, iris_3d_right):
    """
    캘리브레이션된 상태에서 시선 벡터 계산
    
    Returns:
        left_gaze_dir, right_gaze_dir, combined_gaze_dir
    """
    if not calib_state.is_calibrated():
        return None, None, None
    
    current_nose_scale = compute_scale(nose_points_3d)
    
    # Left eye sphere
    scale_ratio_l = current_nose_scale / calib_state.left_calibration_nose_scale \
                    if calib_state.left_calibration_nose_scale else 1.0
    scaled_offset_l = calib_state.left_sphere_local_offset * scale_ratio_l
    sphere_world_l = head_center + R_final @ scaled_offset_l
    
    # Right eye sphere
    scale_ratio_r = current_nose_scale / calib_state.right_calibration_nose_scale \
                    if calib_state.right_calibration_nose_scale else 1.0
    scaled_offset_r = calib_state.right_sphere_local_offset * scale_ratio_r
    sphere_world_r = head_center + R_final @ scaled_offset_r
    
    # Gaze directions
    left_gaze_dir = iris_3d_left - sphere_world_l
    left_gaze_dir /= np.linalg.norm(left_gaze_dir)
    
    right_gaze_dir = iris_3d_right - sphere_world_r
    right_gaze_dir /= np.linalg.norm(right_gaze_dir)
    
    # Combined
    combined_gaze_dir = (left_gaze_dir + right_gaze_dir) / 2
    combined_gaze_dir /= np.linalg.norm(combined_gaze_dir)
    
    return left_gaze_dir, right_gaze_dir, combined_gaze_dir