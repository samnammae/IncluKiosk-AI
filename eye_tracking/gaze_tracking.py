"""
Gaze Tracking Module
시선 추적 및 좌표 변환 관련 함수
"""
import numpy as np
import math
from scipy.spatial.transform import Rotation as Rscipy
from . import config


def _normalize(v):
    """벡터 정규화"""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def compute_scale(points_3d):
    """3D 점들의 평균 거리 계산"""
    n = len(points_3d)
    total = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(points_3d[i] - points_3d[j])
            total += dist
            count += 1
    return total / count if count > 0 else 1.0


def compute_coordinate_box(face_landmarks, indices, ref_matrix_container, w, h):
    """
    얼굴 랜드마크로부터 좌표계 계산
    
    Returns:
        center: 중심점
        R_final: 회전 행렬
        points_3d: 3D 점들
    """
    points_3d = np.array([
        [face_landmarks[i].x * w, face_landmarks[i].y * h, face_landmarks[i].z * w]
        for i in indices
    ])
    
    center = np.mean(points_3d, axis=0)
    centered = points_3d - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:, np.argsort(-eigvals)]
    
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1
    
    r = Rscipy.from_matrix(eigvecs)
    roll, pitch, yaw = r.as_euler('zyx', degrees=False)
    R_final = Rscipy.from_euler('zyx', [roll, pitch, yaw]).as_matrix()
    
    if ref_matrix_container[0] is None:
        ref_matrix_container[0] = R_final.copy()
    else:
        R_ref = ref_matrix_container[0]
        for i in range(3):
            if np.dot(R_final[:, i], R_ref[:, i]) < 0:
                R_final[:, i] *= -1
    
    return center, R_final, points_3d


def create_monitor_plane(head_center, R_final, face_landmarks, w, h, 
                         forward_hint=None, gaze_origin=None, gaze_dir=None):
    """
    가상 모니터 평면 생성
    
    Returns:
        corners: 모니터 코너 4개
        center_w: 모니터 중심
        normal_w: 모니터 법선 벡터
        upc: units per cm
    """
    try:
        lm_chin = face_landmarks[config.CHIN_IDX]
        lm_fore = face_landmarks[config.FOREHEAD_IDX]
        chin_w = np.array([lm_chin.x * w, lm_chin.y * h, lm_chin.z * w], dtype=float)
        fore_w = np.array([lm_fore.x * w, lm_fore.y * h, lm_fore.z * w], dtype=float)
        face_h_units = np.linalg.norm(fore_w - chin_w)
        upc = face_h_units / 15.0
    except:
        upc = 5.0
    
    dist_cm = config.USER_MONITOR_DISTANCE
    mon_w_cm, mon_h_cm = config.MONITOR_WIDTH_CM, config.MONITOR_HEIGHT_CM
    half_w = (mon_w_cm * 0.5) * upc
    half_h = (mon_h_cm * 0.5) * upc

    head_forward = -R_final[:, 2]
    if forward_hint is not None:
        head_forward = forward_hint / np.linalg.norm(forward_hint)

    if gaze_origin is not None and gaze_dir is not None:
        gaze_dir = gaze_dir / np.linalg.norm(gaze_dir)
        plane_point = head_center + head_forward * (config.USER_MONITOR_DISTANCE * upc)
        plane_normal = head_forward
        denom = np.dot(plane_normal, gaze_dir)
        if abs(denom) > 1e-6:
            t = np.dot(plane_normal, plane_point - gaze_origin) / denom
            center_w = gaze_origin + t * gaze_dir
        else:
            center_w = head_center + head_forward * (config.USER_MONITOR_DISTANCE * upc)
    else:
        center_w = head_center + head_forward * (config.USER_MONITOR_DISTANCE * upc)

    world_up = np.array([0, -1, 0], dtype=float)
    head_right = np.cross(world_up, head_forward)
    head_right /= np.linalg.norm(head_right)
    head_up = np.cross(head_forward, head_right)
    head_up /= np.linalg.norm(head_up)

    p0 = center_w - head_right * half_w - head_up * half_h
    p1 = center_w + head_right * half_w - head_up * half_h
    p2 = center_w + head_right * half_w + head_up * half_h
    p3 = center_w - head_right * half_w + head_up * half_h

    normal_w = head_forward / (np.linalg.norm(head_forward) + 1e-9)
    
    return [p0, p1, p2, p3], center_w, normal_w, upc


def convert_gaze_to_screen_coordinates(combined_gaze_direction, 
                                       calibration_offset_yaw, 
                                       calibration_offset_pitch):
    """
    시선 벡터를 화면 좌표로 변환
    """
    reference_forward = np.array([0, 0, -1])
    avg_direction = combined_gaze_direction / np.linalg.norm(combined_gaze_direction)
    
    print(f"🟣[GazeTracking] [DEBUG] avg_direction (original): {avg_direction}")
    
    # Yaw 계산
    xz_proj = np.array([avg_direction[0], 0, avg_direction[2]])
    xz_norm = np.linalg.norm(xz_proj)
    if xz_norm < 1e-9:
        yaw_rad = 0.0
    else:
        xz_proj /= xz_norm
        yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz_proj), -1.0, 1.0))
        if avg_direction[0] < 0:
            yaw_rad = -yaw_rad
    
    # Pitch 계산
    yz_proj = np.array([0, avg_direction[1], avg_direction[2]])
    yz_norm = np.linalg.norm(yz_proj)
    if yz_norm < 1e-9:
        pitch_rad = 0.0
    else:
        yz_proj /= yz_norm
        pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz_proj), -1.0, 1.0))
        if avg_direction[1] > 0:
            pitch_rad = -pitch_rad
    
    yaw_deg = math.degrees(yaw_rad)
    pitch_deg = math.degrees(pitch_rad)
    
    print(f"🟣[GazeTracking] [DEBUG] Before flip - yaw: {yaw_deg:.2f}, pitch: {pitch_deg:.2f}")
    
    # ❌ 기존의 잘못된 좌우 반전 제거
    if yaw_deg < 0:
        yaw_deg = -yaw_deg
    elif yaw_deg > 0:
        yaw_deg = -yaw_deg
    
    # ✅ 올바른 좌우 반전 (필요한 경우만)
    # yaw_deg = -yaw_deg  # 한 번만 반전
    
    raw_yaw_deg = yaw_deg
    raw_pitch_deg = pitch_deg
    
    # 보정 적용
    yaw_deg += calibration_offset_yaw
    pitch_deg += calibration_offset_pitch
    
    print(f"🟣[GazeTracking] [DEBUG] After offset - yaw: {yaw_deg:.2f}, pitch: {pitch_deg:.2f}")
    print(f"🟣[GazeTracking] [DEBUG] offset_yaw: {calibration_offset_yaw:.2f}, offset_pitch: {calibration_offset_pitch:.2f}")
    
    # 화면 좌표로 변환
    # ✅ 범위 제한 추가 (극단값 방지)
    yaw_deg = np.clip(yaw_deg, -config.YAW_DEGREES, config.YAW_DEGREES)
    pitch_deg = np.clip(pitch_deg, -config.PITCH_DEGREES, config.PITCH_DEGREES)
    
    screen_x = int(((yaw_deg + config.YAW_DEGREES) / (2 * config.YAW_DEGREES)) * config.MONITOR_WIDTH_PX)
    screen_y = int(((config.PITCH_DEGREES - pitch_deg) / (2 * config.PITCH_DEGREES)) * config.MONITOR_HEIGHT_PX)
    
    print(f"🟣[GazeTracking] [DEBUG] Before flip - screen_x: {screen_x}, screen_y: {screen_y}")
    
    # 좌우반전 (필요한 경우)
    screen_x = config.MONITOR_WIDTH_PX - screen_x
    
    # 클리핑
    screen_x = max(10, min(screen_x, config.MONITOR_WIDTH_PX - 10))
    screen_y = max(10, min(screen_y, config.MONITOR_HEIGHT_PX - 10))
    
    return screen_x, screen_y, raw_yaw_deg, raw_pitch_deg