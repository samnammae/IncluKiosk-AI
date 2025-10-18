import math
import numpy as np
import config_test

# =========================
# 소규모 수학 유틸
# =========================
def _rot_x(a):
    """X축 회전 행렬(라디안)"""
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0],
                     [0, ca, -sa],
                     [0, sa,  ca]], dtype=float)

def _rot_y(a):
    """Y축 회전 행렬(라디안)"""
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ ca, 0, sa],
                     [  0, 1,  0],
                     [-sa, 0, ca]], dtype=float)

def _normalize(v):
    """벡터 정규화(영벡터 방지)"""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def _focal_px(width, fov_deg):
    """수평 FOV 기준 핀홀 모델의 초점거리(픽셀) 계산"""
    # horizontal pinhole focal length
    return 0.5 * width / math.tan(math.radians(fov_deg) * 0.5)


# =========================
# 스케일 추정: 선택된 3D 점들의 평균 쌍거리
# =========================
def compute_scale(points_3d):
    # Use average pairwise distance for robustness
    """
    선택된 3D 점들의 평균 쌍(pairwise) 거리로 스케일을 추정.
    코 주변 밀집 영역의 크기 변화 → 거리 변화 보정(원근에 따른 눈 구체 반경 스케일링 등)
    """
    n = len(points_3d)
    total = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(points_3d[i] - points_3d[j])
            total += dist
            count += 1
    return total / count if count > 0 else 1.0


# =========================
# 모니터 평면 구성(월드 공간)
# =========================
def create_monitor_plane(head_center, R_final, face_landmarks, w, h, forward_hint=None, gaze_origin=None, gaze_dir=None):
    """
    Monitor is oriented horizontally like a real monitor (top edge parallel to global X-axis).
    """
    """
    (간단 모델) 얼굴 앞 약 50cm 위치에 60x40cm 모니터 평면을 생성.
    - upc(units_per_cm): 턱(152)↔이마(10) 높이로 대략적인 월드 단위/센티미터 스케일 추정
    - head_forward: 머리 정면 방향(-R_final[:,2])
    - forward_hint, gaze_ray를 제공하면 그 교차 위치 근처에 평면 중심을 더 정밀하게 위치
    - 코너 p0(좌상), p1(우상), p2(우하), p3(좌하) 순으로 반환
    """
    # 1) 얼굴 높이(턱<->이마)로 스케일 추정(실패 시 기본값)
    try:
        lm_chin = face_landmarks[152]
        lm_fore = face_landmarks[10]
        chin_w = np.array([lm_chin.x * w,  lm_chin.y * h,  lm_chin.z * w], dtype=float)
        fore_w = np.array([lm_fore.x * w,  lm_fore.y * h,  lm_fore.z * w], dtype=float)
        face_h_units = np.linalg.norm(fore_w - chin_w)
        upc = face_h_units / config_test.DEFAULT_FACE_LENGTH  # (경험치) 얼굴 높이 15cm 가정
    except Exception:
        upc = 5.0
    
    # 2) 물리 크기/거리(간단 가정)
    dist_cm = config_test.USER_MONITOR_DISTANCE
    
    mon_w_cm, mon_h_cm = config_test.MONITOR_WIDTH_CM, config_test.MONITOR_HEIGHT_CM
    half_w = (mon_w_cm * 0.5) * upc
    half_h = (mon_h_cm * 0.5) * upc

    # 머리 기준 정면 벡터
    head_forward = -R_final[:, 2]
    if forward_hint is not None:
        head_forward = forward_hint / np.linalg.norm(forward_hint)

    # --- NEW: use gaze ray intersection ---
    # gaze ray 정보를 활용해 화면 중심을 더 자연스럽게 배치(가능하면)
    if gaze_origin is not None and gaze_dir is not None:
        gaze_dir = gaze_dir / np.linalg.norm(gaze_dir)

        # Place the monitor so its center is exactly at some point on the gaze ray
        # For simplicity: choose intersection at 50 cm from head_center along head_forward
        plane_point = head_center + head_forward * (dist_cm * upc)
        plane_normal = head_forward

        denom = np.dot(plane_normal, gaze_dir)
        if abs(denom) > 1e-6:
            t = np.dot(plane_normal, plane_point - gaze_origin) / denom
            center_w = gaze_origin + t * gaze_dir
        else:
            # fallback: use fixed distance
            center_w = head_center + head_forward * (dist_cm * upc)
    else:
        # fallback: original placement
        center_w = head_center + head_forward * (dist_cm * upc)

    # Compute right/up using head orientation
    # 월드 상의 우/상 벡터 구성(머리 방향과 월드 업을 교차 이용)
    world_up = np.array([0, -1, 0], dtype=float) # 이미지 좌표 상단이 -Y
    head_right = np.cross(world_up, head_forward)
    head_right /= np.linalg.norm(head_right)
    head_up = np.cross(head_forward, head_right)
    head_up /= np.linalg.norm(head_up)

    # 모니터 평면 4코너(좌상 p0 기준, 시계방향(?))
    p0 = center_w - head_right * half_w - head_up * half_h
    p1 = center_w + head_right * half_w - head_up * half_h
    p2 = center_w + head_right * half_w + head_up * half_h
    p3 = center_w - head_right * half_w + head_up * half_h

    normal_w = head_forward / (np.linalg.norm(head_forward) + 1e-9)
    return [p0, p1, p2, p3], center_w, normal_w, upc

