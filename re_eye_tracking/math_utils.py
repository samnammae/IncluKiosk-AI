import math
import numpy as np

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