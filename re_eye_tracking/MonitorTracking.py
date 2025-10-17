import cv2
import numpy as np
import mediapipe as mp
import time
import math
from scipy.spatial.transform import Rotation as Rscipy
from collections import deque
import pyautogui
import threading
import keyboard

import config
import math_utils
import world

pyautogui.PAUSE = 0           # ← 기본 0.1초 대기 제거
pyautogui.FAILSAFE = False    # ← 선택: 좌상단 구석에 가면 예외나는 기본 안전장치 비활성화

# Screen and mouse control setup (from old script)

mouse_control_enabled = False       # 마우스 제어 토글 플래그(F7로 on/off). True일 때 보조 스레드가 mouse_target으로 커서를 이동
filter_length = 10                  # 시선 벡터 스무딩 버퍼 길이(최근 N개 평균)
gaze_length = 350                   # 2D 프레임 내에서 시선 가시화(디버그) 선 길이(픽셀)

# =========================
# 3D 디버그(오빗 카메라) 뷰 상태값
# =========================
# yaw/pitch는 라디안, radius는 피벗(피사체)로부터의 거리
orbit_yaw   = -151.0          # 좌우 회전(라디안)
orbit_pitch = 00.0          # 상하 회전(라디안)
orbit_radius = 1500.0       # 카메라-피벗 거리(월드 단위)
orbit_fov_deg = 50.0       # 수평 FOV(도). 투영시 초점거리 계산에 사용

# 디버그 월드 고정(화면 중앙 캘리브레이션 이후 모니터 중심을 피벗으로 고정)
debug_world_frozen = False
orbit_pivot_frozen = None  # 고정할 피벗(모니터 중심)

# 모니터 평면상 마커 저장용(사각형 로컬 좌표 a,b : 0..1)
# a = 0..1 across width (p0->p1), b = 0..1 down height (p0->p3)     # a: 좌→우(p0->p1), b: 상→하(p0->p3)
gaze_markers = []

# =========================
# 3D 모니터 평면 상태(월드 좌표계)
# =========================
monitor_corners = None   # 모니터 모서리 [p0,p1,p2,p3], 시계/반시계(?) 순 회전으로 가정
monitor_center_w = None  # 평면 중심(월드)
monitor_normal_w = None  # 평면 법선(월드)
units_per_cm = None      # 월드 단위/센티미터(대략적 스케일)

# 마우스 목표 좌표(보조 스레드와 공유) + 동기화 락
mouse_target = [config.CENTER_X, config.CENTER_Y]
mouse_lock = threading.Lock()

# Calibration offsets for screen mapping
# 화면 매핑 보정값(캘리브레이션 시 's'로 중앙 정렬하기 위해 yaw/pitch 오프셋 저장)
calibration_offset_yaw = 0
calibration_offset_pitch = 0

# 간단한 화면 가장자리 캘리브레이션 단계(현재 코드에서는 미사용 플래그)
# 0=중앙 대기, 1=좌측 가장자리 대기, 2=완료
# calib_step = 0

# Buffers to store recent gaze data for smoothing
# 최근 결합 시선(combined gaze) 방향 벡터 버퍼(평활화용)
combined_gaze_directions = deque(maxlen=filter_length)

# reference matrices to fix coordinate flipping issue
# These help keep the axes consistent from frame to frame by stabilizing eigenvector directions
# ===== 고정 참조 회전행렬 컨테이너 =====
# PCA 고유벡터 기반 회전에서 프레임 사이 부호 뒤집힘(sign flip)을 막기 위한 참조
R_ref_nose = [None]
R_ref_forehead = [None]
calibration_nose_scale = None


# =========================
# MediaPipe FaceMesh 초기화
# =========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# =========================
# 카메라 열기
# =========================
cap = cv2.VideoCapture(config.CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 안전장치: 최대값 제한
MAX_DIMENSION = 4096  # 안전한 최대값
if w > MAX_DIMENSION or h > MAX_DIMENSION or w <= 0 or h <= 0:
    print(f"[WARNING] Invalid camera resolution: {w}x{h}, using defaults")
    w, h = 640, 480

print(f"[Camera] Resolution set to: {w}x{h}")

# === Nose-only landmark indices (for stable up/down eye sphere tracking) ===
# These landmarks are near the nose and are less affected by lateral head movement
# =========================
# 코 주변 안정 영역(시선 추정에서 상하 흔들림 억제용 샘플 인덱스)
# =========================
# MediaPipe FaceMesh의 468+ 랜드마크 중 코/미간/비근 주변 인덱스들
nose_indices = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241, 
                461, 125, 354, 218, 438, 195, 167, 393, 165, 391,
                3, 248]

# =========================
# 화면 좌표를 파일로 기록
# =========================
# screen_position_file = config.SCREEN_POSITION
# def write_screen_position(x, y):
#     """
#     현재 화면 좌표를 텍스트 파일 1줄로 덮어쓰기 저장.
#     다른 프로세스가 이 파일을 폴링하여 시선 기반 입력과 연동할 수 있음.
#     """
#     with open(screen_position_file, 'w') as f:
#         f.write(f"{x},{y}\n")


# =========================
# 오빗 카메라 키 입력 처리(j/l/i/k/[/]/r)
# =========================
def update_orbit_from_keys():
    """
    키보드 입력으로 오빗 카메라 파라미터를 미세 조정.
    j/l: yaw, i/k: pitch, [/]: 줌, r: 리셋
    키가 눌리는 프레임마다 현재 상태를 프린트.
    """
    global orbit_yaw, orbit_pitch, orbit_radius
    yaw_step   = math.radians(1.5)
    pitch_step = math.radians(1.5)
    zoom_step  = 12.0

    changed = False

    # 회전
    if keyboard.is_pressed('j'):  # yaw left
        orbit_yaw -= yaw_step; changed = True
    if keyboard.is_pressed('l'):  # yaw right
        orbit_yaw += yaw_step; changed = True
    if keyboard.is_pressed('i'):  # pitch up
        orbit_pitch += pitch_step; changed = True
    if keyboard.is_pressed('k'):  # pitch down
        orbit_pitch -= pitch_step; changed = True

    # 확대/축소
    if keyboard.is_pressed('['):  # zoom out
        orbit_radius += zoom_step; changed = True
    if keyboard.is_pressed(']'):  # zoom in
        orbit_radius = max(80.0, orbit_radius - zoom_step); changed = True

    # 리셋 (prints every frame while held)
    if keyboard.is_pressed('r'):
        orbit_yaw = 0.0
        orbit_pitch = 0.0
        orbit_radius = 600.0
        changed = True

    # Clamp pitch & radius
    orbit_pitch = max(math.radians(-89), min(math.radians(89), orbit_pitch))
    orbit_radius = max(80.0, orbit_radius)

    if changed:
        print(f"[Orbit Debug] yaw={math.degrees(orbit_yaw):.2f}°, "
              f"pitch={math.degrees(orbit_pitch):.2f}°, "
              f"radius={orbit_radius:.2f}, "
              f"fov={orbit_fov_deg:.1f}°")

# =========================
# 2D 프레임에 시선(eye_center→iris) 가시화
# =========================
def draw_gaze(frame, eye_center, iris_center, eye_radius, color, gaze_length):
    """지정한 눈 중심에서 홍채 방향으로 gaze_length만큼 선을 그려 시각화.
    프레임 좌표계에서 선분을 두 부분(홍채 뒤/앞)으로 나눠 간접적인 깊이 느낌 제공.
    """
    # 단위 시선 벡터 // Gaze vector
    gaze_direction = iris_center - eye_center
    gaze_direction /= np.linalg.norm(gaze_direction)
    gaze_endpoint = eye_center + gaze_direction * gaze_length

    # 전체 시선 벡터(얇은 선)
    cv2.line(frame, tuple(int(v) for v in eye_center[:2]), tuple(int(v) for v in gaze_endpoint[:2]), color, 2)

    # 홍채 중심쪽으로 약간 앞/뒤 분할 지점 // Segment points
    iris_offset = eye_center + gaze_direction * (1.2 * eye_radius)

    # ---- PART 1: back segment (behind iris) ---- (뒤쪽) 분할선
    cv2.line(
        frame,
        (int(eye_center[0]), int(eye_center[1])),
        (int(iris_offset[0]), int(iris_offset[1])),
        color,
        1
    )

    # ---- IRIS (occludes part of the ray) ---- 
    up_dir = np.array([0, -1, 0])
    right_dir = np.cross(gaze_direction, up_dir)
    if np.linalg.norm(right_dir) < 1e-6:
        right_dir = np.array([1, 0, 0])
    up_dir = np.cross(right_dir, gaze_direction)
    up_dir /= np.linalg.norm(up_dir)
    right_dir /= np.linalg.norm(right_dir)
    ellipse_axes = (
        int((eye_radius / 3) * np.linalg.norm(right_dir[:2])),
        int((eye_radius / 3) * np.linalg.norm(up_dir[:2]))
    )
    angle = math.degrees(math.atan2(gaze_direction[1], gaze_direction[0]))

    # ---- PART 2: front segment (on top of iris) ---- (앞쪽) 분할선
    cv2.line(
        frame,
        (int(iris_offset[0]), int(iris_offset[1])),
        (int(gaze_endpoint[0]), int(gaze_endpoint[1])),
        color,
        1
    )
    
    

# =========================
# 와이어프레임 큐브(머리 로컬 좌표축 시각화용)
# =========================

def draw_wireframe_cube(frame, center, R, size=80):
    # 얼굴에 네모박스 그리는 거
    """중심점(center)와 회전행렬 R이 주어졌을 때, 그 좌표축에 정렬된 큐브를 2D 프레임에 그림.
    X=R[:,0], Y=-R[:,1], Z=-R[:,2] 방향으로 면 확장.
    """
    right = R[:, 0]
    up = -R[:, 1]
    forward = -R[:, 2]

    hw, hh, hd = size * 1, size * 1, size * 1

    def corner(x_sign, y_sign, z_sign):
        return (center +
                x_sign * hw * right +
                y_sign * hh * up +
                z_sign * hd * forward)

    # 8개 큐브의 코너
    corners = [corner(x, y, z) for x in [-1, 1] for y in [1, -1] for z in [-1, 1]]
    projected = [(int(pt[0]), int(pt[1])) for pt in corners]

    # 간선으로 코너 연결
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for i, j in edges:
        cv2.line(frame, projected[i], projected[j], (255, 128, 0), 2)

# =========================
# 코 주변 소영역의 PCA 좌표계 계산 및 그리기
# =========================
def compute_and_draw_coordinate_box(frame, face_landmarks, indices, ref_matrix_container, color=(0, 255, 0), size=80):
    """선택된 랜드마크(indices)의 3D 좌표로 공분산→고유분해(PCA)하여 주축(eigvecs) 획득.
    오른손 좌표계를 강제(det<0이면 축 하나 부호 반전), 이후 참조 회전행렬과 비교하여
    축 부호를 안정화한 R_final을 산출. 2D 프레임 위에 큐브와 XYZ축을 그린 뒤,
    중심점(center), R_final, 사용된 3D 점들을 반환.
    """
    # Extract 3D positions of selected landmarks
    # 선택된 랜드마크의 3D 위치(픽셀 단위 스케일: x*w, y*h, z*w)
    points_3d = np.array([
        [face_landmarks[i].x * w, face_landmarks[i].y * h, face_landmarks[i].z * w]
        for i in indices
    ])

    # Compute the average position as the center of this substructure
    center = np.mean(points_3d, axis=0) # 중심(평균)

    # 선택 점들을 2D 프레임에 표시(디버그)
    for i in indices:
        x, y = int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)
        cv2.circle(frame, (x, y), 3, color, -1)

    # PCA-based orientation: Compute eigenvectors of the covariance matrix
    # 공분산 → 고유분해(내림차순 정렬)
    centered = points_3d - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:, np.argsort(-eigvals)]  # Sort by descending eigenvalue (major axes)

    # Ensure the orientation matrix is right-handed
    # 오른손 좌표계 강제(det<0이면 마지막 축 부호 반전)
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1

    # Convert to Euler angles and re-construct rotation matrix (optional but clarifies the transform)
    # (선택) 오일러 변환 거쳐 재구성 — 큰 의미는 없지만 회전표현을 명시적으로 만드는 과정
    r = Rscipy.from_matrix(eigvecs)
    roll, pitch, yaw = r.as_euler('zyx', degrees=False)
    yaw *= 1
    roll *= 1
    R_final = Rscipy.from_euler('zyx', [roll, pitch, yaw]).as_matrix()

    # === Stabilize rotation with reference matrix to avoid flipping during eigenvector sign change ===
    # 참조 행렬과 축 정렬(연속 프레임 간 축 부호 뒤집힘 방지)
    if ref_matrix_container[0] is None:
        ref_matrix_container[0] = R_final.copy()
    else:
        R_ref = ref_matrix_container[0]
        for i in range(3):
            if np.dot(R_final[:, i], R_ref[:, i]) < 0:
                R_final[:, i] *= -1

    # Draw cube and orientation axes on the image
    draw_wireframe_cube(frame, center, R_final, size)

    # # 큐브 및 XYZ축(초록=X, 파랑=Y, 빨강=Z) 시각화
    axis_length = size * 1.2
    axis_dirs = [R_final[:, 0], -R_final[:, 1], -R_final[:, 2]]
    axis_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

    for i in range(3):
        end_pt = center + axis_dirs[i] * axis_length
        cv2.line(frame, (int(center[0]), int(center[1])), (int(end_pt[0]), int(end_pt[1])), axis_colors[i], 2)

    return center, R_final, points_3d

# =========================
# 시선 벡터 → 화면 좌표 변환(각도 기반 간단 매핑)
# =========================
def convert_gaze_to_screen_coordinates(combined_gaze_direction, calibration_offset_yaw, calibration_offset_pitch):
    """
    This function is adapted from the old script's vector-to-screen mapping logic
    """
    """결합 3D 시선 벡터(월드 기준)를 yaw/pitch 각도로 투영 후 화면 해상도로 선형 매핑.
    - yawDegrees/pitchDegrees 범위 내에서 모니터 좌표로 변환
    - 's' 키로 중앙 보정(calibration_offset_*)을 적용
    반환: (screen_x, screen_y, raw_yaw_deg, raw_pitch_deg)
    """
    # Reference forward direction (camera looking straight ahead)
    # 기준 정면 방향(카메라에서 화면 안쪽 -Z)
    reference_forward = np.array([0, 0, -1])  # Z-axis into the screen

    # Normalize the gaze direction // 단위화
    avg_direction = combined_gaze_direction / np.linalg.norm(combined_gaze_direction)

    # Horizontal (yaw) angle from reference (project onto XZ plane)
    # 수평(yaw): XZ 평면 투영
    xz_proj = np.array([avg_direction[0], 0, avg_direction[2]])
    xz_proj /= np.linalg.norm(xz_proj)
    yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz_proj), -1.0, 1.0))
    if avg_direction[0] < 0:
        yaw_rad = -yaw_rad  # 왼쪽을 음수

    # Vertical (pitch) angle from reference (project onto YZ plane)
    # 수직(pitch): YZ 평면 투영
    yz_proj = np.array([0, avg_direction[1], avg_direction[2]])
    yz_proj /= np.linalg.norm(yz_proj)
    pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz_proj), -1.0, 1.0))
    if avg_direction[1] > 0:
        pitch_rad = -pitch_rad  # 위쪽을 양수

    # Convert to degrees and re-center around 0
    # 도 단위
    yaw_deg = np.degrees(yaw_rad)
    pitch_deg = np.degrees(pitch_rad)

    # Convert left rotations to 0-180 (from old script logic)
    # 왼쪽(-)을 양수로 뒤집는 보정(기존 스크립트 호환)
    if yaw_deg < 0:
        yaw_deg = -(yaw_deg)
    elif yaw_deg > 0:
        yaw_deg = - yaw_deg

    #yaw is now converted to -90 (looking directly left) to +90 (looking directly right), wrt camera
    #pitch is now converted to +90 (looking straight up) and -90 (looking straight down), wrt camera
    
    raw_yaw_deg = yaw_deg
    raw_pitch_deg = pitch_deg

    
    # Specify degrees at which screen border will be reached
    # 화면 경계에 해당하는 각도 범위(경험값)
    yawDegrees = 5 * 3  # x degrees left or right # 좌우 한계(도)
    pitchDegrees = 2.0 * 2.5  # x degrees up or down # 상하 한계(도)

    # Apply calibration offsets
    # 중앙 보정 적용('s' 키로 저장된 오프셋)
    yaw_deg += calibration_offset_yaw
    pitch_deg += calibration_offset_pitch
    
    # Map to full screen resolution
    # 해상도로 선형 매핑
    screen_x = int(((yaw_deg + yawDegrees) / (2 * yawDegrees)) * config.MONITOR_WIDTH)
    screen_y = int(((pitchDegrees - pitch_deg) / (2 * pitchDegrees)) * config.MONITOR_HEIGHT)

    # Clamp screen position to monitor bounds
    # 모니터 가장자리 밖으로 나가지 않도록 클램프(여백 10px)
    screen_x = max(10, min(screen_x, config.MONITOR_WIDTH - 10))
    screen_y = max(10, min(screen_y, config.MONITOR_HEIGHT - 10))

    return screen_x, screen_y, raw_yaw_deg, raw_pitch_deg

# =========================
# 3D 오빗 디버그 뷰 렌더링(별도 창)
# =========================
def render_debug_view_orbit(
    h, w,
    head_center3d=None,
    sphere_world_l=None, scaled_radius_l=None,
    sphere_world_r=None, scaled_radius_r=None,
    iris3d_l=None, iris3d_r=None,
    left_locked=False, right_locked=False,
    landmarks3d=None,
    combined_dir=None,
    gaze_len=430,
    monitor_corners=None,
    monitor_center=None,
    monitor_normal=None,
    gaze_markers=None,
):
    """월드 공간 요소(머리 중심/눈 구체/모니터 평면/시선 등)를 가상 카메라 시점으로 투영하여
    별도 창("Head/Eye Debug")에 렌더링. 오빗 키로 시점 제어.
    """
    # 안전장치: 디버그 뷰 크기 제한
    MAX_DEBUG_SIZE = 1920
    if h > MAX_DEBUG_SIZE or w > MAX_DEBUG_SIZE:
        # 비율 유지하며 축소
        scale = min(MAX_DEBUG_SIZE / h, MAX_DEBUG_SIZE / w)
        h = int(h * scale)
        w = int(w * scale)
        print(f"[Debug View] Resized to: {w}x{h}")
    
    if head_center3d is None:
        return

    debug = np.zeros((h, w, 3), dtype=np.uint8)

    # --- Choose orbit pivot ---
    # (1) 피벗 선택: 캘리브레이션 후에는 모니터 중심을 고정 피벗으로 사용
    head_w = np.asarray(head_center3d, dtype=float)

    # NEW: if we've frozen the world, orbit around the frozen pivot (monitor center at calib)
    # frozen the world <- 이게 calibration을 했다는거임?
    global debug_world_frozen, orbit_pivot_frozen
    if debug_world_frozen and orbit_pivot_frozen is not None:
        pivot_w = np.asarray(orbit_pivot_frozen, dtype=float)
    else:
        if monitor_center is not None:
            pivot_w = (head_w + np.asarray(monitor_center, dtype=float)) * 0.5
        else:
            pivot_w = head_w

    # --- Camera pose (orbit around pivot_w) ---
    # (2) 카메라 포즈(피벗을 향해 orbit)
    f_px = math_utils._focal_px(w, orbit_fov_deg)
    cam_offset = math_utils._rot_y(orbit_yaw) @ (math_utils._rot_x(orbit_pitch) @ np.array([0.0, 0.0, orbit_radius]))
    cam_pos = pivot_w + cam_offset

    up_world = np.array([0.0, -1.0, 0.0])   # image-space up is -Y
    fwd = math_utils._normalize(pivot_w - cam_pos)     # look at pivot // 카메라가 바라보는 방향
    right = math_utils._normalize(np.cross(fwd, up_world))
    up = math_utils._normalize(np.cross(right, fwd))
    V = np.stack([right, up, fwd], axis=0)  # 월드→카메라 좌표 변환

    def project_point(P):
        """월드 점 P를 카메라 평면으로 투영하여 2D 픽셀 좌표 반환(None=뒤쪽)"""
        Pw = np.asarray(P, dtype=float)
        Pc = V @ (Pw - cam_pos)
        if Pc[2] <= 1e-3:
            return None
        x = f_px * (Pc[0] / Pc[2]) + w * 0.5
        y = -f_px * (Pc[1] / Pc[2]) + h * 0.5
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        return (int(x), int(y)), Pc[2]

    # 간단 도우미들(폴리곤/십자/화살표)
    # def draw_poly_3d(pts, color=(0, 200, 255), thickness=2):
    #     projs = [project_point(p) for p in pts]
    #     if any(p is None for p in projs): return
    #     p2 = [p[0] for p in projs]
    #     for a, b in zip(p2, p2[1:] + [p2[0]]):
    #         cv2.line(debug, a, b, color, thickness) <- 사용 안됨

    def draw_cross_3d(P, size=12, color=(255, 0, 255), thickness=2):
        res = project_point(P)
        if res is None: return
        (x, y), _ = res
        cv2.line(debug, (x - size, y), (x + size, y), color, thickness)
        cv2.line(debug, (x, y - size), (x, y + size), color, thickness)

    def draw_arrow_3d(P0, P1, color=(0, 200, 255), thickness=3):
        a = project_point(P0); b = project_point(P1)
        if a is None or b is None: return
        p0, p1 = a[0], b[0]
        cv2.line(debug, p0, p1, color, thickness)
        v = np.array([p1[0]-p0[0], p1[1]-p0[1]], dtype=float)
        n = np.linalg.norm(v)
        if n > 1e-3:
            v /= n
            l = np.array([-v[1], v[0]])
            ah = 10
            a1 = (int(p1[0] - v[0]*ah + l[0]*ah*0.6), int(p1[1] - v[1]*ah + l[1]*ah*0.6))
            a2 = (int(p1[0] - v[0]*ah - l[0]*ah*0.6), int(p1[1] - v[1]*ah - l[1]*ah*0.6))
            cv2.line(debug, p1, a1, color, thickness)
            cv2.line(debug, p1, a2, color, thickness)

    # 랜드마크 점들
    if landmarks3d is not None:
        for P in landmarks3d:
            res = project_point(P)
            if res is not None:
                cv2.circle(debug, res[0], 0, (200, 200, 200), -1)

    # 머리 중심 시각화
    draw_cross_3d(head_w, size=12, color=(255, 0, 255), thickness=2)
    hc2d = project_point(head_w)
    if hc2d is not None:
        cv2.putText(debug, "Head Center", (hc2d[0][0] + 12, hc2d[0][1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

    # --- Pivot visual (small cross + line to head and monitor) ---
    # 피벗 표시(머리/모니터 중심을 연결)
    draw_cross_3d(pivot_w, size=8, color=(180, 120, 255), thickness=2)
    if monitor_center is not None:
        mc2d = project_point(monitor_center)
        pv2d = project_point(pivot_w)
        if mc2d is not None and pv2d is not None and hc2d is not None:
            cv2.line(debug, pv2d[0], hc2d[0], (160, 100, 255), 1)
            cv2.line(debug, pv2d[0], mc2d[0], (160, 100, 255), 1)

    # --- Eyes + per-eye gaze (unchanged from your version) ---
    # 눈 구체/홍채 시각화 및 양안 결합 시선 표시
    left_dir = None
    right_dir = None

    if left_locked and sphere_world_l is not None:
        res = project_point(sphere_world_l)
        if res is not None:
            (cx, cy), z = res
            r_px = max(2, int((scaled_radius_l if scaled_radius_l else 6) * f_px / max(z, 1e-3)))
            cv2.circle(debug, (cx, cy), r_px, (255, 255, 25), 1)
            if iris3d_l is not None:
                left_dir = np.asarray(iris3d_l) - np.asarray(sphere_world_l)
                p1 = project_point(np.asarray(sphere_world_l) + math_utils._normalize(left_dir) * gaze_len)
                if p1 is not None:
                    cv2.line(debug, (cx, cy), p1[0], (155, 155, 25), 1)
    elif iris3d_l is not None:
        res = project_point(iris3d_l)
        if res is not None:
            cv2.circle(debug, res[0], 2, (255, 255, 25), 1)

    if right_locked and sphere_world_r is not None:
        res = project_point(sphere_world_r)
        if res is not None:
            (cx, cy), z = res
            r_px = max(2, int((scaled_radius_r if scaled_radius_r else 6) * f_px / max(z, 1e-3)))
            cv2.circle(debug, (cx, cy), r_px, (25, 255, 255), 1)
            if iris3d_r is not None:
                right_dir = np.asarray(iris3d_r) - np.asarray(sphere_world_r)
                p1 = project_point(np.asarray(sphere_world_r) + math_utils._normalize(right_dir) * gaze_len)
                if p1 is not None:
                    cv2.line(debug, (cx, cy), p1[0], (25, 155, 155), 1)
    elif iris3d_r is not None:
        res = project_point(iris3d_r)
        if res is not None:
            cv2.circle(debug, res[0], 2, (25, 255, 255), 1)

    if left_locked and right_locked and sphere_world_l is not None and sphere_world_r is not None:
        origin_mid = (np.asarray(sphere_world_l) + np.asarray(sphere_world_r)) / 2.0
        if combined_dir is None and (left_dir is not None or right_dir is not None):
            parts = []
            if left_dir is not None:  parts.append(math_utils._normalize(left_dir))
            if right_dir is not None: parts.append(math_utils._normalize(right_dir))
            if parts:
                combined_dir = math_utils._normalize(np.mean(parts, axis=0))
        if combined_dir is not None:
            p0 = project_point(origin_mid)
            p1 = project_point(origin_mid + math_utils._normalize(combined_dir) * (gaze_len * 1.2))
            if p0 is not None and p1 is not None:
                cv2.line(debug, p0[0], p1[0], (155, 200, 10), 2)

    # 모니터 평면/법선/대각선 시각화
    if monitor_corners is not None:
        def draw_poly(points, color, thickness):
            projs = [project_point(p) for p in points]
            if any(p is None for p in projs): return
            p2 = [p[0] for p in projs]
            for a, b in zip(p2, p2[1:] + [p2[0]]):
                cv2.line(debug, a, b, color, thickness)
        draw_poly(monitor_corners, (0, 200, 255), 2)
        draw_poly([monitor_corners[0], monitor_corners[2]], (0, 150, 210), 1)
        draw_poly([monitor_corners[1], monitor_corners[3]], (0, 150, 210), 1)
        if monitor_center is not None:
            draw_cross_3d(monitor_center, size=8, color=(0, 200, 255), thickness=2)
            if monitor_normal is not None:
                tip = np.asarray(monitor_center) + np.asarray(monitor_normal) * (20.0 * (units_per_cm or 1.0))
                draw_arrow_3d(monitor_center, tip, color=(0, 220, 255), thickness=2)

    # --- Stored gaze markers on the monitor plane (green circles) ---
    # 저장된 (a,b) 마커를 모니터 평면에 표시(투영)
    if (gaze_markers and monitor_corners is not None):
        p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in monitor_corners]
        u = p1 - p0  # width direction // 가로
        v = p3 - p0  # height direction // 세로
        width_world = float(np.linalg.norm(u))
        if width_world > 1e-9:
            u_hat = u / width_world
            r_world = 0.01 * width_world  # 2% of width // 원 반지름(모니터 폭의 1%)
            for (a, b) in gaze_markers:
                Pm = p0 + a * u + b * v
                projP = project_point(Pm)
                projR = project_point(Pm + u_hat * r_world)
                if projP is not None and projR is not None:
                    center_px = projP[0]
                    r_px = int(max(1, np.linalg.norm(np.array(projR[0]) - np.array(center_px))))
                    cv2.circle(debug, center_px, r_px, (0, 255, 0), 1, lineType=cv2.LINE_AA)



    # --- Gaze hit on monitor plane (circle at intersection) ---
    # 결합 시선과 모니터 평면의 교차점 강조(원)
    if (monitor_corners is not None and monitor_center is not None and monitor_normal is not None
        and combined_dir is not None
        and sphere_world_l is not None and sphere_world_r is not None):

        # Ray: origin at midpoint between eyes; direction = combined gaze
        O = (np.asarray(sphere_world_l, dtype=float) + np.asarray(sphere_world_r, dtype=float)) * 0.5
        D = math_utils._normalize(np.asarray(combined_dir, dtype=float))

        # Plane: through monitor_center with normal = monitor_normal
        C = np.asarray(monitor_center, dtype=float)
        N = math_utils._normalize(np.asarray(monitor_normal, dtype=float))

        denom = float(np.dot(N, D))
        if abs(denom) > 1e-6:
            t = float(np.dot(N, (C - O)) / denom)
            if t > 0.0:
                P = O + t * D  # world-space intersection point

                # Inside-quad test using monitor's local axes (top-left p0, top-right p1, bottom-left p3)
                p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in monitor_corners]
                u = p1 - p0             # horizontal (width) vector
                v = p3 - p0             # vertical (height) vector
                wv = P  - p0

                u_len2 = float(np.dot(u, u))
                v_len2 = float(np.dot(v, v))
                if u_len2 > 1e-9 and v_len2 > 1e-9:
                    a = float(np.dot(wv, u) / u_len2)  # 0..1 across width
                    b = float(np.dot(wv, v) / v_len2)  # 0..1 across height

                    if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
                        # Project center to pixels
                        projP = project_point(P)
                        if projP is not None:
                            center_px = projP[0]

                            # Circle radius = 5% of monitor width (world), projected to pixels
                            width_world = math.sqrt(u_len2)
                            r_world = 0.05 * width_world
                            u_hat = u / max(width_world, 1e-9)

                            projR = project_point(P + u_hat * r_world)
                            if projR is not None:
                                r_px = int(max(1, np.linalg.norm(np.array(projR[0]) - np.array(center_px))))
                                cv2.circle(debug, center_px, r_px, (0, 255, 255), 2, lineType=cv2.LINE_AA)

    cv2.imshow("Head/Eye Debug", debug)

# =========================
# 마우스 이동 보조 스레드(토글 시 mouse_target으로 이동)
# =========================
def mouse_mover():
    """Mouse movement thread from old script"""
    """mouse_control_enabled가 True일 때, 10ms 간격으로 mouse_target 위치로 커서를 이동."""
    while True:
        if mouse_control_enabled:
            with mouse_lock:
                x, y = mouse_target
            pyautogui.moveTo(x, y, _pause=False)
        time.sleep(0.016)  # adjust for responsiveness

# Start mouse movement thread
# 데몬 스레드 시작
threading.Thread(target=mouse_mover, daemon=True).start()

# Eye sphere tracking variables (from new script)
# =========================
# 눈 구체(eye sphere) 추정/고정 관련 상태값
# =========================
left_sphere_locked = False
left_sphere_local_offset = None
left_calibration_nose_scale = None

right_sphere_locked = False
right_sphere_local_offset = None
right_calibration_nose_scale = None

# =========================
# 메인 루프: 프레임 처리
# =========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    combined_dir = None  # will be filled once you compute a smoothed direction // 현재 프레임의 결합 시선(평활화 후)

    # MediaPipe는 RGB 입력 요구
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark

        # 미디어 파이프 iris 모델 양안 홍채 중심 인덱스
        left_iris_idx = 468
        right_iris_idx = 473
        left_iris = face_landmarks[left_iris_idx]
        right_iris = face_landmarks[right_iris_idx]

        # Compute and draw stabilized coordinate frame from nose region
        # 코 주변 영역의 PCA 좌표계 및 중심/회전 획득 + 디버그 큐브/축 그리기
        head_center, R_final, nose_points_3d = compute_and_draw_coordinate_box(
            frame,
            face_landmarks,
            nose_indices,
            R_ref_nose,
            color=(0, 255, 0),
            size=80
        )

        # TODO compute this radius using canthus during calibration
        # 눈 구체 반경(기준) — 실제로는 canthus(안각) 등으로 보정하는 것이 좋음
        base_radius = 20  # radius at calibration distance

        # (좌안) 홍채 위치 표시(잠금 전에는 홍채만, 잠금 후엔 구체를 그림)
        x_iris_l = int(left_iris.x * w)
        y_iris_l = int(left_iris.y * h)
        # === LEFT EYE visualization ===
        if not left_sphere_locked:
            cv2.circle(frame, (x_iris_l, y_iris_l), 10, (255, 25, 25), 2)
        else:
            current_nose_scale = math_utils.compute_scale(nose_points_3d)
            scale_ratio = current_nose_scale / left_calibration_nose_scale if left_calibration_nose_scale else 1.0
            scaled_offset = left_sphere_local_offset * scale_ratio
            sphere_world_l = head_center + R_final @ scaled_offset
            x_sphere_l, y_sphere_l = int(sphere_world_l[0]), int(sphere_world_l[1])
            scaled_radius_l = int(base_radius * scale_ratio)
            cv2.circle(frame, (x_sphere_l, y_sphere_l), scaled_radius_l, (255, 255, 25), 2)

        # (우안)
        x_iris_r = int(right_iris.x * w)
        y_iris_r = int(right_iris.y * h)
        # === RIGHT EYE visualization ===
        if not right_sphere_locked:
            cv2.circle(frame, (x_iris_r, y_iris_r), 10, (25, 255, 25), 2)
        else:
            current_nose_scale = math_utils.compute_scale(nose_points_3d)
            scale_ratio_r = current_nose_scale / right_calibration_nose_scale if right_calibration_nose_scale else 1.0
            scaled_offset_r = right_sphere_local_offset * scale_ratio_r
            sphere_world_r = head_center + R_final @ scaled_offset_r
            x_sphere_r, y_sphere_r = int(sphere_world_r[0]), int(sphere_world_r[1])
            scaled_radius_r = int(base_radius * scale_ratio_r)
            cv2.circle(frame, (x_sphere_r, y_sphere_r), scaled_radius_r, (25, 255, 255), 2)

        # 홍채의 3D 위치(픽셀 스케일)
        iris_3d_left = np.array([left_iris.x * w, left_iris.y * h, left_iris.z * w])
        iris_3d_right = np.array([right_iris.x * w, right_iris.y * h, right_iris.z * w])

        # 양안 구체가 잠겼다면 시선 계산/시각화 및 화면 좌표 변환 수행
        if left_sphere_locked and right_sphere_locked:
            # ==== DRAW LEFT AND RIGHT GAZE ====
            # # (디버그) 프레임 위 시선 선 그리기
            draw_gaze(frame, sphere_world_l, iris_3d_left, scaled_radius_l, (55, 255, 0), 130)   
            draw_gaze(frame, sphere_world_r, iris_3d_right, scaled_radius_r, (55, 255, 0), 130)  

            # ==== COMPUTE COMBINED GAZE DIRECTION FOR SCREEN MAPPING ====
            # Calculate individual gaze directions
            # (1) 개별 시선 벡터 → (2) 평균 → (3) 정규화
            left_gaze_dir = iris_3d_left - sphere_world_l
            left_gaze_dir /= np.linalg.norm(left_gaze_dir)
            
            right_gaze_dir = iris_3d_right - sphere_world_r
            right_gaze_dir /= np.linalg.norm(right_gaze_dir)
            
            # Combine gaze directions (average)
            # 최근 N개 방향 평균으로 평활화
            raw_combined_direction = (left_gaze_dir + right_gaze_dir) / 2
            raw_combined_direction /= np.linalg.norm(raw_combined_direction)

            # Update direction buffer for smoothing
            combined_gaze_directions.append(raw_combined_direction)

            # Smoothed direction
            avg_combined_direction = np.mean(combined_gaze_directions, axis=0)
            avg_combined_direction /= np.linalg.norm(avg_combined_direction)

            combined_dir = avg_combined_direction

            # ==== CONVERT GAZE TO SCREEN COORDINATES ====
            # 화면 좌표로 변환(중앙 보정 포함)
            screen_x, screen_y, raw_yaw, raw_pitch = convert_gaze_to_screen_coordinates(
                avg_combined_direction, 
                calibration_offset_yaw, 
                calibration_offset_pitch
            )

            # 마우스 이동 목표 업데이트(스레드가 이동 수행)
            if mouse_control_enabled:
                with mouse_lock:
                    mouse_target[0] = screen_x
                    mouse_target[1] = screen_y

            # 좌표 파일에 마우스 위치 입력 (screen_position.txt)
            # write_screen_position(screen_x, screen_y)

            # Draw combined gaze ray for visualization
            # (디버그) 결합 시선 벡터 선분 그리기
            combined_origin = (sphere_world_l + sphere_world_r) / 2
            combined_target = combined_origin + avg_combined_direction * gaze_length
            cv2.line(
                frame,
                (int(combined_origin[0]), int(combined_origin[1])),
                (int(combined_target[0]), int(combined_target[1])),
                (255, 255, 10), 3
            )

            # 상단 중앙에 텍스트 표시(화면 좌표)
            texts = [
                f"Screen: ({screen_x}, {screen_y})",
                #f"Mouse: {'ON' if mouse_control_enabled else 'OFF'}"
            ]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            line_spacing = 30

            for i, text in enumerate(texts):
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                center_x = (w - text_width) // 2
                #center_y = (h // 2) + (i - len(texts)//2) * line_spacing
                
                color = (0, 255, 0) if "Mouse: ON" not in text else (0, 255, 0) if mouse_control_enabled else (0, 0, 255)
                cv2.putText(frame, text, (center_x, 30), font, font_scale, color, thickness)

        
        # 모든 랜드마크를 흰 점으로 표현(밀집 표시)
        for idx, lm in enumerate(face_landmarks):
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 0, (255, 255, 255), -1)

        # 오빗 카메라 입력 업데이트
        update_orbit_from_keys()

        # Build 3D landmarks in your existing scale (x*w, y*h, z*w)
        # 3D 디버그 뷰에 사용할 전체 랜드마크(월드 스케일) 구성
        landmarks3d = None
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            landmarks3d = np.array([[p.x * w, p.y * h, p.z * w] for p in lm], dtype=float)

        # 디버그 뷰 렌더링(별도 창)
        render_debug_view_orbit(
            h, w,
            head_center3d=head_center if 'head_center' in locals() else None,
            sphere_world_l=sphere_world_l if left_sphere_locked and 'sphere_world_l' in locals() else None,
            scaled_radius_l=scaled_radius_l if left_sphere_locked and 'scaled_radius_l' in locals() else None,
            sphere_world_r=sphere_world_r if right_sphere_locked and 'sphere_world_r' in locals() else None,
            scaled_radius_r=scaled_radius_r if right_sphere_locked and 'scaled_radius_r' in locals() else None,
            iris3d_l=iris_3d_left if 'iris_3d_left' in locals() else None,
            iris3d_r=iris_3d_right if 'iris_3d_right' in locals() else None,
            left_locked=left_sphere_locked,
            right_locked=right_sphere_locked,
            landmarks3d=landmarks3d,
            combined_dir=avg_combined_direction if 'avg_combined_direction' in locals() else None,
            gaze_len=5230, # 디버그 시선 길이(과장(?))
            monitor_corners=monitor_corners,
            monitor_center=monitor_center_w,
            monitor_normal=monitor_normal_w,
            gaze_markers=gaze_markers
        )

    # 메인 2D 뷰 갱신
    cv2.imshow("Integrated Eye Tracking", frame)

    # -------------------------
    # 키보드 입력 처리(전역)
    # -------------------------
    # F7: 마우스 제어 토글(디바운싱)
    if keyboard.is_pressed('f7'):
        mouse_control_enabled = not mouse_control_enabled
        print(f"[Mouse Control] {'Enabled' if mouse_control_enabled else 'Disabled'}")
        time.sleep(0.3)  # debounce to prevent rapid toggling

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('c') and not (left_sphere_locked and right_sphere_locked):
        # 1) 현 프레임의 코 영역 스케일 측정
        current_nose_scale = math_utils.compute_scale(nose_points_3d)
        
        # 2) (좌안) 홍채의 머리 로컬 오프셋을 계산하고 구체 중심을 앞(z+)으로 base_radius만큼 이동
        left_sphere_local_offset = R_final.T @ (iris_3d_left - head_center)
        camera_dir_world = np.array([0, 0, 1])                      # 카메라 z+ 방향(프레임 전방)
        camera_dir_local = R_final.T @ camera_dir_world             # 머리 로컬로 변환
        left_sphere_local_offset += base_radius * camera_dir_local
        left_calibration_nose_scale = current_nose_scale
        left_sphere_locked = True # Lock LEFT eye

        # 3) (우안) 동일 로직
        right_sphere_local_offset = R_final.T @ (iris_3d_right - head_center)
        right_sphere_local_offset += base_radius * camera_dir_local  # use same camera_dir_local
        right_calibration_nose_scale = current_nose_scale
        right_sphere_locked = True # Lock RIGHT eye

        # 4) 캘리브레이션 시점의 월드 좌표 구체 중심(스케일 1 가정)
        # === Create 3D monitor plane at calibration ===
        # Compute instantaneous sphere positions at calibration distance (scale=1)
        sphere_world_l_calib = head_center + R_final @ left_sphere_local_offset
        sphere_world_r_calib = head_center + R_final @ right_sphere_local_offset

        # 5) 양안 시선 평균으로 정면 힌트(forward_hint) 계산
        # Estimate a forward gaze direction from the two eyes
        left_dir  = iris_3d_left  - sphere_world_l_calib
        right_dir = iris_3d_right - sphere_world_r_calib
        # Normalize (guard zero)
        if np.linalg.norm(left_dir)  > 1e-9: left_dir  /= np.linalg.norm(left_dir)
        if np.linalg.norm(right_dir) > 1e-9: right_dir /= np.linalg.norm(right_dir)
        forward_hint = (left_dir + right_dir) * 0.5
        if np.linalg.norm(forward_hint) > 1e-9:
            forward_hint /= np.linalg.norm(forward_hint)
        else:
            forward_hint = None  # fallback to head frame

        gaze_origin = (sphere_world_l_calib + sphere_world_r_calib) / 2
        gaze_dir = forward_hint  # already normalized
        
        # 6) 모니터 평면 생성 + 디버그 월드 고정(모니터 중심을 피벗으로)
        monitor_corners, monitor_center_w, monitor_normal_w, units_per_cm = world.create_monitor_plane(
            head_center, R_final, face_landmarks, w, h,
            forward_hint=forward_hint,
            gaze_origin=gaze_origin,
            gaze_dir=gaze_dir
        )

        # Freeze the debug world's orbit pivot at the calibrated monitor center
        #global debug_world_frozen, orbit_pivot_frozen
        debug_world_frozen = True
        orbit_pivot_frozen = monitor_center_w.copy()
        print("[Debug View] World pivot frozen at monitor center.")

        print(f"[Monitor] units_per_cm={units_per_cm:.3f}, center={monitor_center_w}, normal={monitor_normal_w}")


        print("[Both Spheres Locked] Eye sphere calibration complete.")
    elif key == ord('s') and left_sphere_locked and right_sphere_locked:
        # 화면 중앙 캘리브레이션: 현재 시선을 (0,0) 기준으로 간주하여 오프셋 저장
        # Screen calibration - user should look at center of screen when pressing 's'
        # Get current gaze direction
        left_gaze_dir = iris_3d_left - sphere_world_l
        left_gaze_dir /= np.linalg.norm(left_gaze_dir)
        right_gaze_dir = iris_3d_right - sphere_world_r
        right_gaze_dir /= np.linalg.norm(right_gaze_dir)
        current_combined_direction = (left_gaze_dir + right_gaze_dir) / 2
        current_combined_direction /= np.linalg.norm(current_combined_direction)
        
        # Calculate what the raw angles would be without calibration
        _, _, raw_yaw, raw_pitch = convert_gaze_to_screen_coordinates(
            current_combined_direction, 0, 0  # no calibration offset
        )
        
        # Set calibration offsets to center the gaze
        calibration_offset_yaw = 0 - raw_yaw
        calibration_offset_pitch = 0 - raw_pitch
        
        print(f"[Screen Calibrated] Offset Yaw: {calibration_offset_yaw:.2f}, Offset Pitch: {calibration_offset_pitch:.2f}")
    elif key == ord('x'):
        # 현재 결합 시선과 모니터 평면의 교차 지점을 (a,b)로 변환하여 마커 저장
        # Drop a marker at the current gaze∩monitor point
        if (monitor_corners is not None and monitor_center_w is not None and monitor_normal_w is not None
            and left_sphere_locked and right_sphere_locked):
            # Recompute current eye-sphere positions (scale-aware)
            current_nose_scale = math_utils.compute_scale(nose_points_3d)
            scale_ratio_l = current_nose_scale / left_calibration_nose_scale if left_calibration_nose_scale else 1.0
            scale_ratio_r = current_nose_scale / right_calibration_nose_scale if right_calibration_nose_scale else 1.0
            sphere_world_l_now = head_center + R_final @ (left_sphere_local_offset * scale_ratio_l)
            sphere_world_r_now = head_center + R_final @ (right_sphere_local_offset * scale_ratio_r)

            # Combined gaze direction (use smoothed if available; otherwise instantaneous)
            if 'avg_combined_direction' in locals() and avg_combined_direction is not None:
                D = math_utils._normalize(np.asarray(avg_combined_direction, dtype=float))
            else:
                lg = iris_3d_left  - sphere_world_l_now
                rg = iris_3d_right - sphere_world_r_now
                if np.linalg.norm(lg) < 1e-9 or np.linalg.norm(rg) < 1e-9:
                    print("[Marker] Gaze direction invalid; try again.")
                    D = None
                else:
                    lg /= np.linalg.norm(lg)
                    rg /= np.linalg.norm(rg)
                    D = math_utils._normalize(lg + rg)

            if D is not None:
                O = (sphere_world_l_now + sphere_world_r_now) * 0.5
                C = np.asarray(monitor_center_w, dtype=float)
                N = math_utils._normalize(np.asarray(monitor_normal_w, dtype=float))
                denom = float(np.dot(N, D))
                if abs(denom) < 1e-6:
                    print("[Marker] Gaze ray parallel to monitor; no marker.")
                else:
                    t = float(np.dot(N, (C - O)) / denom)
                    if t <= 0.0:
                        print("[Marker] Intersection behind/at eye; no marker.")
                    else:
                        P = O + t * D  # world-space intersection
                        # Map P to monitor local (a,b), then store if inside the quad
                        p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in monitor_corners]
                        u = p1 - p0
                        v = p3 - p0
                        u_len2 = float(np.dot(u, u))
                        v_len2 = float(np.dot(v, v))
                        if u_len2 > 1e-9 and v_len2 > 1e-9:
                            wv = P - p0
                            a = float(np.dot(wv, u) / u_len2)
                            b = float(np.dot(wv, v) / v_len2)
                            if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
                                gaze_markers.append((a, b))
                                print(f"[Marker] Added at a={a:.3f}, b={b:.3f}")
                            else:
                                print("[Marker] Gaze not on monitor; no marker.")
                        else:
                            print("[Marker] Monitor dimensions degenerate; no marker.")
        else:
            print("[Marker] Monitor/gaze not ready; complete center calibration first.")


cap.release()

cv2.destroyAllWindows()
