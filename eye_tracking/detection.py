"""
Detection Module
Face, Hand detection 및 관련 유틸리티 함수
"""
import cv2
import numpy as np
import mediapipe as mp
from types import SimpleNamespace
from . import config


# MediaPipe 초기화
def init_mediapipe():
    """MediaPipe 모델 초기화"""
    # Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=config.FACE_DETECTION_MODEL,
        min_detection_confidence=config.FACE_DETECTION_CONFIDENCE
    )
    
    # Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=config.FACEMESH_MAX_FACES,
        refine_landmarks=True,
        min_detection_confidence=config.FACEMESH_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.FACEMESH_MIN_TRACKING_CONFIDENCE
    )
    
    # Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        model_complexity=config.HANDS_MODEL_COMPLEXITY,
        max_num_hands=config.HANDS_MAX_NUM,
        min_detection_confidence=config.HANDS_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.HANDS_MIN_TRACKING_CONFIDENCE
    )
    
    return face_detection, face_mesh, hands


# =====================
# Face Detection
# =====================
def mediapipe_face_detect(face_detection, frame_bgr):
    """MediaPipe로 얼굴 검출"""
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)
    
    bboxes = []
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            score = detection.score[0]
            x0 = int(bbox.xmin * w)
            y0 = int(bbox.ymin * h)
            x1 = int((bbox.xmin + bbox.width) * w)
            y1 = int((bbox.ymin + bbox.height) * h)
            x0 = max(0, min(x0, w-1))
            y0 = max(0, min(y0, h-1))
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            if x1 > x0 and y1 > y0:
                bboxes.append((x0, y0, x1, y1, score))
    
    return bboxes


def expand_and_clip_bbox(bbox, margin, w, h):
    """BBox 확장 및 클리핑"""
    x0, y0, x1, y1 = bbox
    bw = x1 - x0
    bh = y1 - y0
    mx = int(bw * margin)
    my = int(bh * margin)
    x0 = max(0, x0 - mx)
    y0 = max(0, y0 - my)
    x1 = min(w-1, x1 + mx)
    y1 = min(h-1, y1 + my)
    return x0, y0, x1, y1


def to_global_landmarks(landmarks, roi, full_w, full_h):
    """ROI 좌표를 전체 프레임 좌표로 변환"""
    x0, y0, x1, y1 = roi
    cw = x1 - x0
    ch = y1 - y0
    out = []
    for lm in landmarks:
        gx = (x0 + lm.x * cw) / float(full_w)
        gy = (y0 + lm.y * ch) / float(full_h)
        gz = lm.z
        out.append(SimpleNamespace(x=gx, y=gy, z=gz))
    return out


# =====================
# Hand Detection
# =====================
def _lm_xy(hand_landmarks, idx, w, h):
    """랜드마크 XY 좌표 추출"""
    lm = hand_landmarks.landmark[idx]
    return np.array([lm.x * w, lm.y * h], dtype=float)


def is_finger_curled(hand_landmarks, tip_idx, pip_idx, wrist_idx, w, h):
    """손가락이 구부러졌는지 확인"""
    tip = _lm_xy(hand_landmarks, tip_idx, w, h)
    pip = _lm_xy(hand_landmarks, pip_idx, w, h)
    wrist = _lm_xy(hand_landmarks, wrist_idx, w, h)
    return np.linalg.norm(tip - wrist) < np.linalg.norm(pip - wrist)


def is_thumb_curled(hand_landmarks, w, h):
    """엄지가 구부러졌는지 확인"""
    wrist = _lm_xy(hand_landmarks, 0, w, h)
    tip = _lm_xy(hand_landmarks, 4, w, h)
    mcp = _lm_xy(hand_landmarks, 2, w, h)
    return np.linalg.norm(tip - wrist) < np.linalg.norm(mcp - wrist)


def is_fist(hand_landmarks, w, h):
    """주먹 제스처 감지"""
    curled = 0
    curled += int(is_finger_curled(hand_landmarks, 8, 6, 0, w, h))   # 검지
    curled += int(is_finger_curled(hand_landmarks, 12, 10, 0, w, h)) # 중지
    curled += int(is_finger_curled(hand_landmarks, 16, 14, 0, w, h)) # 약지
    curled += int(is_finger_curled(hand_landmarks, 20, 18, 0, w, h)) # 소지
    curled += int(is_thumb_curled(hand_landmarks, w, h))              # 엄지
    return curled >= 5 # 5개가 다 구부러져야 주먹으로 감지