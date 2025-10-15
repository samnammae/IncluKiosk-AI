"""객체 감지 유틸리티 함수"""
import collections
import numpy as np
import cv2

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])


class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """바운딩 박스"""
    __slots__ = ()

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin

    @property
    def area(self):
        return self.width * self.height

    @property
    def valid(self):
        return self.width >= 0 and self.height >= 0

    def scale(self, sx, sy):
        return BBox(
            xmin=sx * self.xmin,
            ymin=sy * self.ymin,
            xmax=sx * self.xmax,
            ymax=sy * self.ymax
        )

    def translate(self, dx, dy):
        return BBox(
            xmin=dx + self.xmin,
            ymin=dy + self.ymin,
            xmax=dx + self.xmax,
            ymax=dy + self.ymax
        )

    def map(self, f):
        return BBox(
            xmin=f(self.xmin),
            ymin=f(self.ymin),
            xmax=f(self.xmax),
            ymax=f(self.ymax)
        )

    @staticmethod
    def intersect(a, b):
        return BBox(
            xmin=max(a.xmin, b.xmin),
            ymin=max(a.ymin, b.ymin),
            xmax=min(a.xmax, b.xmax),
            ymax=min(a.ymax, b.ymax)
        )

    @staticmethod
    def union(a, b):
        return BBox(
            xmin=min(a.xmin, b.xmin),
            ymin=min(a.ymin, b.ymin),
            xmax=max(a.xmax, b.xmax),
            ymax=max(a.ymax, b.ymax)
        )

    @staticmethod
    def iou(a, b):
        intersection = BBox.intersect(a, b)
        if not intersection.valid:
            return 0.0
        area = intersection.area
        return area / (a.area + b.area - area)


def input_size(interpreter):
    """입력 이미지 크기 반환 (width, height)"""
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    return width, height


def input_tensor(interpreter):
    """입력 텐서 반환 (height, width, 3)"""
    tensor_index = interpreter.get_input_details()[0]['index']
    return interpreter.tensor(tensor_index)()[0]


def output_tensor(interpreter, i):
    """출력 텐서 반환"""
    tensor = interpreter.tensor(interpreter.get_output_details()[i]['index'])()
    return np.squeeze(tensor)


def get_output(interpreter, score_threshold, image_scale=(1.0, 1.0)):
    """감지된 객체 리스트 반환"""
    boxes = output_tensor(interpreter, 0)
    class_ids = output_tensor(interpreter, 1)
    scores = output_tensor(interpreter, 2)
    count = int(output_tensor(interpreter, 3))

    width, height = input_size(interpreter)
    image_scale_x, image_scale_y = image_scale
    sx, sy = width / image_scale_x, height / image_scale_y

    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(class_ids[i]),
            score=float(scores[i]),
            bbox=BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax).scale(sx, sy).map(int)
        )

    return [make(i) for i in range(count) if scores[i] >= score_threshold]


def detect_faces(interpreter, frame_bgr, score_threshold=0.5, debug=False):
    """
    얼굴 감지
    
    Returns:
        list: [(xmin, ymin, xmax, ymax, score), ...] 정규화된 좌표 (0~1)
    """
    if frame_bgr is None or frame_bgr.size == 0:
        if debug:
            print("[DEBUG] detect_faces: 빈 프레임")
        return []
    
    H, W = frame_bgr.shape[:2]
    if H <= 0 or W <= 0:
        if debug:
            print(f"[DEBUG] detect_faces: 잘못된 크기 {W}x{H}")
        return []
    
    try:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        width, height = input_size(interpreter)
        resized = cv2.resize(frame_rgb, (width, height))
        
        tensor = input_tensor(interpreter)
        tensor.fill(0)
        tensor[:, :] = resized.copy()
        del tensor
        
        interpreter.invoke()
        objs = get_output(interpreter, score_threshold, (1.0, 1.0))
        
        if debug:
            print(f"[DEBUG] detect_faces: {len(objs)} 얼굴 검출")
        
        faces = []
        for obj in objs:
            bbox = obj.bbox
            # 정규화된 좌표로 변환 (0~1)
            xmin = max(0.0, min(1.0, bbox.xmin / width))
            ymin = max(0.0, min(1.0, bbox.ymin / height))
            xmax = max(0.0, min(1.0, bbox.xmax / width))
            ymax = max(0.0, min(1.0, bbox.ymax / height))
            faces.append((xmin, ymin, xmax, ymax, obj.score))
            
            if debug:
                print(f"  Face: score={obj.score:.2f}, "
                      f"bbox=({xmin:.2f},{ymin:.2f},{xmax:.2f},{ymax:.2f})")
        
        return faces
        
    except Exception as e:
        print(f"[Face Detection Error] {e}")
        import traceback
        traceback.print_exc()
        return []


def detect_person(interpreter, frame_bgr, score_threshold=0.4, debug=False):
    """
    사람 감지 (COCO 클래스 ID 0)
    
    Returns:
        tuple or None: (xmin, ymin, xmax, ymax) 정규화된 좌표, 또는 None
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    
    H, W = frame_bgr.shape[:2]
    if H <= 0 or W <= 0:
        return None
    
    try:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        width, height = input_size(interpreter)
        resized = cv2.resize(frame_rgb, (width, height))
        
        tensor = input_tensor(interpreter)
        tensor.fill(0)
        tensor[:, :] = resized.copy()
        del tensor
        
        interpreter.invoke()
        objs = get_output(interpreter, score_threshold, (1.0, 1.0))
        
        best = None
        best_area = -1.0
        person_count = 0
        
        # COCO 데이터셋에서 사람은 클래스 ID 0
        for obj in objs:
            if obj.id != 0:
                continue
            
            person_count += 1
            bbox = obj.bbox
            
            # 정규화된 좌표로 변환
            xmin = max(0.0, min(1.0, bbox.xmin / width))
            ymin = max(0.0, min(1.0, bbox.ymin / height))
            xmax = max(0.0, min(1.0, bbox.xmax / width))
            ymax = max(0.0, min(1.0, bbox.ymax / height))
            
            area = (xmax - xmin) * (ymax - ymin)
            if area > best_area:
                best_area = area
                best = (xmin, ymin, xmax, ymax)
        
        if debug:
            print(f"[DEBUG] detect_person: {person_count} 사람 검출")
            if best:
                print(f"  Best person: bbox=({best[0]:.2f},{best[1]:.2f},"
                      f"{best[2]:.2f},{best[3]:.2f})")
        
        return best
        
    except Exception as e:
        print(f"[Person Detection Error] {e}")
        import traceback
        traceback.print_exc()
        return None