"""
Detection Test Script
Face/Hand detection 테스트용 스크립트
"""
import cv2
import time
from .detection import init_mediapipe, mediapipe_face_detect, is_fist
from . import config


def test_face_detection():
    """Face detection 테스트"""
    print("=== Face Detection Test ===")
    print("Press 'q' to quit")
    
    # Initialize
    face_detection, face_mesh, hands = init_mediapipe()
    
    # Camera
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
    
    print(f"Camera opened: {cap.isOpened()}")
    print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    frame_count = 0
    last_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        frame_count += 1
        
        # Face detection every 4 frames
        if frame_count % 4 == 0:
            bboxes = mediapipe_face_detect(face_detection, frame)
            
            # Draw bboxes
            for bbox in bboxes:
                x0, y0, x1, y1, score = bbox
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(frame, f"{score:.2f}", (x0, y0-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # FPS
        current_time = time.time()
        fps = 1.0 / (current_time - last_time) if (current_time - last_time) > 0 else 0
        last_time = current_time
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Face Detection Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test completed")


def test_hand_detection():
    """Hand detection 테스트"""
    print("=== Hand Detection Test ===")
    print("Make a fist to test detection")
    print("Press 'q' to quit")
    
    # Initialize
    face_detection, face_mesh, hands = init_mediapipe()
    
    # Camera
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Camera opened: {cap.isOpened()}")
    print(f"Resolution: {w}x{h}")
    
    frame_count = 0
    last_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        frame_count += 1
        
        # Hand detection every 4 frames
        if frame_count % 4 == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(frame_rgb)
            
            fist_detected = False
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    if is_fist(hand_landmarks, w, h):
                        fist_detected = True
                        cv2.putText(frame, "FIST DETECTED!", (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    
                    # Draw hand landmarks
                    import mediapipe as mp
                    mp_drawing = mp.solutions.drawing_utils
                    mp_hands = mp.solutions.hands
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
        
        # FPS
        current_time = time.time()
        fps = 1.0 / (current_time - last_time) if (current_time - last_time) > 0 else 0
        last_time = current_time
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Hand Detection Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test completed")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m eye_tracking.test_detection face    # Test face detection")
        print("  python -m eye_tracking.test_detection hand    # Test hand detection")
        sys.exit(1)
    
    test_type = sys.argv[1].lower()
    
    if test_type == "face":
        test_face_detection()
    elif test_type == "hand":
        test_hand_detection()
    else:
        print(f"Unknown test type: {test_type}")
        print("Available tests: face, hand")