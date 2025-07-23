import cv2
import numpy as np
import mediapipe as mp
import json
import time
import os
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from app.utils import load_gesture_data

GESTURE_FILE = "data/gestures.json"
frame_sequence = ["start", "mid1", "mid2", "end"]
keypoints_to_check = [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20]
COOLDOWN_TIME = 0.5
STAGE_TIMEOUT = 5

def normalize_landmarks(landmarks):
    min_x, min_y, _ = np.min(landmarks, axis=0)
    max_x, max_y, _ = np.max(landmarks, axis=0)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    centered_landmarks = landmarks - np.array([center_x, center_y, 0])
    scale = max(max_x - min_x, max_y - min_y)
    return centered_landmarks / scale if scale > 0 else centered_landmarks

gesture_dict = load_gesture_data(GESTURE_FILE)
pending_gesture = None
frame_stage = 0
stage_start_time = None
last_detection_time = 0
detected_words = []

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
frame_counter = 0

print("[recognize_signs.py] Subprocess started.", flush=True)
print("Press 'q' in the camera window to stop recognition.", flush=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (320, 240))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_counter += 1
    results = None
    current_time = time.time()
    if frame_counter % 3 == 0:
        results = hands.process(rgb_frame)
        if current_time - last_detection_time >= COOLDOWN_TIME:
            if frame_stage > 0 and stage_start_time is not None:
                if current_time - stage_start_time > STAGE_TIMEOUT:
                    print("Gesture stage timeout, resetting user state.", flush=True)
                    pending_gesture = None
                    frame_stage = 0
                    stage_start_time = None
            if results and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = np.array(
                        [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark], dtype=np.float32
                    )
                    if landmarks.shape[0] < max(keypoints_to_check):
                        continue
                    normalized_landmarks = normalize_landmarks(landmarks)
                    normalized_keypoints = normalized_landmarks[keypoints_to_check]
                    # Gesture matching
                    if frame_stage == 0:
                        for gesture_name, frames in gesture_dict.items():
                            keyframe_points = frames["start"].reshape(-1, 3)
                            if keyframe_points.shape != normalized_keypoints.shape:
                                continue
                            distance, _ = fastdtw(keyframe_points, normalized_keypoints, dist=euclidean)
                            if distance < 0.9:
                                pending_gesture = gesture_name
                                frame_stage = 1
                                stage_start_time = current_time
                                break
                    elif pending_gesture:
                        stage_name = frame_sequence[frame_stage]
                        keyframe_points = gesture_dict[pending_gesture][stage_name].reshape(-1, 3)
                        if keyframe_points.shape == normalized_keypoints.shape:
                            distance, _ = fastdtw(keyframe_points, normalized_keypoints, dist=euclidean)
                            if distance < 0.9:
                                frame_stage += 1
                                stage_start_time = current_time
                                if frame_stage == 3:
                                    detected_sign = pending_gesture.split('_')[0]
                                    detected_words.append(detected_sign)
                                    print(f"Detected sign: {detected_sign}", flush=True)
                                    last_detection_time = current_time
                                    pending_gesture = None
                                    frame_stage = 0
                                    stage_start_time = None
    cv2.imshow("Gestura Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print("Recognition stopped.", flush=True) 