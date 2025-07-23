import cv2
import numpy as np
import mediapipe as mp
import json
import sys
import os
from app.utils import normalize_landmarks

GESTURE_FILE = "data/gestures.json"
keypoints_to_check = [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20]

if len(sys.argv) < 2:
    print("Usage: python record_sign.py <sign_name>")
    sys.exit(1)

sign_name = sys.argv[1]

# Load existing gestures
if os.path.exists(GESTURE_FILE):
    with open(GESTURE_FILE, "r") as f:
        gesture_dict = json.load(f)
        gesture_dict = {
            k: {
                "start": np.array(v["start"], dtype=np.float32),
                "mid1": np.array(v["mid1"], dtype=np.float32),
                "mid2": np.array(v["mid2"], dtype=np.float32),
                "end": np.array(v["end"], dtype=np.float32),
            }
            for k, v in gesture_dict.items()
        }
else:
    gesture_dict = {}

if sign_name in gesture_dict:
    print(f"Sign '{sign_name}' already exists.")
    sys.exit(1)

frames = {}
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

for frame_name in ["start", "mid1", "mid2", "end"]:
    print(f"Get ready to record '{frame_name}' frame for '{sign_name}'. Press SPACE to capture.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        display = frame.copy()
        cv2.putText(display, f"{sign_name} - {frame_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Record Frame", display)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = np.array(
                        [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark], dtype=np.float32
                    )
                    normalized = normalize_landmarks(landmarks)
                    frames[frame_name] = normalized[keypoints_to_check]
                    print(f"Captured '{frame_name}' frame.")
                    break
                else:
                    print("No hand detected. Try again.")
                    continue
                break
            else:
                print("No hand detected. Try again.")
    cv2.destroyWindow("Record Frame")
cap.release()
cv2.destroyAllWindows()
gesture_dict[sign_name] = frames
# Save as lists for JSON
serializable = {
    k: {frame: v[frame].tolist() for frame in v} for k, v in gesture_dict.items()
}
with open(GESTURE_FILE, "w") as f:
    json.dump(serializable, f, indent=2)
print(f"Sign '{sign_name}' added successfully.") 