import numpy as np
import os
import json

def normalize_landmarks(landmarks):
    min_x, min_y, _ = np.min(landmarks, axis=0)
    max_x, max_y, _ = np.max(landmarks, axis=0)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    centered_landmarks = landmarks - np.array([center_x, center_y, 0])
    scale = max(max_x - min_x, max_y - min_y)
    return centered_landmarks / scale if scale > 0 else centered_landmarks

def load_gesture_data(file_path):
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r") as f:
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
    return gesture_dict

def save_gesture_data(file_path, gesture_dict):
    # Convert numpy arrays to lists for JSON serialization
    serializable = {
        k: {frame: v[frame].tolist() for frame in v} for k, v in gesture_dict.items()
    }
    with open(file_path, "w") as f:
        json.dump(serializable, f, indent=2)

def load_actions(file_path):
    import os, json
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return []

def save_actions(file_path, actions):
    import json
    with open(file_path, "w") as f:
        json.dump(actions, f, indent=2)

def load_mappings(file_path):
    import os, json
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return []

def save_mappings(file_path, mappings):
    import json
    with open(file_path, "w") as f:
        json.dump(mappings, f, indent=2) 