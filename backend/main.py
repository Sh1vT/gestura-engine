# python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# Import necessary libraries
import asyncio
import cv2
import numpy as np
import uvicorn
import mediapipe as mp
import json
import os
import time
from dotenv import load_dotenv
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from fastapi import FastAPI, WebSocket
from google import genai
from typing import List
from pydantic import BaseModel

# NLP class using pydantic and typing for auto-type checks
class NLPRequest(BaseModel):
    words: List[str]

# Load environment variables
def load_environment_variables():
    load_dotenv("secret.env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise Exception("API key not set.")
    return api_key

# Initialize Mediapipe
def initialize_mediapipe():
    return mp.solutions.hands.Hands(
        min_detection_confidence=0.7, min_tracking_confidence=0.7
    )

# Load gesture data
def load_gesture_data(file_path):
    if os.path.exists(file_path):
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
        print(gesture_dict)
        return gesture_dict
    return {}

# Normalize landmarks
def normalize_landmarks(landmarks):
    min_x, min_y, _ = np.min(landmarks, axis=0)
    max_x, max_y, _ = np.max(landmarks, axis=0)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    centered_landmarks = landmarks - np.array([center_x, center_y, 0])
    scale = max(max_x - min_x, max_y - min_y)
    return centered_landmarks / scale if scale > 0 else centered_landmarks

# Handle frame processing
async def process_frame(img, state, hands, gesture_dict, frame_sequence, keypoints_to_check, cooldown_time, stage_timeout, websocket):
    current_time = time.time()

    # Enforce cooldown
    if current_time - state["last_detection_time"] < cooldown_time:
        return

    # Handle stage timeout
    if state["frame_stage"] > 0 and state["stage_start_time"] is not None:
        if current_time - state["stage_start_time"] > stage_timeout:
            reset_user_state(state)

    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = np.array(
                [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark], dtype=np.float32
            )
            if landmarks.shape[0] < max(keypoints_to_check):
                continue

            normalized_landmarks = normalize_landmarks(landmarks)
            normalized_keypoints = normalized_landmarks[keypoints_to_check]
            await handle_gesture_matching(
                normalized_keypoints, state, gesture_dict, frame_sequence, current_time, websocket
            )

# Handle gesture matching
async def handle_gesture_matching(normalized_keypoints, state, gesture_dict, frame_sequence, current_time, websocket):
    if state["frame_stage"] == 0:
        for gesture_name, frames in gesture_dict.items():
            keyframe_points = frames["start"].reshape(-1, 3)
            if keyframe_points.shape != normalized_keypoints.shape:
                print("normalize check failed")
                continue
            distance, _ = fastdtw(keyframe_points, normalized_keypoints, dist=euclidean)
            if distance < 0.9:
                state["pending_gesture"] = gesture_name
                state["frame_stage"] = 1
                state["stage_start_time"] = current_time
                print(f"🟢 'Start' frame matched for '{gesture_name}'!")
                ###
                break
    elif state["pending_gesture"]:
        stage_name = frame_sequence[state["frame_stage"]]
        keyframe_points = gesture_dict[state["pending_gesture"]][stage_name].reshape(-1, 3)
        if keyframe_points.shape == normalized_keypoints.shape:
            distance, _ = fastdtw(keyframe_points, normalized_keypoints, dist=euclidean)
            if distance < 0.9:
                print(f"🟢 '{stage_name.capitalize()}' frame matched for '{state['pending_gesture']}'!")
                state["frame_stage"] += 1
                state["stage_start_time"] = current_time
                if state["frame_stage"] == 3:
                    if(len(state["detected_words"])==0 or state["detected_words"][-1].split('_')[0]!=state["pending_gesture"].split('_')[0]):
                        state["detected_words"].append(state["pending_gesture"].split('_')[0])
                    # print(state["detected_words"])
                    await websocket.send_text(json.dumps(state["detected_words"]))
                    # print(state["detected_words"])
                    print(f"✅ Detected Sign: {state['pending_gesture']}")
                    state["last_detection_time"] = current_time
                    reset_user_state(state)

# Reset user state
def reset_user_state(state):
    state["pending_gesture"] = None
    state["frame_stage"] = 0
    state["stage_start_time"] = None

# Instantiation of app and global variables
app = FastAPI()
client = genai.Client(api_key=load_environment_variables())
user_states = {}

# Configuration constants
COOLDOWN_TIME = 0.5  # Time (in seconds) to wait between detections
STAGE_TIMEOUT = 5  # Timeout for each stage of gesture matching
frame_sequence = ["start", "mid1", "mid2", "end"]  # 4-frame stages
keypoints_to_check =  [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20]  # Key landmarks to track
hands = initialize_mediapipe()
gesture_dict = load_gesture_data("gestures.json")

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    user_states[websocket] = {
        "pending_gesture": None,
        "frame_stage": 0,
        "stage_start_time": None,
        "last_detection_time": 0,
        "fc": 0,
        "detected_words": []
    }
    print("WebSocket connection established")

    try:
        while True:
            data = await asyncio.wait_for(websocket.receive_bytes(), timeout=2.0)
            user_state = user_states[websocket]
            user_state["fc"] += 1
            np_arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is not None and user_state["fc"] % 3 == 0:
                # and user_state["fc"] % 3 == 0
                print(f"processing frame {user_state['fc']} for {websocket.client}")
                await process_frame(
                    img, user_state, hands, gesture_dict, frame_sequence, keypoints_to_check, COOLDOWN_TIME, STAGE_TIMEOUT, websocket
                )
    except Exception as e:
        print(f"WebSocket connection error: {e}")
    finally:
        await websocket.close()
        print("WebSocket connection closed.")

@app.post("/nlp_process")
def nlp_processed_text(request: NLPRequest):
    prompt = "This is an API call. Generate a meaningful sentence from these words in beginning. If some sign is named as sign-name_some-number then you can ignore everything after the underscore. Only respond with the sentence and nothign else. So something like return_5 is just return etc. Try to make the sentence sound casua\ as well. The words start now: "+" ".join(request.words)
    print(prompt)
    response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[prompt]
    )
    generated_sentence = response.text
    print(generated_sentence)
    return {"sentence": generated_sentence}

# Main application
if __name__ == "__main__":
    GEMINI_API_KEY = load_environment_variables()
    hands = initialize_mediapipe()
    gesture_dict = load_gesture_data("gestures.json")
    uvicorn.run(app, host="0.0.0.0", port=8000)
