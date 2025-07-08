import cv2
import numpy as np
import mediapipe as mp
import json
import time
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import threading
import subprocess
import queue
recognition_proc = None
recognition_queue = queue.Queue()
recognition_thread = None

# --- Gesture Logic ---
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

def normalize_landmarks(landmarks):
    min_x, min_y, _ = np.min(landmarks, axis=0)
    max_x, max_y, _ = np.max(landmarks, axis=0)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    centered_landmarks = landmarks - np.array([center_x, center_y, 0])
    scale = max(max_x - min_x, max_y - min_y)
    return centered_landmarks / scale if scale > 0 else centered_landmarks

# --- Sign Manager Logic ---
def add_sign():
    global gesture_dict
    sign_name = simpledialog.askstring("Add Sign", "Enter sign name:")
    if not sign_name:
        return
    if sign_name in gesture_dict:
        messagebox.showerror("Error", f"Sign '{sign_name}' already exists.")
        return
    # Launch record_sign.py as a subprocess
    try:
        result = subprocess.run([
            "python3", "app/record_sign.py", sign_name
        ], check=True)
        print("[Main App] Subprocess for sign recording finished.")
        import time
        import cv2
        time.sleep(1)
        cv2.destroyAllWindows()
        print("[Main App] Called cv2.destroyAllWindows() after subprocess.")
        # After subprocess, reload gesture_dict and update list
        gesture_dict = load_gesture_data(GESTURE_FILE)
        update_sign_listbox()
        messagebox.showinfo("Success", f"Sign '{sign_name}' added.")
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", f"Failed to add sign '{sign_name}'. See terminal for details.")

def delete_sign():
    selection = sign_listbox.curselection()
    if not selection:
        return
    sign_name = sign_listbox.get(selection[0])
    if messagebox.askyesno("Delete Sign", f"Are you sure you want to delete '{sign_name}'?"):
        gesture_dict.pop(sign_name, None)
        save_gesture_data(GESTURE_FILE, gesture_dict)
        update_sign_listbox()
        messagebox.showinfo("Deleted", f"Sign '{sign_name}' deleted.")

def update_sign_listbox():
    sign_listbox.delete(0, tk.END)
    for sign in sorted(gesture_dict.keys()):
        sign_listbox.insert(tk.END, sign)

# --- Recognition Logic (unchanged, but moved to a function for threading) ---
def recognition_loop():
    global running, last_detection_time, stage_start_time, camera_in_use
    if camera_in_use:
        print("[Recognition] Camera is already in use. Recognition will not start.")
        return
    camera_in_use = True
    print("[Recognition] Camera opened for recognition.")
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    frame_counter = 0
    while running:
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
                        print("Gesture stage timeout, resetting user state.")
                        reset_user_state()
                if results and results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = np.array(
                            [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark], dtype=np.float32
                        )
                        if landmarks.shape[0] < max(keypoints_to_check):
                            continue
                        normalized_landmarks = normalize_landmarks(landmarks)
                        normalized_keypoints = normalized_landmarks[keypoints_to_check]
                        handle_gesture_matching(normalized_keypoints, current_time)
        cv2.imshow("Gestura Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    print("[Recognition] Camera released after recognition.")
    cv2.destroyAllWindows()
    print("[Recognition] All OpenCV windows destroyed after recognition.")
    camera_in_use = False
    stop_recognition()

def reset_user_state():
    global pending_gesture, frame_stage, stage_start_time
    pending_gesture = None
    frame_stage = 0
    stage_start_time = None

def handle_gesture_matching(normalized_keypoints, current_time):
    global pending_gesture, frame_stage, stage_start_time, last_detection_time
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
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
                    print(f"Detected sign: {detected_sign}")
                    last_detection_time = current_time
                    reset_user_state()
                    update_sign_listbox()

def start_recognition():
    global recognition_proc, recognition_thread
    recog_start_button.config(state=tk.DISABLED)
    recog_stop_button.config(state=tk.NORMAL)
    try:
        recognition_proc = subprocess.Popen([
            "python3", "-u", "app/recognize_signs.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        print("[Main App] Recognition subprocess started.")
        # Start a thread to read stdout
        recognition_thread = threading.Thread(target=read_recognition_output, daemon=True)
        recognition_thread.start()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start recognition: {e}")
        recog_start_button.config(state=tk.NORMAL)
        recog_stop_button.config(state=tk.DISABLED)

def stop_recognition():
    global recognition_proc, recognition_thread
    if recognition_proc is not None:
        recognition_proc.terminate()
        recognition_proc.wait()
        print("[Main App] Recognition subprocess terminated.")
        recognition_proc = None
    recog_start_button.config(state=tk.NORMAL)
    recog_stop_button.config(state=tk.DISABLED)
    # Optionally, clear the detected signs listbox
    # detected_signs_listbox.delete(0, tk.END)

def read_recognition_output():
    global recognition_proc
    if recognition_proc is None or recognition_proc.stdout is None:
        return
    for line in recognition_proc.stdout:
        if line.startswith("Detected sign: ") or "timeout" in line:
            print(line.strip())
        if line.startswith("Detected sign: "):
            sign = line.strip().split(": ", 1)[1]
            recognition_queue.put(sign)

def update_detected_signs_listbox():
    detected_signs_listbox.delete(0, tk.END)
    for sign in detected_words[-6:]:
        detected_signs_listbox.insert(tk.END, sign)

def poll_recognition_queue():
    while not recognition_queue.empty():
        sign = recognition_queue.get()
        detected_words.append(sign)
        update_detected_signs_listbox()
    root.after(100, poll_recognition_queue)

# --- Globals and State ---
import os
GESTURE_FILE = "data/gestures.json"
gesture_dict = load_gesture_data(GESTURE_FILE)
frame_sequence = ["start", "mid1", "mid2", "end"]
keypoints_to_check = [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20]
detected_words = []
pending_gesture = None
frame_stage = 0
stage_start_time = None
last_detection_time = 0
COOLDOWN_TIME = 0.5
STAGE_TIMEOUT = 5
running = False

camera_in_use = False

# --- In-memory Actions and Mappings ---
actions = []  # List of dicts: {"name": ...}
mappings = []  # List of dicts: {"sign": ..., "action": ...}

def add_action():
    name = simpledialog.askstring("Add Action", "Enter action name:")
    if name:
        actions.append({"name": name})
        update_action_listbox()
        update_mapping_dropdowns()

def delete_action():
    sel = action_listbox.curselection()
    if sel:
        idx = sel[0]
        actions.pop(idx)
        update_action_listbox()
        update_mapping_dropdowns()
        update_mapping_listbox()

def update_action_listbox():
    action_listbox.delete(0, tk.END)
    for a in actions:
        action_listbox.insert(tk.END, a["name"])

def update_mapping_dropdowns():
    sign_dropdown['values'] = list(gesture_dict.keys())
    action_dropdown['values'] = [a["name"] for a in actions]
    if gesture_dict:
        sign_dropdown.current(0)
    if actions:
        action_dropdown.current(0)

def update_mapping_listbox():
    mapping_listbox.delete(0, tk.END)
    for m in mappings:
        mapping_listbox.insert(tk.END, f"{m['sign']} → {m['action']}")

def map_sign_to_action():
    sign = sign_var.get()
    action = action_var.get()
    if sign and action:
        global mappings
        mappings = [m for m in mappings if m["sign"] != sign]
        mappings.append({"sign": sign, "action": action})
        update_mapping_listbox()

def unmap_sign():
    sign = sign_var.get()
    global mappings
    mappings = [m for m in mappings if m["sign"] != sign]
    update_mapping_listbox()

# --- Tkinter UI ---
root = tk.Tk()
root.title("Gestura - ISL Interpreter")
root.geometry("600x500")

notebook = ttk.Notebook(root)
notebook.pack(fill=tk.BOTH, expand=True)

# Recognition Tab
recog_frame = ttk.Frame(notebook)
notebook.add(recog_frame, text="Recognition")

# --- Recognized Signs Section ---
recog_signs_frame = ttk.LabelFrame(recog_frame, text="Recognized Signs")
recog_signs_frame.pack(fill=tk.X, padx=20, pady=(15, 5))

detected_signs_listbox = tk.Listbox(recog_signs_frame, width=40, height=8)
detected_signs_listbox.pack(fill=tk.X, padx=10, pady=(10, 10))

recog_btn_frame = ttk.Frame(recog_signs_frame)
recog_btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
recog_start_button = ttk.Button(recog_btn_frame, text="Start Recognition", command=start_recognition)
recog_start_button.pack(side=tk.LEFT, padx=(0, 10))
recog_stop_button = ttk.Button(recog_btn_frame, text="Stop Recognition", command=stop_recognition, state=tk.DISABLED)
recog_stop_button.pack(side=tk.LEFT)

# Sign Manager Tab (refactored layout)
sign_manager_frame = ttk.Frame(notebook)
notebook.add(sign_manager_frame, text="Sign Manager")

# --- All Signs Section ---
signs_frame = ttk.LabelFrame(sign_manager_frame, text="All Signs")
signs_frame.pack(fill=tk.X, padx=20, pady=(15, 5))

sign_listbox = tk.Listbox(signs_frame, width=40, height=8)
sign_listbox.pack(fill=tk.X, padx=10, pady=(10, 10))

sign_btn_frame = ttk.Frame(signs_frame)
sign_btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
add_sign_button = ttk.Button(sign_btn_frame, text="Add Sign", command=add_sign)
add_sign_button.pack(side=tk.LEFT, padx=(0, 10))
delete_sign_button = ttk.Button(sign_btn_frame, text="Delete Sign", command=delete_sign)
delete_sign_button.pack(side=tk.LEFT)

# --- Actions Section ---
action_frame = ttk.LabelFrame(sign_manager_frame, text="Actions")
action_frame.pack(fill=tk.X, padx=20, pady=(5, 5))

action_listbox = tk.Listbox(action_frame, width=30, height=6)
action_listbox.pack(fill=tk.X, padx=10, pady=(10, 10))

action_btn_frame = ttk.Frame(action_frame)
action_btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
add_action_btn = ttk.Button(action_btn_frame, text="Add", command=add_action)
add_action_btn.pack(side=tk.LEFT, padx=(0, 10))
delete_action_btn = ttk.Button(action_btn_frame, text="Delete", command=delete_action)
delete_action_btn.pack(side=tk.LEFT)

# --- Mappings Section ---
mapping_frame = ttk.LabelFrame(sign_manager_frame, text="Mappings")
mapping_frame.pack(fill=tk.X, padx=20, pady=(5, 15))

mapping_listbox = tk.Listbox(mapping_frame, width=40, height=6)
mapping_listbox.pack(fill=tk.X, padx=10, pady=(10, 10))

mapping_dropdown_frame = ttk.Frame(mapping_frame)
mapping_dropdown_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
sign_var = tk.StringVar()
action_var = tk.StringVar()
sign_dropdown = ttk.Combobox(mapping_dropdown_frame, textvariable=sign_var, state="readonly", width=18)
sign_dropdown.pack(side=tk.LEFT, padx=(0, 10))
action_dropdown = ttk.Combobox(mapping_dropdown_frame, textvariable=action_var, state="readonly", width=18)
action_dropdown.pack(side=tk.LEFT, padx=(0, 10))
map_btn = ttk.Button(mapping_dropdown_frame, text="Map", command=map_sign_to_action)
map_btn.pack(side=tk.LEFT, padx=(0, 10))
unmap_btn = ttk.Button(mapping_dropdown_frame, text="Unmap", command=unmap_sign)
unmap_btn.pack(side=tk.LEFT)

# --- Update all lists on startup and after changes ---
def update_all_lists():
    update_sign_listbox()
    update_action_listbox()
    update_mapping_dropdowns()
    update_mapping_listbox()

update_all_lists()

root.after(100, poll_recognition_queue)
root.mainloop() 