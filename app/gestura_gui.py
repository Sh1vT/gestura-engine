import cv2
import numpy as np
import mediapipe as mp
import json
import time
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox, filedialog
import threading
import subprocess
import queue
import pyautogui
import shutil
import zipfile
import tempfile
from app.utils import load_gesture_data, save_gesture_data, load_actions, save_actions, load_mappings, save_mappings
recognition_proc = None
recognition_queue = queue.Queue()
recognition_thread = None

# --- Gesture Logic ---
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
            "python3", "-u", "-m", "app.record_sign", sign_name
        ], check=True)
        print("[Main App] Subprocess for sign recording finished.")
        import time
        import cv2
        time.sleep(1)
        cv2.destroyAllWindows()
        print("[Main App] Called cv2.destroyAllWindows() after subprocess.")
        # After subprocess, reload all modules
        global actions, mappings
        gesture_dict, actions, mappings = load_all_module_data()
        update_all_lists()
        messagebox.showinfo("Success", f"Sign '{sign_name}' added.")
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", f"Failed to add sign '{sign_name}'. See terminal for details.")

def delete_sign():
    selection = sign_listbox.curselection()
    if not selection:
        return
    sign_name = sign_listbox.get(selection[0])
    if messagebox.askyesno("Delete Sign", f"Are you sure you want to delete '{sign_name}'?"):
        # Remove from data/gestures.json
        gestures_path = os.path.join("data", "gestures.json")
        if os.path.exists(gestures_path):
            with open(gestures_path, "r") as f:
                gestures = json.load(f)
            gestures.pop(sign_name, None)
            with open(gestures_path, "w") as f:
                json.dump(gestures, f, indent=2)
        # Reload all modules
        global gesture_dict, actions, mappings
        gesture_dict, actions, mappings = load_all_module_data()
        update_all_lists()
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
            "python3", "-u", "-m", "app.recognize_signs"
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
        execute_action_for_sign(sign)
    root.after(100, poll_recognition_queue)

def execute_action_for_sign(sign):
    # Find mapping
    mapped = next((m for m in mappings if m["sign"] == sign), None)
    if not mapped:
        return
    action = next((a for a in actions if a["name"] == mapped["action"]), None)
    if not action:
        return
    action_type = action.get("type")
    params = action.get("params", {})
    if action_type == "macro":
        keys = params.get("keys", "")
        if keys:
            try:
                # pyautogui expects keys as a list, split on +
                key_list = [k.strip() for k in keys.split("+")]
                pyautogui.hotkey(*key_list)
            except Exception as e:
                messagebox.showerror("Macro Error", f"Failed to execute macro: {e}")
    elif action_type == "log":
        logfile = params.get("logfile")
        tracked_signs = params.get("signs", [])
        threshold = params.get("threshold", 1)
        if logfile and sign in tracked_signs:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                # Count occurrences so far
                count = 0
                try:
                    with open(logfile, "r") as f:
                        lines = f.readlines()
                        count = sum(1 for line in lines if sign in line)
                except FileNotFoundError:
                    pass
                # Write log
                with open(logfile, "a") as f:
                    f.write(f"{timestamp} - {sign}\n")
                # If threshold reached, write at top
                if count + 1 >= threshold:
                    with open(logfile, "r+") as f:
                        content = f.read()
                        f.seek(0, 0)
                        f.write(f"Threshold reached for {sign}\n" + content)
            except Exception as e:
                messagebox.showerror("Log Error", f"Failed to log sign: {e}")
    elif action_type == "function":
        code = params.get("code", "")
        if code:
            try:
                exec(code, {"sign": sign})
            except Exception as e:
                messagebox.showerror("Function Error", f"Error in user function: {e}")

# --- Globals and State ---
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACTIONS_FILE = os.path.join(BASE_DIR, "../data/actions.json")
MAPPINGS_FILE = os.path.join(BASE_DIR, "../data/map.json")
GESTURE_FILE = "data/gestures.json"
MODULES_DIR = os.path.join(BASE_DIR, '../modules')
if not os.path.exists(MODULES_DIR):
    os.makedirs(MODULES_DIR)

# --- Utility: Load and merge from modules ---
def get_available_modules_list():
    return [name for name in os.listdir(MODULES_DIR) if os.path.isdir(os.path.join(MODULES_DIR, name))]

def load_all_module_data(active_modules=None):
    gesture_dict = {}
    actions = []
    mappings = []
    # Always load gestures from the main data/gestures.json
    if os.path.exists("data/gestures.json"):
        gesture_dict.update(load_gesture_data("data/gestures.json"))
    # Always load actions from the main data/actions.json
    if os.path.exists("data/actions.json"):
        try:
            with open("data/actions.json", "r") as f:
                actions.extend(json.load(f))
        except Exception as e:
            print(f"[Data Load] Error loading actions from data/actions.json: {e}")
    # Always load mappings from the main data/map.json
    if os.path.exists("data/map.json"):
        try:
            with open("data/map.json", "r") as f:
                mappings.extend(json.load(f))
        except Exception as e:
            print(f"[Data Load] Error loading mappings from data/map.json: {e}")
    # Load from selected modules (or all if not specified)
    if os.path.exists(MODULES_DIR):
        modules_to_load = active_modules if active_modules is not None else os.listdir(MODULES_DIR)
        for mod in modules_to_load:
            mod_path = os.path.join(MODULES_DIR, mod)
            if not os.path.isdir(mod_path):
                continue
            # Gestures
            gfile = os.path.join(mod_path, "gestures.json")
            if os.path.exists(gfile):
                gdata = load_gesture_data(gfile)
                gesture_dict.update(gdata)
            # Actions
            afile = os.path.join(mod_path, "actions.json")
            if os.path.exists(afile):
                try:
                    with open(afile, "r") as f:
                        adata = json.load(f)
                        actions.extend(adata)
                except Exception as e:
                    print(f"[Module Load] Error loading actions from {afile}: {e}")
            # Mappings
            mfile = os.path.join(mod_path, "map.json")
            if os.path.exists(mfile):
                try:
                    with open(mfile, "r") as f:
                        mdata = json.load(f)
                        mappings.extend(mdata)
                except Exception as e:
                    print(f"[Module Load] Error loading mappings from {mfile}: {e}")
    return gesture_dict, actions, mappings

# --- Use merged data from modules ---
# On startup, load all modules (default: all enabled)
all_modules_startup = get_available_modules_list()
gesture_dict, actions, mappings = load_all_module_data(all_modules_startup)

# Update all update_* functions to use the new globals (already done, just ensure they use gesture_dict/actions/mappings)
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

# --- Persistent Actions and Mappings ---
actions = load_actions(ACTIONS_FILE)  # List of dicts: {"name": ...}
mappings = load_mappings(MAPPINGS_FILE)  # List of dicts: {"sign": ..., "action": ...}

def save_actions_wrapper():
    save_actions(ACTIONS_FILE, actions)

def save_mappings_wrapper():
    save_mappings(MAPPINGS_FILE, mappings)

# --- In-memory Actions and Mappings ---
# actions = load_actions()  # List of dicts: {"name": ...}
# mappings = load_mappings()  # List of dicts: {"sign": ..., "action": ...}

# Update all places where actions/mappings are modified to save to disk

def add_action():
    dialog = tk.Toplevel(root)
    dialog.title("Add Action")
    dialog.geometry("400x400")
    dialog.grab_set()

    tk.Label(dialog, text="Action Name:").pack(anchor="w", padx=10, pady=(10, 0))
    name_var = tk.StringVar()
    name_entry = tk.Entry(dialog, textvariable=name_var)
    name_entry.pack(fill=tk.X, padx=10, pady=(0, 10))

    tk.Label(dialog, text="Action Type:").pack(anchor="w", padx=10, pady=(0, 0))
    type_var = tk.StringVar(value="macro")
    type_combo = ttk.Combobox(dialog, textvariable=type_var, state="readonly", values=["macro", "log", "function"])
    type_combo.pack(fill=tk.X, padx=10, pady=(0, 10))

    # Frames for each type
    macro_frame = tk.Frame(dialog)
    log_frame = tk.Frame(dialog)
    func_frame = tk.Frame(dialog)

    # Macro
    tk.Label(macro_frame, text="Key Combination (e.g. ctrl+shift+s):").pack(anchor="w", padx=0, pady=(0, 0))
    macro_keys_var = tk.StringVar()
    macro_entry = tk.Entry(macro_frame, textvariable=macro_keys_var)
    macro_entry.pack(fill=tk.X, padx=0, pady=(0, 10))

    # Log
    tk.Label(log_frame, text="Logfile Name:").pack(anchor="w", padx=0, pady=(0, 0))
    log_file_var = tk.StringVar()
    log_file_entry = tk.Entry(log_frame, textvariable=log_file_var)
    log_file_entry.pack(fill=tk.X, padx=0, pady=(0, 10))
    tk.Label(log_frame, text="Select Signs to Track:").pack(anchor="w", padx=0, pady=(0, 0))
    log_signs_listbox = tk.Listbox(log_frame, selectmode=tk.MULTIPLE, height=5)
    for sign in gesture_dict.keys():
        log_signs_listbox.insert(tk.END, sign)
    log_signs_listbox.pack(fill=tk.X, padx=0, pady=(0, 10))
    tk.Label(log_frame, text="Threshold (applies to all selected signs):").pack(anchor="w", padx=0, pady=(0, 0))
    log_threshold_var = tk.StringVar(value="1")
    log_threshold_entry = tk.Entry(log_frame, textvariable=log_threshold_var)
    log_threshold_entry.pack(fill=tk.X, padx=0, pady=(0, 10))

    # Function
    tk.Label(func_frame, text="Python Function (def ...):").pack(anchor="w", padx=0, pady=(0, 0))
    func_text = tk.Text(func_frame, height=8)
    func_text.pack(fill=tk.BOTH, padx=0, pady=(0, 10), expand=True)

    # Show/hide frames
    def show_type_frame(*args):
        macro_frame.pack_forget()
        log_frame.pack_forget()
        func_frame.pack_forget()
        if type_var.get() == "macro":
            macro_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        elif type_var.get() == "log":
            log_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        elif type_var.get() == "function":
            func_frame.pack(fill=tk.BOTH, padx=10, pady=(0, 10), expand=True)
    type_var.trace_add("write", show_type_frame)
    show_type_frame()

    def on_ok():
        name = name_var.get().strip()
        if not name:
            messagebox.showerror("Error", "Action name required.", parent=dialog)
            return
        global actions
        actions = load_actions(ACTIONS_FILE)  # Reload from disk before modifying
        if any(a["name"] == name for a in actions):
            messagebox.showerror("Error", "Action name must be unique.", parent=dialog)
            return
        action_type = type_var.get()
        params = {}
        if action_type == "macro":
            keys = macro_keys_var.get().strip()
            if not keys:
                messagebox.showerror("Error", "Key combination required.", parent=dialog)
                return
            params["keys"] = keys
        elif action_type == "log":
            logfile = log_file_var.get().strip()
            if not logfile:
                messagebox.showerror("Error", "Logfile name required.", parent=dialog)
                return
            selected = [log_signs_listbox.get(i) for i in log_signs_listbox.curselection()]
            if not selected:
                messagebox.showerror("Error", "Select at least one sign to track.", parent=dialog)
                return
            try:
                threshold = int(log_threshold_var.get())
            except ValueError:
                messagebox.showerror("Error", "Threshold must be an integer.", parent=dialog)
                return
            params["logfile"] = logfile
            params["signs"] = selected
            params["threshold"] = threshold
        elif action_type == "function":
            code = func_text.get("1.0", tk.END).strip()
            if not code:
                messagebox.showerror("Error", "Function code required.", parent=dialog)
                return
            params["code"] = code
        actions.append({"name": name, "type": action_type, "params": params})
        update_action_listbox()
        update_mapping_dropdowns()
        save_actions_wrapper()
        dialog.destroy()

    ok_btn = ttk.Button(dialog, text="OK", command=on_ok)
    ok_btn.pack(side=tk.RIGHT, padx=10, pady=10)
    cancel_btn = ttk.Button(dialog, text="Cancel", command=dialog.destroy)
    cancel_btn.pack(side=tk.RIGHT, padx=0, pady=10)

def delete_action():
    global actions
    actions = load_actions(ACTIONS_FILE)  # Reload from disk before modifying
    sel = action_listbox.curselection()
    if sel:
        idx = sel[0]
        actions.pop(idx)
        update_action_listbox()
        update_mapping_dropdowns()
        update_mapping_listbox()
        save_actions()

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
    global mappings
    mappings = load_mappings(MAPPINGS_FILE)  # Reload from disk before modifying
    sign = sign_var.get()
    action = action_var.get()
    if sign and action:
        mappings = [m for m in mappings if m["sign"] != sign]
        mappings.append({"sign": sign, "action": action})
        update_mapping_listbox()
        save_mappings_wrapper()

def unmap_sign():
    global mappings
    mappings = load_mappings(MAPPINGS_FILE)  # Reload from disk before modifying
    sign = sign_var.get()
    mappings = [m for m in mappings if m["sign"] != sign]
    update_mapping_listbox()
    save_mappings_wrapper()

# --- Module Import/Export UI ---
def export_selected_modules():
    import os
    import shutil
    import zipfile
    import tempfile
    selected_modules = [mod for mod, var in module_vars.items() if var.get()]
    include_data = include_data_var.get()
    data_module_name = data_module_name_var.get().strip()
    if not selected_modules and not include_data:
        messagebox.showerror("Export", "No modules selected for export.")
        return
    # Create temp dir
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy selected modules
        for mod in selected_modules:
            src = os.path.join("modules", mod)
            dst = os.path.join(temp_dir, mod)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
        # Optionally include data as a module
        if include_data and data_module_name:
            data_dst = os.path.join(temp_dir, data_module_name)
            os.makedirs(data_dst, exist_ok=True)
            for fname in ["gestures.json", "actions.json", "map.json"]:
                src = os.path.join("data", fname)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(data_dst, fname))
        # Ask user for save location
        zip_path = filedialog.asksaveasfilename(
            defaultextension=".zip",
            filetypes=[("Zip files", "*.zip")],
            title="Export Modules As Zip"
        )
        if not zip_path:
            return
        # Zip contents
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, temp_dir)
                    zipf.write(abs_path, rel_path)
        messagebox.showinfo("Export", f"Exported to {zip_path}")


def import_modules():
    import os
    import shutil
    import zipfile
    import tempfile
    # Ask user for zip file
    zip_path = filedialog.askopenfilename(
        filetypes=[("Zip files", "*.zip")],
        title="Import Modules Zip"
    )
    if not zip_path:
        return
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, "r") as zipf:
            zipf.extractall(temp_dir)
        # For each folder in temp_dir, copy to modules/
        for item in os.listdir(temp_dir):
            src = os.path.join(temp_dir, item)
            dst = os.path.join("modules", item)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    # Prompt user for conflict resolution
                    resp = messagebox.askquestion(
                        "Module Conflict",
                        f"Module '{item}' already exists. Overwrite? (Yes = Overwrite, No = Skip, Cancel = Rename)",
                        icon='warning'
                    )
                    if resp == 'yes':
                        shutil.rmtree(dst)
                        shutil.copytree(src, dst)
                    elif resp == 'no':
                        continue
                    else:  # Cancel = Rename
                        # Ask for new name
                        new_name = simpledialog.askstring("Rename Module", f"Enter new name for imported module '{item}':")
                        if not new_name:
                            continue
                        new_dst = os.path.join("modules", new_name)
                        if os.path.exists(new_dst):
                            messagebox.showerror("Import", f"Module '{new_name}' already exists. Skipping.")
                            continue
                        shutil.copytree(src, new_dst)
                else:
                    shutil.copytree(src, dst)
        refresh_module_checkboxes()
        messagebox.showinfo("Import", "Import complete.")

# --- Tkinter UI ---
root = tk.Tk()
root.title("Gestura - ISL Interpreter")
root.geometry("600x900")

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

# --- Modules Subsection (for activation) ---
modules_select_frame = ttk.LabelFrame(signs_frame, text="Modules")
modules_select_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

active_module_vars = {}
active_module_checkboxes = []

def on_active_modules_changed():
    # Get checked modules
    active_modules = [mod for mod, var in active_module_vars.items() if var.get()]
    # Reload data from only checked modules (plus main data)
    global gesture_dict, actions, mappings
    gesture_dict, actions, mappings = load_all_module_data(active_modules)
    update_all_lists()

def refresh_active_module_checkboxes():
    for cb in active_module_checkboxes:
        cb.destroy()
    active_module_checkboxes.clear()
    active_module_vars.clear()
    modules = get_available_modules_list()
    for mod in modules:
        var = tk.BooleanVar(value=False)  # Default: all unchecked
        cb = tk.Checkbutton(modules_select_frame, text=mod, variable=var, command=on_active_modules_changed)
        cb.pack(anchor="w")
        active_module_vars[mod] = var
        active_module_checkboxes.append(cb)
    on_active_modules_changed()  # Ensure data matches UI at startup

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

# --- Module Import/Export UI (moved here) ---
module_export_frame = ttk.LabelFrame(sign_manager_frame, text="Module Import/Export")
module_export_frame.pack(fill=tk.X, padx=20, pady=(10, 5))

# Module selection
module_vars = {}
module_checkboxes = []
def refresh_module_checkboxes():
    for cb in module_checkboxes:
        cb.destroy()
    module_checkboxes.clear()
    module_vars.clear()
    modules = get_available_modules_list()
    for mod in modules:
        var = tk.BooleanVar()
        cb = tk.Checkbutton(module_export_frame, text=mod, variable=var)
        cb.pack(anchor="w")
        module_vars[mod] = var
        module_checkboxes.append(cb)
refresh_module_checkboxes()

# Option to include current data folder as a module
include_data_var = tk.BooleanVar()
data_module_name_var = tk.StringVar(value="my_custom_module")
data_cb = tk.Checkbutton(module_export_frame, text="Include current data folder as module:", variable=include_data_var)
data_cb.pack(anchor="w")
data_name_entry = tk.Entry(module_export_frame, textvariable=data_module_name_var, width=20)
data_name_entry.pack(anchor="w", padx=30)

# Export button
export_btn = ttk.Button(module_export_frame, text="Export Selected Modules", command=lambda: export_selected_modules())
export_btn.pack(side=tk.LEFT, padx=(10, 10), pady=10)
# Import button
import_btn = ttk.Button(module_export_frame, text="Import Modules", command=lambda: import_modules())
import_btn.pack(side=tk.LEFT, padx=(0, 10), pady=10)

# --- Update all lists on startup and after changes ---
def update_all_lists():
    update_sign_listbox()
    update_action_listbox()
    update_mapping_dropdowns()
    update_mapping_listbox()
    # refresh_active_module_checkboxes()  # Removed to prevent checkbox reset on every reload

refresh_active_module_checkboxes()  # Call after all functions are defined

print("Loaded actions:", actions)
print("Loaded mappings:", mappings)
update_all_lists()

root.after(100, poll_recognition_queue)
root.mainloop() 