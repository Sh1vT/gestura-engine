# Gestura Engine

**Gestura Engine** is a modular, GUI-based Sign interpreter and gesture manager. It allows you to record, recognize, map, and manage sign gestures, and supports modular import/export of gesture sets. It aims for lower-end devices and uses algorithmic flow of FastDTW with keyframe sequencing and Mediapipe, staying away from heavier alternatives like CNN.  

---

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Gesture Matching Algorithm](#gesture-matching-algorithm)
  - [Dynamic Time Warping (DTW)](#dynamic-time-warping-dtw)
  - [Matching Stages](#matching-stages)
  - [DTW vs. CNN Comparison](#dtw-vs-cnn-comparison)
- [Complexity Analysis](#complexity-analysis)
  - [Time Complexity](#time-complexity)
  - [Space Complexity](#space-complexity)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Features
- **Real-time ISL gesture recognition** using webcam and MediaPipe.
- **Sign Manager**: Add, delete, and manage custom signs.
- **Actions**: Map signs to keyboard macros, logging, or custom Python functions.
- **Module System**: Organize signs/actions/mappings into modules for easy sharing.
- **Import/Export**: Zip and share modules, or import modules from others.
- **Dynamic Module Activation**: Enable/disable modules on the fly.
- **Persistent Data**: All data is stored in JSON files for easy editing and backup.

---

## Installation
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd gestura-engine
   ```
2. **Set up a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Install system dependencies:**
   - Python 3.7+
   - OpenCV (with webcam support)
   - [MediaPipe](https://google.github.io/mediapipe/)

---

## Usage
### Main UI
Start the GUI:
```bash
python3 app/gestura_gui.py
```
### Recognition
- Go to the **Recognition** tab.
- Click **Start Recognition** to begin gesture detection using your webcam.
- Detected signs will appear in the list.

### Sign Manager
- **All Signs**: View all available signs (from main data and enabled modules).
- **Add Sign**: Record a new sign using your webcam.
- **Delete Sign**: Remove a sign from your data.

### Actions
- **Add Action**: Create a new action (macro, log, or Python function).
- **Delete Action**: Remove an action.

### Mappings
- Map signs to actions.
- Unmap signs as needed.

### Modules
- **Enable/Disable Modules**: Use checkboxes in the All Signs section to activate/deactivate modules.
- **Import/Export**: Use the Module Import/Export section to share or load modules.

---

## Gesture Matching Algorithm

### 4.1 Dynamic Time Warping (DTW)
- Compares similarity between two time-series sequences.
- FastDTW reduces complexity from O(N²) to O(N).
- Ideal for real-time applications.

### 4.2 Matching Stages
- Gesture progression:
  - Start → Mid1 → Mid2 → End
- System uses a timeout mechanism to reset the sequence if the user pauses too long.

### 4.3 DTW vs. CNN Comparison
| Metric           | Gesture Detection (DTW) | CNN (Deep Learning) |
|------------------|------------------------|---------------------|
| Training         | No training needed      | Requires labeled dataset |
| Real-Time        | Yes                    | Slower without GPU  |
| Accuracy         | Moderate               | High                |
| Time Complexity  | O(G × N × M)           | O(C × D × F)        |
| Space Complexity | O(G × N × 3)           | High                |
| Speed            | Fast                   | Slower              |
| Scalability      | Very scalable          | Limited by model size |
| Hardware         | Low requirements        | GPU required        |
| Flexibility      | High (record anytime)  | Low (requires retraining) |
| Error Rate       | Higher                 | Lower               |
| Ease of Use      | Simple setup           | Complex ML pipeline |

---

## Complexity Analysis

### 5.1 Time Complexity
- O(G × N × M)
  - G = number of predefined gestures
  - N = keypoints per frame (typically 2 × 11 = 22)
  - M = number of recorded frames (typically 4)

### 5.2 Space Complexity
- O(G × N × 3)
  - G = number of gestures
  - N = number of keypoints per gesture
  - 3D = (x, y, z) per keypoint

---

## File Structure
```
gestura-engine/
├── app/
│   ├── gestura_gui.py
│   ├── recognize_signs.py
│   └── record_sign.py
├── data/
│   ├── gestures.json
│   ├── actions.json
│   └── map.json
├── modules/
│   └── <module folders>/
├── requirements.txt
└── README.md
```

---

## Troubleshooting
- **No webcam detected**: Ensure your webcam is connected and accessible by OpenCV.
- **No signs in list**: Make sure you have signs in `data/gestures.json` or enabled modules.
- **Module checkboxes not working**: Ensure you have at least one module folder in `modules/`.

---

## Contributing
Pull requests and suggestions are welcome! Please open an issue or PR for any improvements.

---

## License
[MIT License](LICENSE)