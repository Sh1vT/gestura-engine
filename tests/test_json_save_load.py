import os
import json
import numpy as np
import tempfile
from app.utils import save_gesture_data, load_gesture_data

def test_save_gesture_data_and_load():
    # Arrange: create a gesture dict with numpy arrays
    gestures = {
        "wave": {
            "start": np.array([[0,0,0],[1,1,1]], dtype=np.float32),
            "mid1": np.array([[2,2,2],[3,3,3]], dtype=np.float32),
            "mid2": np.array([[4,4,4],[5,5,5]], dtype=np.float32),
            "end": np.array([[6,6,6],[7,7,7]], dtype=np.float32)
        }
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "gestures.json")
        # Act: save and then load
        save_gesture_data(test_file, gestures)
        loaded = load_gesture_data(test_file)
        # Assert: structure and values match
        assert "wave" in loaded
        for stage in ["start", "mid1", "mid2", "end"]:
            assert isinstance(loaded["wave"][stage], np.ndarray)
            assert loaded["wave"][stage].shape == (2,3)
            np.testing.assert_array_equal(
                loaded["wave"][stage], gestures["wave"][stage]
            ) 

def test_load_gesture_data_success():
    # Create a temporary directory and file
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "gestures.json")
        # Example gesture data
        data = {
            "wave": {
                "start": [[0,0,0],[1,1,1]],
                "mid1": [[2,2,2],[3,3,3]],
                "mid2": [[4,4,4],[5,5,5]],
                "end": [[6,6,6],[7,7,7]]
            }
        }
        with open(test_file, "w") as f:
            json.dump(data, f)
        # Act
        gestures = load_gesture_data(test_file)
        # Assert
        assert "wave" in gestures
        for stage in ["start", "mid1", "mid2", "end"]:
            assert isinstance(gestures["wave"][stage], np.ndarray)
            assert gestures["wave"][stage].shape == (2,3)
            np.testing.assert_array_equal(
                gestures["wave"][stage], np.array(data["wave"][stage], dtype=np.float32)
            )

def test_load_gesture_data_missing_file():
    # Should return empty dict if file does not exist
    gestures = load_gesture_data("/tmp/nonexistent_file_12345.json")
    assert gestures == {} 