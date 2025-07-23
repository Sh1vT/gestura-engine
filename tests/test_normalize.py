import numpy as np
from app.utils import normalize_landmarks

def test_normalize_landmarks():
    # Arrange: create a simple square in 3D
    landmarks = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ], dtype=np.float32)
    # Act
    normalized = normalize_landmarks(landmarks)
    # Assert: center should be at (0,0), and max extent should be 1
    assert np.allclose(np.mean(normalized, axis=0)[:2], [0, 0], atol=1e-6)
    assert np.isclose(np.max(normalized[:,0]) - np.min(normalized[:,0]), 1.0, atol=1e-6)
    assert np.isclose(np.max(normalized[:,1]) - np.min(normalized[:,1]), 1.0, atol=1e-6) 