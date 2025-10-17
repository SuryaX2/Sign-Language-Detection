"""Quick test script to verify all dependencies are installed correctly."""

print("Testing installations...\n")

# Test TensorFlow
try:
    import tensorflow as tf
    print(f"✓ TensorFlow: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"  GPU Available: {len(gpus)} GPU(s) detected")
except Exception as e:
    print(f"✗ TensorFlow: {e}")

# Test MediaPipe
try:
    import mediapipe as mp
    print(f"✓ MediaPipe: {mp.__version__}")
except Exception as e:
    print(f"✗ MediaPipe: {e}")

# Test OpenCV
try:
    import cv2
    print(f"✓ OpenCV: {cv2.__version__}")
except Exception as e:
    print(f"✗ OpenCV: {e}")

# Test NumPy
try:
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")
except Exception as e:
    print(f"✗ NumPy: {e}")

# Test scikit-learn
try:
    import sklearn
    print(f"✓ scikit-learn: {sklearn.__version__}")
except Exception as e:
    print(f"✗ scikit-learn: {e}")

# Test Matplotlib
try:
    import matplotlib
    print(f"✓ Matplotlib: {matplotlib.__version__}")
except Exception as e:
    print(f"✗ Matplotlib: {e}")

# Test Seaborn
try:
    import seaborn as sns
    print(f"✓ Seaborn: {sns.__version__}")
except Exception as e:
    print(f"✗ Seaborn: {e}")

print("\n" + "="*50)
print("Testing imports from our project...")

try:
    from src.config import ACTIONS, SEQUENCE_LENGTH, create_directories
    print(f"✓ Config module works")
    print(f"  Actions: {len(ACTIONS)} classes")
    print(f"  Sequence length: {SEQUENCE_LENGTH}")
except Exception as e:
    print(f"✗ Config module: {e}")

try:
    from src.utils.mediapipe_utils import extract_keypoints_holistic
    print(f"✓ MediaPipe utils module works")
except Exception as e:
    print(f"✗ MediaPipe utils module: {e}")

print("\n✅ All tests completed!")