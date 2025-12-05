"""
Configuration file for Sign Language Detection project.
All paths, hyperparameters, and constants are defined here.
"""

import os

# ============================================================================
# PROJECT PATHS
# ============================================================================
# Get the project root directory (assumes config.py is in src/config/)
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw", "Indian")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Model paths
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "action_transformer.h5")

# Logs and results
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
# All sign language actions (A-Z, 1-9)
ACTIONS = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]

# UPDATED: Number of sequences per action - changed from 40 to 20
NO_SEQUENCES = 20

# Number of frames per sequence
SEQUENCE_LENGTH = 20

# ============================================================================
# MEDIAPIPE CONFIGURATION
# ============================================================================
# MediaPipe detection confidence thresholds
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Keypoint dimensions
POSE_KEYPOINTS = 33 * 4  # 33 landmarks with x, y, z, visibility
FACE_KEYPOINTS = 468 * 3  # 468 landmarks with x, y, z
HAND_KEYPOINTS = 21 * 3  # 21 landmarks with x, y, z per hand
TOTAL_KEYPOINTS = POSE_KEYPOINTS + FACE_KEYPOINTS + (HAND_KEYPOINTS * 2)

# ============================================================================
# OPTIMIZED MODEL HYPERPARAMETERS FOR BETTER ACCURACY + FASTER TRAINING
# ============================================================================
# Transformer architecture - Increased capacity for better accuracy
EMBED_DIM = 512  # Increased from 256
NUM_HEADS = 6  # Increased from 4 for better attention
FF_DIM = 768  # Increased from 512
DROPOUT_RATE = 0.4  # Increased from 0.3 for better regularization

# Training parameters - Optimized for speed and accuracy
BATCH_SIZE = 64  # Increased from 32 for faster training
EPOCHS = 100  # Reduced from 200 but with better callbacks
LEARNING_RATE = 0.0005  # Slightly reduced for better convergence
VALIDATION_SPLIT = 0.20  # Increased from 0.15 for better validation
RANDOM_STATE = 42

# Prediction threshold
PREDICTION_THRESHOLD = 0.7  # Increased from 0.6 for higher confidence

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
# Colors for MediaPipe landmarks (BGR format)
FACE_COLOR = (80, 110, 10)
FACE_CONNECTION_COLOR = (80, 256, 121)
POSE_COLOR = (80, 22, 10)
POSE_CONNECTION_COLOR = (80, 44, 121)
LEFT_HAND_COLOR = (121, 22, 76)
LEFT_HAND_CONNECTION_COLOR = (121, 44, 250)
RIGHT_HAND_COLOR = (245, 117, 66)
RIGHT_HAND_CONNECTION_COLOR = (245, 66, 230)

# Drawing specifications
LANDMARK_THICKNESS = 2
LANDMARK_RADIUS = 4
CONNECTION_THICKNESS = 2


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def create_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SAVED_MODELS_DIR,
        LOGS_DIR,
        RESULTS_DIR,
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    # Create action folders in processed data directory
    for action in ACTIONS:
        action_dir = os.path.join(PROCESSED_DATA_DIR, action)
        os.makedirs(action_dir, exist_ok=True)

        # Create sequence folders
        for sequence in range(NO_SEQUENCES):
            sequence_dir = os.path.join(action_dir, str(sequence))
            os.makedirs(sequence_dir, exist_ok=True)

    print(f"✓ All directories created successfully")
    print(f"  - Raw data: {RAW_DATA_DIR}")
    print(f"  - Processed data: {PROCESSED_DATA_DIR}")
    print(f"  - Models: {SAVED_MODELS_DIR}")
    print(f"  - Logs: {LOGS_DIR}")
    print(f"  - Results: {RESULTS_DIR}")


def get_label_map():
    """Get mapping from action labels to numeric indices."""
    return {label: num for num, label in enumerate(ACTIONS)}


def get_num_classes():
    """Get total number of action classes."""
    return len(ACTIONS)


if __name__ == "__main__":
    # Test configuration by creating directories
    print("Creating project directory structure...")
    create_directories()
    print(f"\nTotal actions: {get_num_classes()}")
    print(f"Actions: {ACTIONS}")
