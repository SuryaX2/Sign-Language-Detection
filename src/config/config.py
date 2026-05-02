"""
Central configuration for Sign Language Detection.
Extractor: hands-normalized (wrist-relative, position invariant).
"""

import os

# ---------------------------------------------------------------------------
# EXTRACTOR CONTRACT — enforced at preprocessing, training, and inference
# Changing this value without re-running preprocessing + retraining will break
# the train/inference distribution match.
# ---------------------------------------------------------------------------
EXTRACTOR_MODE = "hands_normalized"  # only valid value in this project

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw", "Indian")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "action_transformer.keras")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
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

NO_SEQUENCES = 120
SEQUENCE_LENGTH = 10

# ---------------------------------------------------------------------------
# MediaPipe — Hands detector only
# ---------------------------------------------------------------------------
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Feature dimensions
# pose(132) and face(1404) are always zeros in hands_normalized mode.
# They are kept to preserve model input shape compatibility.
POSE_KEYPOINTS = 33 * 4  # 132  — zeros
FACE_KEYPOINTS = 468 * 3  # 1404 — zeros
HAND_KEYPOINTS = 21 * 3  # 63   — real data per hand
HAND_ONLY_KEYPOINTS = HAND_KEYPOINTS * 2
TOTAL_KEYPOINTS = POSE_KEYPOINTS + FACE_KEYPOINTS + (HAND_KEYPOINTS * 2)

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
EMBED_DIM = 256
NUM_HEADS = 4
FF_DIM = 512
DROPOUT_RATE = 0.3

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.20
RANDOM_STATE = 42
PREDICTION_THRESHOLD = 0.7

# ---------------------------------------------------------------------------
# Visualization (BGR)
# ---------------------------------------------------------------------------
LEFT_HAND_COLOR = (121, 22, 76)
LEFT_HAND_CONNECTION_COLOR = (121, 44, 250)
RIGHT_HAND_COLOR = (245, 117, 66)
RIGHT_HAND_CONNECTION_COLOR = (245, 66, 230)
LANDMARK_THICKNESS = 2
LANDMARK_RADIUS = 4
CONNECTION_THICKNESS = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def create_directories(no_sequences=None):
    n = no_sequences or NO_SEQUENCES
    dirs = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SAVED_MODELS_DIR,
        LOGS_DIR,
        RESULTS_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    for action in ACTIONS:
        for seq in range(n):
            os.makedirs(
                os.path.join(PROCESSED_DATA_DIR, action, str(seq)), exist_ok=True
            )
    print(f"✓ Directories ready  |  extractor={EXTRACTOR_MODE}  |  sequences={n}")


def get_label_map():
    return {label: idx for idx, label in enumerate(ACTIONS)}


def get_num_classes():
    return len(ACTIONS)
