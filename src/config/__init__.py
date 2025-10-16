"""Configuration package for sign language detection."""

from .config import (
    # Paths
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAVED_MODELS_DIR,
    MODEL_PATH,
    LOGS_DIR,
    RESULTS_DIR,
    
    # Dataset config
    ACTIONS,
    NO_SEQUENCES,
    SEQUENCE_LENGTH,
    
    # MediaPipe config
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    TOTAL_KEYPOINTS,
    
    # Model hyperparameters
    EMBED_DIM,
    NUM_HEADS,
    FF_DIM,
    DROPOUT_RATE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    VALIDATION_SPLIT,
    RANDOM_STATE,
    PREDICTION_THRESHOLD,
    
    # Helper functions
    create_directories,
    get_label_map,
    get_num_classes
)

__all__ = [
    'PROJECT_ROOT',
    'DATA_DIR',
    'RAW_DATA_DIR',
    'PROCESSED_DATA_DIR',
    'SAVED_MODELS_DIR',
    'MODEL_PATH',
    'LOGS_DIR',
    'RESULTS_DIR',
    'ACTIONS',
    'NO_SEQUENCES',
    'SEQUENCE_LENGTH',
    'MIN_DETECTION_CONFIDENCE',
    'MIN_TRACKING_CONFIDENCE',
    'TOTAL_KEYPOINTS',
    'EMBED_DIM',
    'NUM_HEADS',
    'FF_DIM',
    'DROPOUT_RATE',
    'BATCH_SIZE',
    'EPOCHS',
    'LEARNING_RATE',
    'VALIDATION_SPLIT',
    'RANDOM_STATE',
    'PREDICTION_THRESHOLD',
    'create_directories',
    'get_label_map',
    'get_num_classes'
]
