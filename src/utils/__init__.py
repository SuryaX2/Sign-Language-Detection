"""Utilities package for sign language detection."""

from .mediapipe_utils import (
    mediapipe_detection,
    draw_hand_landmarks,
    extract_keypoints_hands_normalized,  # canonical extractor
    HandsDetector,
    # legacy — kept for data inspection only, do not use in training/inference
    draw_styled_landmarks,
    extract_keypoints_holistic,
    extract_keypoints_hands,
    HolisticDetector,
)

from .data_utils import (
    load_sequences,
    split_data,
    verify_data_integrity,
    save_sequence,
    get_action_statistics,
)

from .visualization import (
    visualize_sequence,
    visualize_all_actions_sample,
    plot_class_distribution,
    visualize_keypoint_variance,
    create_prediction_heatmap,
    plot_misclassifications,
)

__all__ = [
    "mediapipe_detection",
    "draw_hand_landmarks",
    "extract_keypoints_hands_normalized",
    "HandsDetector",
    "draw_styled_landmarks",
    "extract_keypoints_holistic",
    "extract_keypoints_hands",
    "HolisticDetector",
    "load_sequences",
    "split_data",
    "verify_data_integrity",
    "save_sequence",
    "get_action_statistics",
    "visualize_sequence",
    "visualize_all_actions_sample",
    "plot_class_distribution",
    "visualize_keypoint_variance",
    "create_prediction_heatmap",
    "plot_misclassifications",
]
