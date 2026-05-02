"""
MediaPipe utilities for sign language detection.

Canonical extractor : extract_keypoints_hands_normalized()
                      → wrist-relative coords, HandsDetector
                      → shape (1662,): zeros(1536) + lh(63) + rh(63)

Legacy extractor    : extract_keypoints_holistic()  [DO NOT USE for new training]
                      → kept only for reference / migration inspection
"""

import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def mediapipe_detection(image: np.ndarray, model) -> tuple[np.ndarray, object]:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = model.process(rgb)
    rgb.flags.writeable = True
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), results


def draw_hand_landmarks(image: np.ndarray, results) -> None:
    if not results.multi_hand_landmarks:
        return
    for lm in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            lm,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )


def extract_keypoints_hands_normalized(results) -> np.ndarray:
    """
    Canonical extractor — use this everywhere (preprocessing + inference).

    Wrist-relative normalization makes keypoints position invariant:
    each landmark = (x,y,z) - wrist(x,y,z), so wrist is always (0,0,0).

    Output shape: (1662,)
      [zeros(132) | zeros(1404) | left_hand(63) | right_hand(63)]
    """
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)

    if results.multi_hand_landmarks and results.multi_handedness:
        for lm, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            pts = np.array([[p.x, p.y, p.z] for p in lm.landmark])
            pts -= pts[0]  # wrist-relative: landmark 0 becomes (0,0,0)
            label = handedness.classification[0].label
            if label == "Left":
                lh = pts.flatten()
            elif label == "Right":
                rh = pts.flatten()

    return np.concatenate([np.zeros(33 * 4), np.zeros(468 * 3), lh, rh])


def draw_styled_landmarks(image: np.ndarray, results) -> None:
    """Draw landmarks for holistic results (used only during legacy inspection)."""
    pairs = [
        (
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            (80, 110, 10),
            (80, 256, 121),
            1,
            1,
        ),
        (
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            (80, 22, 10),
            (80, 44, 121),
            2,
            4,
        ),
        (
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            (121, 22, 76),
            (121, 44, 250),
            2,
            4,
        ),
        (
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            (245, 117, 66),
            (245, 66, 230),
            2,
            4,
        ),
    ]
    for lm, conn, lm_color, conn_color, thickness, radius in pairs:
        if lm:
            mp_drawing.draw_landmarks(
                image,
                lm,
                conn,
                mp_drawing.DrawingSpec(
                    color=lm_color, thickness=thickness, circle_radius=radius
                ),
                mp_drawing.DrawingSpec(
                    color=conn_color, thickness=thickness, circle_radius=radius // 2
                ),
            )


# ---------------------------------------------------------------------------
# LEGACY — DO NOT USE FOR NEW TRAINING OR INFERENCE
# Kept only to allow inspection of old processed data or migration checks.
# Using this extractor with a model trained on hands_normalized will produce
# near-random predictions.
# ---------------------------------------------------------------------------
def extract_keypoints_holistic(results) -> np.ndarray:
    """LEGACY: absolute coords, pose+face+hands. NOT compatible with current model."""
    pose = (
        np.array(
            [[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )
    face = (
        np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark]).flatten()
        if results.face_landmarks
        else np.zeros(468 * 3)
    )
    lh = (
        np.array(
            [[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    rh = (
        np.array(
            [[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([pose, face, lh, rh])


def extract_keypoints_hands(results) -> np.ndarray:
    """LEGACY: absolute hand coords, no normalization. NOT compatible with current model."""
    lh = rh = np.zeros(21 * 3)
    if results.multi_hand_landmarks and results.multi_handedness:
        for lm, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            pts = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()
            if handedness.classification[0].label == "Left":
                lh = pts
            else:
                rh = pts
    return np.concatenate([np.zeros(33 * 4), np.zeros(468 * 3), lh, rh])


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------
class HandsDetector:
    def __init__(
        self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ):
        self._cfg = dict(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._hands = None

    def __enter__(self):
        self._hands = mp_hands.Hands(**self._cfg)
        return self._hands

    def __exit__(self, *_):
        if self._hands:
            self._hands.close()


class HolisticDetector:
    """LEGACY context manager — only needed for old data inspection."""

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self._cfg = dict(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._holistic = None

    def __enter__(self):
        self._holistic = mp_holistic.Holistic(**self._cfg)
        return self._holistic

    def __exit__(self, *_):
        if self._holistic:
            self._holistic.close()
