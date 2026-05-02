"""
Data preprocessing — extracts wrist-normalized hand keypoints from raw images.
Extractor: hands_normalized (HandsDetector + wrist-relative coords).
This is the ONLY preprocessing script. The holistic version has been removed.
"""

import os
import cv2
import time
import numpy as np
from tqdm import tqdm

from src.config import (
    EXTRACTOR_MODE,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    ACTIONS,
    NO_SEQUENCES,
    SEQUENCE_LENGTH,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    TOTAL_KEYPOINTS,
)
from src.utils.mediapipe_utils import (
    mediapipe_detection,
    extract_keypoints_hands_normalized,
    HandsDetector,
    draw_hand_landmarks,
)
from src.utils.data_utils import save_sequence, verify_data_integrity

assert (
    EXTRACTOR_MODE == "hands_normalized"
), f"Preprocessing only supports extractor='hands_normalized', got '{EXTRACTOR_MODE}'"


def _sorted_images(action_path: str) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [
        f for f in os.listdir(action_path) if os.path.splitext(f.lower())[1] in exts
    ]
    try:
        return sorted(files, key=lambda f: int(os.path.splitext(f)[0]))
    except ValueError:
        return sorted(files)


def _create_dirs():
    for action in ACTIONS:
        for seq in range(NO_SEQUENCES):
            os.makedirs(
                os.path.join(PROCESSED_DATA_DIR, action, str(seq)), exist_ok=True
            )


def _validate_keypoint_shape(keypoints: np.ndarray):
    if keypoints.shape != (TOTAL_KEYPOINTS,):
        raise RuntimeError(
            f"Extractor output shape mismatch: expected ({TOTAL_KEYPOINTS},), "
            f"got {keypoints.shape}. Re-check extractor and config."
        )


def process(visualize: bool = False, limit_actions: list | None = None) -> dict:
    actions = limit_actions or ACTIONS
    _create_dirs()

    stats = {
        "processed": 0,
        "skipped": 0,
        "sequences": 0,
        "failed_frames": 0,
        "start": time.time(),
    }

    with HandsDetector(
        max_num_hands=2,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands:
        for action in actions:
            action_path = os.path.join(RAW_DATA_DIR, action)
            if not os.path.exists(action_path):
                print(f"  ✗ missing: {action_path}")
                stats["skipped"] += 1
                continue

            files = _sorted_images(action_path)
            if not files:
                print(f"  ✗ no images: {action_path}")
                stats["skipped"] += 1
                continue

            n_seqs = min(NO_SEQUENCES, len(files) // SEQUENCE_LENGTH)
            if n_seqs == 0:
                print(
                    f"  ✗ '{action}': only {len(files)} images, need {SEQUENCE_LENGTH}"
                )
                stats["skipped"] += 1
                continue

            if len(files) < NO_SEQUENCES * SEQUENCE_LENGTH:
                print(
                    f"  ⚠ '{action}': {len(files)} images → {n_seqs} sequences (expected {NO_SEQUENCES})"
                )

            for seq_num in tqdm(
                range(n_seqs), desc=f"  {action}", ncols=70, leave=False
            ):
                batch = files[
                    seq_num * SEQUENCE_LENGTH : (seq_num + 1) * SEQUENCE_LENGTH
                ]
                for frame_num, fname in enumerate(batch):
                    frame = cv2.imread(os.path.join(action_path, fname))
                    if frame is None:
                        stats["failed_frames"] += 1
                        continue

                    image, results = mediapipe_detection(frame, hands)
                    keypoints = extract_keypoints_hands_normalized(results)
                    _validate_keypoint_shape(keypoints)
                    save_sequence(keypoints, action, seq_num, frame_num)

                    if visualize:
                        draw_hand_landmarks(image, results)
                        cv2.imshow(f"Preprocessing — {action}", image)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            visualize = False
                            cv2.destroyAllWindows()

                stats["sequences"] += 1
            stats["processed"] += 1
            print(f"  ✓ '{action}': {n_seqs} sequences")

    if visualize:
        cv2.destroyAllWindows()

    elapsed = time.time() - stats["start"]
    stats["duration_s"] = elapsed
    return stats


def main():
    print(f"\n{'='*60}")
    print(f"  DATA PREPROCESSING  |  extractor={EXTRACTOR_MODE}")
    print(f"{'='*60}")
    print(f"  Source : {RAW_DATA_DIR}")
    print(f"  Output : {PROCESSED_DATA_DIR}")
    print(
        f"  Actions: {len(ACTIONS)}  |  Sequences: {NO_SEQUENCES}  |  Frames: {SEQUENCE_LENGTH}"
    )
    print(f"{'='*60}\n")

    if input("Proceed? (y/n): ").strip().lower() != "y":
        return

    visualize = input("Visualize? (y/n): ").strip().lower() == "y"

    stats = process(visualize=visualize)

    print(f"\n{'='*60}")
    print(f"  Done  |  {stats['processed']} actions  |  {stats['sequences']} sequences")
    print(
        f"  Failed frames: {stats['failed_frames']}  |  Time: {stats['duration_s']:.1f}s"
    )
    print(f"{'='*60}\n")

    verify_data_integrity()


if __name__ == "__main__":
    main()
