"""
Data preprocessing script for sign language detection - HANDS ONLY VERSION.
Processes images using MediaPipe Hands instead of Holistic for better live demo performance.
"""

import os
import cv2
import time
import numpy as np
from tqdm import tqdm

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    ACTIONS,
    NO_SEQUENCES,
    SEQUENCE_LENGTH,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)

from src.utils.mediapipe_utils import (
    mediapipe_detection,
    extract_keypoints_hands_normalized,  # Using hands-only extraction
    HandsDetector,  # Using Hands instead of Holistic
)

from src.utils.data_utils import save_sequence, verify_data_integrity


def get_sorted_image_files(action_path):
    """
    Get sorted list of image files from action folder.
    """
    all_files = os.listdir(action_path)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [
        f for f in all_files if os.path.splitext(f.lower())[1] in image_extensions
    ]

    try:
        sorted_files = sorted(image_files, key=lambda f: int(os.path.splitext(f)[0]))
    except ValueError:
        print(f"  ⚠ Could not sort files numerically. Using alphabetical sort.")
        sorted_files = sorted(image_files)

    return sorted_files


def create_output_directories():
    """Create output directory structure for processed data."""
    print("Creating output directory structure...")

    for action in ACTIONS:
        for sequence in range(NO_SEQUENCES):
            sequence_dir = os.path.join(PROCESSED_DATA_DIR, action, str(sequence))
            os.makedirs(sequence_dir, exist_ok=True)

    print(
        f"✓ Created directories for {len(ACTIONS)} actions × {NO_SEQUENCES} sequences"
    )


def process_images_to_sequences(visualize=False, limit_actions=None):
    """
    Process raw images and extract keypoints using MediaPipe Hands.

    Args:
        visualize: If True, display images with landmarks
        limit_actions: List of specific actions to process

    Returns:
        Dictionary with processing statistics
    """
    actions_to_process = limit_actions if limit_actions else ACTIONS

    print("=" * 70)
    print("STARTING DATA PREPROCESSING (HANDS ONLY)")
    print("=" * 70)
    print(f"Source: {RAW_DATA_DIR}")
    print(f"Output: {PROCESSED_DATA_DIR}")
    print(f"Actions to process: {len(actions_to_process)}")
    print(f"Sequences per action: {NO_SEQUENCES}")
    print(f"Frames per sequence: {SEQUENCE_LENGTH}")
    print("=" * 70)

    stats = {
        "total_actions": len(actions_to_process),
        "total_sequences": 0,
        "total_frames": 0,
        "skipped_actions": 0,
        "failed_frames": 0,
        "start_time": time.time(),
    }

    create_output_directories()

    # Initialize MediaPipe Hands (CHANGED from Holistic)
    with HandsDetector(
        max_num_hands=2,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands:

        for action in actions_to_process:
            print(f"\n{'='*70}")
            print(f"Processing action: '{action}'")
            print(f"{'='*70}")

            action_path = os.path.join(RAW_DATA_DIR, action)

            if not os.path.exists(action_path):
                print(f"  ✗ Folder not found: {action_path}")
                stats["skipped_actions"] += 1
                continue

            sorted_files = get_sorted_image_files(action_path)

            if len(sorted_files) == 0:
                print(f"  ✗ No image files found in {action_path}")
                stats["skipped_actions"] += 1
                continue

            required_images = NO_SEQUENCES * SEQUENCE_LENGTH
            if len(sorted_files) < required_images:
                print(f"  ⚠ Warning: Not enough images for action '{action}'")
                print(f"    Found: {len(sorted_files)}, Required: {required_images}")
                actual_sequences = len(sorted_files) // SEQUENCE_LENGTH
            else:
                actual_sequences = NO_SEQUENCES

            print(f"  Found {len(sorted_files)} images")
            print(f"  Creating {actual_sequences} sequences...")

            for sequence_num in tqdm(
                range(actual_sequences), desc=f"  Sequences for '{action}'", ncols=70
            ):

                start_index = sequence_num * SEQUENCE_LENGTH
                end_index = start_index + SEQUENCE_LENGTH
                sequence_files = sorted_files[start_index:end_index]

                for frame_num, filename in enumerate(sequence_files):
                    image_path = os.path.join(action_path, filename)

                    frame = cv2.imread(image_path)

                    if frame is None:
                        print(f"\n    ✗ Could not read: {image_path}")
                        stats["failed_frames"] += 1
                        continue

                    # Make detections with Hands
                    image, results = mediapipe_detection(frame, hands)

                    # Extract keypoints using hands-only method
                    keypoints = extract_keypoints_hands_normalized(results)

                    # Save keypoints
                    save_sequence(keypoints, action, sequence_num, frame_num)

                    stats["total_frames"] += 1

                    if visualize:
                        from src.utils.mediapipe_utils import draw_hand_landmarks

                        draw_hand_landmarks(image, results)
                        cv2.imshow(f"Processing: {action}", image)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            visualize = False
                            cv2.destroyAllWindows()

                stats["total_sequences"] += 1

            print(f"  ✓ Completed '{action}': {actual_sequences} sequences processed")

    if visualize:
        cv2.destroyAllWindows()

    stats["end_time"] = time.time()
    stats["duration"] = stats["end_time"] - stats["start_time"]
    stats["frames_per_second"] = (
        stats["total_frames"] / stats["duration"] if stats["duration"] > 0 else 0
    )

    return stats


def print_statistics(stats):
    """Print processing statistics."""
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETED")
    print("=" * 70)
    print(
        f"Total actions processed: {stats['total_actions'] - stats['skipped_actions']}/{stats['total_actions']}"
    )
    print(f"Total sequences created: {stats['total_sequences']}")
    print(f"Total frames processed: {stats['total_frames']}")
    print(f"Failed frames: {stats['failed_frames']}")
    print(
        f"Duration: {stats['duration']:.2f} seconds ({stats['duration']/60:.2f} minutes)"
    )
    print(f"Processing speed: {stats['frames_per_second']:.2f} frames/second")
    print("=" * 70)


def main():
    """Main preprocessing pipeline."""
    print("\n" + "🚀 " * 35)
    print("SIGN LANGUAGE DATA PREPROCESSING (HANDS ONLY)")
    print("🚀 " * 35 + "\n")

    print(f"This will process images from: {RAW_DATA_DIR}")
    print(f"Output will be saved to: {PROCESSED_DATA_DIR}")
    print(
        f"\nTotal images to process: ~{len(ACTIONS) * NO_SEQUENCES * SEQUENCE_LENGTH}"
    )

    response = input("\nProceed with preprocessing? (y/n): ").lower().strip()

    if response != "y":
        print("Preprocessing cancelled.")
        return

    visualize = input("Enable visualization? (slower) (y/n): ").lower().strip() == "y"

    print("\n" + "⏳ Starting preprocessing... This may take 15-30 minutes...\n")

    try:
        stats = process_images_to_sequences(visualize=visualize)
        print_statistics(stats)

        print("\n" + "=" * 70)
        print("VERIFYING PROCESSED DATA")
        print("=" * 70)
        verify_data_integrity()

        print("\n✅ Preprocessing completed successfully!")
        print(f"📁 Processed data saved to: {PROCESSED_DATA_DIR}")
        print("\n⚠ IMPORTANT: Now retrain your model with:")
        print("   python -m src.train_model")

    except KeyboardInterrupt:
        print("\n\n⚠ Preprocessing interrupted by user!")
    except Exception as e:
        print(f"\n\n✗ Error during preprocessing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
