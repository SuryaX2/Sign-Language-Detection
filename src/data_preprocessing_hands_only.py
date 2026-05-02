"""
Data preprocessing - uses ALL available images with shuffled sequence construction.
"""

import os
import cv2
import time
import random
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
    RANDOM_STATE,
)

from src.utils.mediapipe_utils import (
    mediapipe_detection,
    extract_keypoints_hands_normalized,
    HandsDetector,
)

from src.utils.data_utils import save_sequence, verify_data_integrity


def get_all_image_files(action_path):
    """
    Get ALL image files from action folder, sorted numerically.
    """
    all_files = os.listdir(action_path)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [
        f for f in all_files if os.path.splitext(f.lower())[1] in image_extensions
    ]

    try:
        sorted_files = sorted(image_files, key=lambda f: int(os.path.splitext(f)[0]))
    except ValueError:
        sorted_files = sorted(image_files)

    return sorted_files


def compute_max_sequences(num_images, sequence_length):
    """
    Compute how many non-overlapping sequences we can make from available images.
    """
    return num_images // sequence_length


def create_output_directories(no_sequences):
    """Create output directory structure."""
    print("Creating output directory structure...")
    for action in ACTIONS:
        for sequence in range(no_sequences):
            sequence_dir = os.path.join(PROCESSED_DATA_DIR, action, str(sequence))
            os.makedirs(sequence_dir, exist_ok=True)
    print(
        f"✓ Created directories for {len(ACTIONS)} actions × {no_sequences} sequences"
    )


def process_images_to_sequences(
    visualize=False,
    limit_actions=None,
    shuffle_before_grouping=True,
    use_all_images=True,
):
    """
    Process ALL raw images and extract keypoints using MediaPipe Hands.

    Args:
        visualize: Show images with landmarks while processing
        limit_actions: Process only specific actions (None = all)
        shuffle_before_grouping: Shuffle images before making sequences
                                 so each sequence has diverse static views
        use_all_images: Use ALL available images (ignores NO_SEQUENCES from config)
    """
    actions_to_process = limit_actions if limit_actions else ACTIONS
    random.seed(RANDOM_STATE)

    print("=" * 70)
    print("STARTING DATA PREPROCESSING (HANDS ONLY — FULL DATASET)")
    print("=" * 70)
    print(f"Source         : {RAW_DATA_DIR}")
    print(f"Output         : {PROCESSED_DATA_DIR}")
    print(f"Shuffle images : {shuffle_before_grouping}")
    print(f"Use all images : {use_all_images}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print("=" * 70)

    stats = {
        "total_actions": len(actions_to_process),
        "total_sequences": 0,
        "total_frames": 0,
        "skipped_actions": 0,
        "failed_frames": 0,
        "start_time": time.time(),
    }

    # --- First pass: figure out max sequences per action ---
    action_image_counts = {}
    for action in actions_to_process:
        action_path = os.path.join(RAW_DATA_DIR, action)
        if os.path.exists(action_path):
            files = get_all_image_files(action_path)
            action_image_counts[action] = len(files)
        else:
            action_image_counts[action] = 0

    if use_all_images:
        # Use the minimum across all actions so dataset stays balanced
        valid_counts = [c for c in action_image_counts.values() if c > 0]
        if not valid_counts:
            print("✗ No images found in any action folder!")
            return stats

        min_images = min(valid_counts)
        max_possible_sequences = min_images // SEQUENCE_LENGTH

        print(f"\nImage counts per action:")
        for action, count in action_image_counts.items():
            seq_possible = count // SEQUENCE_LENGTH
            print(f"  '{action}': {count} images → {seq_possible} possible sequences")

        print(f"\n✓ Minimum images across classes : {min_images}")
        print(f"✓ Sequences per class (balanced): {max_possible_sequences}")
        print(
            f"✓ Total sequences                : {max_possible_sequences * len(actions_to_process)}"
        )
        print(
            f"✓ Total frames to process        : {max_possible_sequences * SEQUENCE_LENGTH * len(actions_to_process)}"
        )

        actual_no_sequences = max_possible_sequences
    else:
        from src.config import NO_SEQUENCES

        actual_no_sequences = NO_SEQUENCES

    # Create directories
    create_output_directories(actual_no_sequences)

    # --- Main processing loop ---
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

            all_files = get_all_image_files(action_path)

            if len(all_files) == 0:
                print(f"  ✗ No image files found")
                stats["skipped_actions"] += 1
                continue

            # Shuffle before grouping if requested
            files_to_use = all_files.copy()
            if shuffle_before_grouping:
                random.shuffle(files_to_use)

            # Trim to exactly actual_no_sequences × SEQUENCE_LENGTH images
            # so all classes are balanced
            total_images_needed = actual_no_sequences * SEQUENCE_LENGTH
            files_to_use = files_to_use[:total_images_needed]

            print(f"  Total images available : {len(all_files)}")
            print(f"  Images being used      : {len(files_to_use)}")
            print(f"  Sequences to create    : {actual_no_sequences}")
            print(f"  Shuffled               : {shuffle_before_grouping}")

            for sequence_num in tqdm(
                range(actual_no_sequences),
                desc=f"  '{action}'",
                ncols=70,
            ):
                start_idx = sequence_num * SEQUENCE_LENGTH
                end_idx = start_idx + SEQUENCE_LENGTH
                sequence_files = files_to_use[start_idx:end_idx]

                for frame_num, filename in enumerate(sequence_files):
                    image_path = os.path.join(action_path, filename)
                    frame = cv2.imread(image_path)

                    if frame is None:
                        print(f"\n    ✗ Could not read: {image_path}")
                        stats["failed_frames"] += 1
                        # Save zeros so sequence stays complete
                        from src.config import TOTAL_KEYPOINTS
                        save_sequence(np.zeros(TOTAL_KEYPOINTS), action, sequence_num, frame_num)
                        continue

                    image, results = mediapipe_detection(frame, hands)
                    keypoints = extract_keypoints_hands_normalized(results)
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

            print(f"  ✓ Done '{action}': {actual_no_sequences} sequences")

    if visualize:
        cv2.destroyAllWindows()

    stats["end_time"] = time.time()
    stats["duration"] = stats["end_time"] - stats["start_time"]
    stats["frames_per_second"] = (
        stats["total_frames"] / stats["duration"] if stats["duration"] > 0 else 0
    )
    stats["actual_no_sequences"] = actual_no_sequences

    return stats


def print_statistics(stats):
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETED")
    print("=" * 70)
    print(
        f"Actions processed : {stats['total_actions'] - stats['skipped_actions']}/{stats['total_actions']}"
    )
    print(f"Sequences created : {stats['total_sequences']}")
    print(f"Frames processed  : {stats['total_frames']}")
    print(f"Failed frames     : {stats['failed_frames']}")
    print(
        f"Duration          : {stats['duration']:.2f}s ({stats['duration']/60:.2f} min)"
    )
    print(f"Speed             : {stats['frames_per_second']:.2f} frames/sec")
    print("=" * 70)


def main():
    print("\n" + "🚀 " * 20)
    print("SIGN LANGUAGE PREPROCESSING — FULL DATASET")
    print("🚀 " * 20 + "\n")

    print(f"Source : {RAW_DATA_DIR}")
    print(f"Output : {PROCESSED_DATA_DIR}")
    print(f"\nThis will use ALL available images from the dataset.")
    print(f"Sequence length: {SEQUENCE_LENGTH} frames per sequence")

    response = input("\nProceed? (y/n): ").lower().strip()
    if response != "y":
        print("Cancelled.")
        return

    shuffle = (
        input("Shuffle images before grouping? (recommended) (y/n): ").lower().strip()
        == "y"
    )
    visualize = input("Enable visualization? (slower) (y/n): ").lower().strip() == "y"

    try:
        stats = process_images_to_sequences(
            visualize=visualize,
            shuffle_before_grouping=shuffle,
            use_all_images=True,
        )
        print_statistics(stats)

        print("\n" + "=" * 70)
        print("VERIFYING DATA")
        print("=" * 70)
        verify_data_integrity(no_sequences=stats.get("actual_no_sequences"))

        print("\n✅ Done!")
        print(f"📁 Data saved to: {PROCESSED_DATA_DIR}")
        print("\n⚠ Now update config.py:")
        print(f"   NO_SEQUENCES = {stats.get('actual_no_sequences')}")
        print("   Then retrain: python -m src.train_model")

    except KeyboardInterrupt:
        print("\n⚠ Interrupted!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
