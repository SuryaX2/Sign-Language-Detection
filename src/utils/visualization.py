"""
Visualization utilities for sign language detection project.
Provides functions for visualizing data, predictions, and model analysis.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation

from src.config import PROCESSED_DATA_DIR, RESULTS_DIR, ACTIONS, SEQUENCE_LENGTH


def visualize_sequence(action, sequence_idx, save=False):
    """
    Visualize a single sequence by plotting keypoint trajectories.

    Args:
        action: Action label
        sequence_idx: Sequence index
        save: Whether to save the visualization
    """
    sequence_path = os.path.join(PROCESSED_DATA_DIR, action, str(sequence_idx))

    if not os.path.exists(sequence_path):
        print(f"✗ Sequence not found: {sequence_path}")
        return

    # Load sequence frames
    frames = []
    for frame_num in range(SEQUENCE_LENGTH):
        frame_path = os.path.join(sequence_path, f"{frame_num}.npy")
        if os.path.exists(frame_path):
            frames.append(np.load(frame_path))

    if len(frames) == 0:
        print(f"✗ No frames found in sequence")
        return

    frames = np.array(frames)

    # Extract hand keypoints (last 126 values = 2 hands × 21 landmarks × 3 coordinates)
    left_hand = frames[:, -126:-63].reshape(SEQUENCE_LENGTH, 21, 3)
    right_hand = frames[:, -63:].reshape(SEQUENCE_LENGTH, 21, 3)

    # Create visualization
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 2, figure=fig)

    # Left hand trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    for landmark_idx in range(21):
        ax1.plot(
            left_hand[:, landmark_idx, 0],
            left_hand[:, landmark_idx, 1],
            alpha=0.6,
            linewidth=1,
        )
    ax1.set_title(f"Left Hand Trajectory - {action}", fontsize=12, fontweight="bold")
    ax1.set_xlabel("X coordinate")
    ax1.set_ylabel("Y coordinate")
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)

    # Right hand trajectory
    ax2 = fig.add_subplot(gs[0, 1])
    for landmark_idx in range(21):
        ax2.plot(
            right_hand[:, landmark_idx, 0],
            right_hand[:, landmark_idx, 1],
            alpha=0.6,
            linewidth=1,
        )
    ax2.set_title(f"Right Hand Trajectory - {action}", fontsize=12, fontweight="bold")
    ax2.set_xlabel("X coordinate")
    ax2.set_ylabel("Y coordinate")
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)

    # Left hand Z-axis movement
    ax3 = fig.add_subplot(gs[1, 0])
    for landmark_idx in range(21):
        ax3.plot(
            range(SEQUENCE_LENGTH),
            left_hand[:, landmark_idx, 2],
            alpha=0.6,
            linewidth=1,
        )
    ax3.set_title("Left Hand Depth (Z-axis) Over Time", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Frame")
    ax3.set_ylabel("Z coordinate")
    ax3.grid(True, alpha=0.3)

    # Right hand Z-axis movement
    ax4 = fig.add_subplot(gs[1, 1])
    for landmark_idx in range(21):
        ax4.plot(
            range(SEQUENCE_LENGTH),
            right_hand[:, landmark_idx, 2],
            alpha=0.6,
            linewidth=1,
        )
    ax4.set_title("Right Hand Depth (Z-axis) Over Time", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Frame")
    ax4.set_ylabel("Z coordinate")
    ax4.grid(True, alpha=0.3)

    plt.suptitle(
        f"Sequence Visualization: {action} (Sequence {sequence_idx})",
        fontsize=14,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()

    if save:
        save_path = os.path.join(RESULTS_DIR, f"sequence_{action}_{sequence_idx}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Visualization saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_all_actions_sample(num_sequences=3, save=True):
    """
    Visualize sample sequences from all actions.

    Args:
        num_sequences: Number of sequences to visualize per action
        save: Whether to save visualizations
    """
    print("\n" + "=" * 70)
    print("VISUALIZING SAMPLE SEQUENCES")
    print("=" * 70)

    for action in ACTIONS:
        print(f"Visualizing '{action}'...")
        for seq_idx in range(min(num_sequences, 3)):
            visualize_sequence(action, seq_idx, save=save)

    print("=" * 70)
    print(f"✓ Visualizations completed!")


def plot_class_distribution(save=True):
    """
    Plot distribution of sequences across all action classes.

    Args:
        save: Whether to save the plot
    """
    print("\n" + "=" * 70)
    print("ANALYZING CLASS DISTRIBUTION")
    print("=" * 70)

    # Count sequences per action
    action_counts = {}
    for action in ACTIONS:
        action_path = os.path.join(PROCESSED_DATA_DIR, action)
        if os.path.exists(action_path):
            count = len(
                [
                    d
                    for d in os.listdir(action_path)
                    if os.path.isdir(os.path.join(action_path, d))
                ]
            )
            action_counts[action] = count
        else:
            action_counts[action] = 0

    # Create bar plot
    plt.figure(figsize=(16, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(ACTIONS)))

    bars = plt.bar(
        action_counts.keys(),
        action_counts.values(),
        color=colors,
        edgecolor="black",
        linewidth=1.2,
    )

    plt.xlabel("Action", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Sequences", fontsize=14, fontweight="bold")
    plt.title(
        "Dataset Distribution Across Actions", fontsize=16, fontweight="bold", pad=20
    )
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, count in zip(bars, action_counts.values()):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(count)}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Add statistics
    total = sum(action_counts.values())
    avg = total / len(action_counts) if len(action_counts) > 0 else 0
    plt.text(
        0.02,
        0.98,
        f"Total: {total}\nAverage: {avg:.1f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save:
        save_path = os.path.join(RESULTS_DIR, "class_distribution.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Class distribution saved to: {save_path}")
    else:
        plt.show()

    plt.close()

    # Print statistics
    print(f"\nTotal sequences: {total}")
    print(f"Average per class: {avg:.1f}")
    print(
        f"Min: {min(action_counts.values())} ({min(action_counts, key=action_counts.get)})"
    )
    print(
        f"Max: {max(action_counts.values())} ({max(action_counts, key=action_counts.get)})"
    )
    print("=" * 70)


def visualize_keypoint_variance(action, num_sequences=10, save=True):
    """
    Visualize variance in keypoint positions across sequences of the same action.

    Args:
        action: Action label to analyze
        num_sequences: Number of sequences to compare
        save: Whether to save the plot
    """
    print(f"\nAnalyzing variance for action: '{action}'")

    # Load sequences
    all_sequences = []
    for seq_idx in range(num_sequences):
        sequence_path = os.path.join(PROCESSED_DATA_DIR, action, str(seq_idx))
        if not os.path.exists(sequence_path):
            continue

        frames = []
        for frame_num in range(SEQUENCE_LENGTH):
            frame_path = os.path.join(sequence_path, f"{frame_num}.npy")
            if os.path.exists(frame_path):
                frames.append(np.load(frame_path))

        if len(frames) == SEQUENCE_LENGTH:
            all_sequences.append(np.array(frames))

    if len(all_sequences) == 0:
        print(f"✗ No sequences found for action '{action}'")
        return

    all_sequences = np.array(
        all_sequences
    )  # Shape: (num_sequences, SEQUENCE_LENGTH, features)

    # Calculate mean and std across sequences
    mean_sequence = np.mean(all_sequences, axis=0)
    std_sequence = np.std(all_sequences, axis=0)

    # Extract hand keypoints
    hand_features = mean_sequence[:, -126:]  # Last 126 features (both hands)
    hand_std = std_sequence[:, -126:]

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Mean values
    im1 = axes[0].imshow(hand_features.T, aspect="auto", cmap="viridis")
    axes[0].set_title(
        f"Mean Hand Keypoints Over Time - {action}", fontsize=12, fontweight="bold"
    )
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Keypoint Feature")
    plt.colorbar(im1, ax=axes[0], label="Value")

    # Standard deviation
    im2 = axes[1].imshow(hand_std.T, aspect="auto", cmap="hot")
    axes[1].set_title(
        f"Standard Deviation of Hand Keypoints - {action}",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Keypoint Feature")
    plt.colorbar(im2, ax=axes[1], label="Std Dev")

    plt.tight_layout()

    if save:
        save_path = os.path.join(RESULTS_DIR, f"variance_{action}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Variance plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def create_prediction_heatmap(predictions, true_labels, actions, save=True):
    """
    Create a normalized heatmap showing prediction patterns.

    Args:
        predictions: Array of predicted labels
        true_labels: Array of true labels
        actions: List of action names
        save: Whether to save the plot
    """
    from sklearn.metrics import confusion_matrix

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    # Normalize by row (true labels)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create heatmap
    plt.figure(figsize=(18, 15))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        xticklabels=actions,
        yticklabels=actions,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Proportion"},
    )

    plt.title(
        "Normalized Confusion Matrix (Row-wise)", fontsize=16, fontweight="bold", pad=20
    )
    plt.ylabel("True Label", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save:
        save_path = os.path.join(RESULTS_DIR, "normalized_confusion_matrix.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Normalized confusion matrix saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_misclassifications(y_true, y_pred, actions, top_n=10, save=True):
    """
    Identify and visualize most common misclassifications.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        actions: List of action names
        top_n: Number of top misclassifications to show
        save: Whether to save the plot
    """
    from collections import Counter

    # Find misclassifications
    misclassified = [(actions[t], actions[p]) for t, p in zip(y_true, y_pred) if t != p]

    if len(misclassified) == 0:
        print("✓ No misclassifications found!")
        return

    # Count most common
    common_errors = Counter(misclassified).most_common(top_n)

    # Prepare data
    labels = [f"{true} → {pred}" for (true, pred), count in common_errors]
    counts = [count for _, count in common_errors]

    # Plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(counts)))

    bars = plt.barh(labels, counts, color=colors, edgecolor="black", linewidth=1.2)

    plt.xlabel("Number of Misclassifications", fontsize=12, fontweight="bold")
    plt.ylabel("True → Predicted", fontsize=12, fontweight="bold")
    plt.title(
        f"Top {top_n} Most Common Misclassifications", fontsize=14, fontweight="bold"
    )
    plt.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for bar, count in zip(bars, counts):
        width = bar.get_width()
        plt.text(
            width,
            bar.get_y() + bar.get_height() / 2.0,
            f" {int(count)}",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()

    if save:
        save_path = os.path.join(RESULTS_DIR, "top_misclassifications.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Misclassifications plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()

    # Print details
    print("\n" + "=" * 70)
    print(f"TOP {top_n} MISCLASSIFICATIONS")
    print("=" * 70)
    for (true, pred), count in common_errors:
        print(f"  {true} → {pred}: {count} times")
    print("=" * 70)


def main():
    """Main function to demonstrate visualization capabilities."""
    print("\n" + "🎨 " * 35)
    print("SIGN LANGUAGE VISUALIZATION UTILITIES")
    print("🎨 " * 35 + "\n")

    print("Available visualizations:")
    print("  1. Sequence trajectories")
    print("  2. Class distribution")
    print("  3. Keypoint variance")
    print("  4. Sample sequences for all actions")
    print("\nGenerating visualizations...\n")

    # Create results directory if needed
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Generate visualizations
    try:
        # Class distribution
        plot_class_distribution(save=True)

        # Sample sequences
        print("\nVisualizing sample sequences (this may take a moment)...")
        visualize_sequence(ACTIONS[0], 0, save=True)

        # Variance analysis
        print("\nAnalyzing keypoint variance...")
        visualize_keypoint_variance(ACTIONS[0], num_sequences=10, save=True)

        print("\n✅ Visualizations completed!")
        print(f"📁 Results saved to: {RESULTS_DIR}")

    except Exception as e:
        print(f"\n✗ Error generating visualizations: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
