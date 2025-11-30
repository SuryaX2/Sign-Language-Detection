"""
Model evaluation script with detailed TP/FP/FN/TN metrics.
Loads a trained model and evaluates it on test data.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.config import (
    MODEL_PATH,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    ACTIONS,
    NO_SEQUENCES,
    SEQUENCE_LENGTH,
    VALIDATION_SPLIT,
    RANDOM_STATE,
)

from src.models.transformer_model import load_trained_model
from src.utils.data_utils import load_sequences, split_data


def calculate_tp_fp_fn_tn(y_true, y_pred, num_classes):
    """
    Calculate TP, FP, FN, TN for each class.

    Args:
        y_true: True labels (1D array)
        y_pred: Predicted labels (1D array)
        num_classes: Number of classes

    Returns:
        Dictionary with metrics for each class
    """
    metrics = {}

    for class_idx in range(num_classes):
        # True Positives: Predicted as class AND actually is class
        tp = np.sum((y_pred == class_idx) & (y_true == class_idx))

        # False Positives: Predicted as class BUT actually is NOT class
        fp = np.sum((y_pred == class_idx) & (y_true != class_idx))

        # False Negatives: Predicted as NOT class BUT actually IS class
        fn = np.sum((y_pred != class_idx) & (y_true == class_idx))

        # True Negatives: Predicted as NOT class AND actually is NOT class
        tn = np.sum((y_pred != class_idx) & (y_true != class_idx))

        # Calculate additional metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics[class_idx] = {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "specificity": specificity,
            "support": int(tp + fn),
        }

    return metrics


def print_detailed_metrics(metrics, actions):
    """
    Print detailed metrics table with TP/FP/FN/TN.

    Args:
        metrics: Dictionary of metrics per class
        actions: List of action names
    """
    print("\n" + "=" * 120)
    print("DETAILED METRICS PER CLASS (TP/FP/FN/TN)")
    print("=" * 120)

    # Header
    header = f"{'Class':<8} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Specificity':>12} {'Support':>8}"
    print(header)
    print("-" * 120)

    # Print each class
    for idx, action in enumerate(actions):
        m = metrics[idx]
        row = (
            f"{action:<8} "
            f"{m['tp']:>6} "
            f"{m['fp']:>6} "
            f"{m['fn']:>6} "
            f"{m['tn']:>6} "
            f"{m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} "
            f"{m['f1_score']:>10.4f} "
            f"{m['specificity']:>12.4f} "
            f"{m['support']:>8}"
        )
        print(row)

    print("=" * 120)

    # Calculate overall metrics
    total_tp = sum(m["tp"] for m in metrics.values())
    total_fp = sum(m["fp"] for m in metrics.values())
    total_fn = sum(m["fn"] for m in metrics.values())
    total_tn = sum(m["tn"] for m in metrics.values())

    print("\nOVERALL STATISTICS:")
    print(f"  Total True Positives:  {total_tp}")
    print(f"  Total False Positives: {total_fp}")
    print(f"  Total False Negatives: {total_fn}")
    print(f"  Total True Negatives:  {total_tn}")
    print("=" * 120)


def save_metrics_to_csv(metrics, actions, save_path):
    """
    Save detailed metrics to CSV file.

    Args:
        metrics: Dictionary of metrics per class
        actions: List of action names
        save_path: Path to save CSV
    """
    data = []
    for idx, action in enumerate(actions):
        m = metrics[idx]
        data.append(
            {
                "Class": action,
                "TP": m["tp"],
                "FP": m["fp"],
                "FN": m["fn"],
                "TN": m["tn"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1-Score": m["f1_score"],
                "Specificity": m["specificity"],
                "Support": m["support"],
            }
        )

    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"✓ Detailed metrics saved to: {save_path}")


def plot_tp_fp_fn_tn_chart(metrics, actions):
    """
    Create visualization of TP/FP/FN/TN for each class.

    Args:
        metrics: Dictionary of metrics per class
        actions: List of action names
    """
    # Prepare data
    tp_values = [metrics[i]["tp"] for i in range(len(actions))]
    fp_values = [metrics[i]["fp"] for i in range(len(actions))]
    fn_values = [metrics[i]["fn"] for i in range(len(actions))]
    tn_values = [metrics[i]["tn"] for i in range(len(actions))]

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(20, 10))

    x = np.arange(len(actions))
    width = 0.6

    # Create bars
    p1 = ax.bar(x, tp_values, width, label="True Positive (TP)", color="#2ecc71")
    p2 = ax.bar(
        x,
        fp_values,
        width,
        bottom=tp_values,
        label="False Positive (FP)",
        color="#e74c3c",
    )

    # Add FN on top of FP
    bottom_fn = [tp_values[i] + fp_values[i] for i in range(len(actions))]
    p3 = ax.bar(
        x,
        fn_values,
        width,
        bottom=bottom_fn,
        label="False Negative (FN)",
        color="#f39c12",
    )

    ax.set_xlabel("Action Class", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count", fontsize=14, fontweight="bold")
    ax.set_title(
        "True Positive, False Positive, False Negative per Class",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(actions, rotation=45, ha="right")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "tp_fp_fn_tn_chart.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ TP/FP/FN/TN chart saved to: {save_path}")
    plt.close()


def plot_precision_recall_chart(metrics, actions):
    """
    Create precision and recall comparison chart.

    Args:
        metrics: Dictionary of metrics per class
        actions: List of action names
    """
    precision_values = [metrics[i]["precision"] for i in range(len(actions))]
    recall_values = [metrics[i]["recall"] for i in range(len(actions))]
    f1_values = [metrics[i]["f1_score"] for i in range(len(actions))]

    fig, ax = plt.subplots(figsize=(18, 8))

    x = np.arange(len(actions))
    width = 0.25

    bars1 = ax.bar(
        x - width,
        precision_values,
        width,
        label="Precision",
        color="#3498db",
        alpha=0.8,
    )
    bars2 = ax.bar(x, recall_values, width, label="Recall", color="#e74c3c", alpha=0.8)
    bars3 = ax.bar(
        x + width, f1_values, width, label="F1-Score", color="#2ecc71", alpha=0.8
    )

    ax.set_xlabel("Action Class", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score", fontsize=14, fontweight="bold")
    ax.set_title(
        "Precision, Recall, and F1-Score per Class",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(actions, rotation=45, ha="right")
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    # Add horizontal line at 0.9
    ax.axhline(
        y=0.9, color="green", linestyle="--", linewidth=2, alpha=0.5, label="90% Target"
    )

    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "precision_recall_f1_chart.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Precision/Recall/F1 chart saved to: {save_path}")
    plt.close()


def evaluate_model_on_test_data(model_path=None, visualize=True):
    """
    Evaluate a trained model on test data with detailed TP/FP/FN/TN metrics.

    Args:
        model_path: Path to saved model (default: from config)
        visualize: Whether to generate visualizations

    Returns:
        Dictionary with evaluation metrics
    """
    model_path = model_path or MODEL_PATH

    print("\n" + "=" * 70)
    print("MODEL EVALUATION WITH DETAILED METRICS")
    print("=" * 70)

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"✗ Model not found at: {model_path}")
        print("  Please train a model first: python -m src.train_model")
        return None

    # Load model
    print(f"\nLoading model from: {model_path}")
    try:
        model = load_trained_model(model_path)
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

    # Load data
    print("\n" + "=" * 70)
    print("LOADING TEST DATA")
    print("=" * 70)

    try:
        X, y, label_map = load_sequences(
            data_path=PROCESSED_DATA_DIR,
            actions=ACTIONS,
            no_sequences=NO_SEQUENCES,
            sequence_length=SEQUENCE_LENGTH,
        )
    except ValueError as e:
        print(f"\n✗ Error loading data: {e}")
        return None

    # Split data (use same split as training)
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE, stratify=True
    )

    print(f"Test set size: {len(X_test)} sequences")

    # Make predictions
    print("\n" + "=" * 70)
    print("GENERATING PREDICTIONS")
    print("=" * 70)

    y_pred_probs = model.predict(X_test, verbose=1)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    # Calculate TP/FP/FN/TN metrics
    print("\n" + "=" * 70)
    print("CALCULATING DETAILED METRICS")
    print("=" * 70)

    metrics = calculate_tp_fp_fn_tn(y_true_labels, y_pred_labels, len(ACTIONS))

    # Print detailed metrics
    print_detailed_metrics(metrics, ACTIONS)

    # Calculate overall accuracy
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    print(f"\n✓ Overall Accuracy: {accuracy * 100:.2f}%")
    print(
        f"✓ Correct Predictions: {np.sum(y_pred_labels == y_true_labels)}/{len(y_true_labels)}"
    )

    # Classification report
    print("\n" + "-" * 70)
    print("STANDARD CLASSIFICATION REPORT")
    print("-" * 70)

    report = classification_report(
        y_true_labels, y_pred_labels, target_names=ACTIONS, zero_division=0
    )
    print(report)

    # Save detailed metrics to CSV
    csv_path = os.path.join(RESULTS_DIR, "detailed_metrics.csv")
    save_metrics_to_csv(metrics, ACTIONS, csv_path)

    # Visualizations
    if visualize:
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)

        # Confusion matrix
        plot_confusion_matrix(y_true_labels, y_pred_labels, ACTIONS)

        # TP/FP/FN/TN chart
        plot_tp_fp_fn_tn_chart(metrics, ACTIONS)

        # Precision/Recall/F1 chart
        plot_precision_recall_chart(metrics, ACTIONS)

        # Prediction confidence distribution
        plot_confidence_distribution(y_pred_probs, y_pred_labels, y_true_labels)

    # Summary
    results = {
        "accuracy": accuracy,
        "total_samples": len(y_test),
        "correct_predictions": np.sum(y_pred_labels == y_true_labels),
        "metrics": metrics,
        "y_true": y_true_labels,
        "y_pred": y_pred_labels,
        "y_pred_probs": y_pred_probs,
    }

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)

    return results


def plot_confusion_matrix(y_true, y_pred, actions):
    """Generate and save confusion matrix heatmap."""
    conf_matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(20, 16))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="RdYlGn",
        xticklabels=actions,
        yticklabels=actions,
        cbar_kws={"label": "Count"},
        linewidths=0.5,
        linecolor="gray",
    )

    plt.title(
        "Confusion Matrix - Model Evaluation", fontsize=20, fontweight="bold", pad=20
    )
    plt.ylabel("True Label", fontsize=16, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=16, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "evaluation_confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Confusion matrix saved to: {save_path}")
    plt.close()


def plot_confidence_distribution(y_pred_probs, y_pred, y_true):
    """Plot distribution of prediction confidences."""
    # Get confidence for predicted class
    confidences = np.max(y_pred_probs, axis=1)

    # Separate correct and incorrect predictions
    correct_mask = y_pred == y_true
    correct_confidences = confidences[correct_mask]
    incorrect_confidences = confidences[~correct_mask]

    plt.figure(figsize=(12, 6))

    plt.hist(
        correct_confidences,
        bins=30,
        alpha=0.7,
        label="Correct Predictions",
        color="green",
        edgecolor="black",
    )
    plt.hist(
        incorrect_confidences,
        bins=30,
        alpha=0.7,
        label="Incorrect Predictions",
        color="red",
        edgecolor="black",
    )

    plt.xlabel("Prediction Confidence", fontsize=12, fontweight="bold")
    plt.ylabel("Frequency", fontsize=12, fontweight="bold")
    plt.title("Prediction Confidence Distribution", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add mean lines
    plt.axvline(
        np.mean(correct_confidences),
        color="darkgreen",
        linestyle="--",
        linewidth=2,
        label=f"Correct Mean: {np.mean(correct_confidences):.3f}",
    )
    if len(incorrect_confidences) > 0:
        plt.axvline(
            np.mean(incorrect_confidences),
            color="darkred",
            linestyle="--",
            linewidth=2,
            label=f"Incorrect Mean: {np.mean(incorrect_confidences):.3f}",
        )

    plt.legend(fontsize=10)
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "confidence_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Confidence distribution saved to: {save_path}")
    plt.close()


def main():
    """Main entry point for evaluation script."""
    print("\n" + "🔍 " * 35)
    print("SIGN LANGUAGE MODEL EVALUATION WITH TP/FP/FN/TN")
    print("🔍 " * 35 + "\n")

    # Check for model
    if not os.path.exists(MODEL_PATH):
        print(f"✗ No trained model found at: {MODEL_PATH}")
        print("  Please train a model first: python -m src.train_model")
        return

    print(f"Model to evaluate: {MODEL_PATH}")

    response = input("\nProceed with evaluation? (y/n): ").lower().strip()

    if response != "y":
        print("Evaluation cancelled.")
        return

    # Run evaluation
    try:
        results = evaluate_model_on_test_data(MODEL_PATH, visualize=True)

        if results is not None:
            print("\n" + "✅ " * 35)
            print("EVALUATION COMPLETED SUCCESSFULLY!")
            print("✅ " * 35)
            print(f"\n📊 Final Accuracy: {results['accuracy'] * 100:.2f}%")
            print(f"📁 Results saved to: {RESULTS_DIR}")
            print(f"📄 Detailed metrics CSV: {RESULTS_DIR}/detailed_metrics.csv")
        else:
            print("\n⚠ Evaluation failed.")

    except KeyboardInterrupt:
        print("\n\n⚠ Evaluation interrupted by user!")
    except Exception as e:
        print(f"\n\n✗ Error during evaluation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
