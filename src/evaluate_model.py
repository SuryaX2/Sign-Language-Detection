"""
Model evaluation script for testing saved models.
Loads a trained model and evaluates it on test data.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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


def evaluate_model_on_test_data(model_path=None, visualize=True):
    """
    Evaluate a trained model on test data.

    Args:
        model_path: Path to saved model (default: from config)
        visualize: Whether to generate visualizations

    Returns:
        Dictionary with evaluation metrics
    """
    model_path = model_path or MODEL_PATH

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
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

    # Calculate metrics
    print("\n" + "=" * 70)
    print("EVALUATION METRICS")
    print("=" * 70)

    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    print(f"\n✓ Overall Accuracy: {accuracy * 100:.2f}%")
    print(
        f"✓ Correct Predictions: {np.sum(y_pred_labels == y_true_labels)}/{len(y_true_labels)}"
    )

    # Classification report
    print("\n" + "-" * 70)
    print("CLASSIFICATION REPORT")
    print("-" * 70)

    report = classification_report(
        y_true_labels, y_pred_labels, target_names=ACTIONS, zero_division=0
    )
    print(report)

    # Per-class accuracy
    print("\n" + "-" * 70)
    print("PER-CLASS ACCURACY")
    print("-" * 70)

    class_accuracies = {}
    for idx, action in enumerate(ACTIONS):
        mask = y_true_labels == idx
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred_labels[mask] == y_true_labels[mask])
            class_accuracies[action] = class_acc
            print(f"  '{action}': {class_acc * 100:.2f}% ({np.sum(mask)} samples)")

    # Visualizations
    if visualize:
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)

        # Confusion matrix
        plot_confusion_matrix(y_true_labels, y_pred_labels, ACTIONS)

        # Prediction confidence distribution
        plot_confidence_distribution(y_pred_probs, y_pred_labels, y_true_labels)

        # Per-class accuracy bar chart
        plot_class_accuracies(class_accuracies)

    # Summary
    results = {
        "accuracy": accuracy,
        "total_samples": len(y_test),
        "correct_predictions": np.sum(y_pred_labels == y_true_labels),
        "class_accuracies": class_accuracies,
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


def plot_class_accuracies(class_accuracies):
    """Plot per-class accuracy as bar chart."""
    actions = list(class_accuracies.keys())
    accuracies = [class_accuracies[action] * 100 for action in actions]

    plt.figure(figsize=(16, 8))
    colors = [
        "green" if acc >= 80 else "orange" if acc >= 60 else "red" for acc in accuracies
    ]

    bars = plt.bar(actions, accuracies, color=colors, edgecolor="black", linewidth=1.2)

    plt.xlabel("Sign Language Action", fontsize=14, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=14, fontweight="bold")
    plt.title("Per-Class Accuracy", fontsize=16, fontweight="bold", pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Add horizontal line at 90% accuracy
    plt.axhline(
        y=90, color="blue", linestyle="--", linewidth=2, alpha=0.5, label="90% Target"
    )
    plt.legend(fontsize=10)

    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "class_accuracies.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Class accuracies chart saved to: {save_path}")
    plt.close()


def compare_models(model_paths, model_names):
    """
    Compare multiple trained models.

    Args:
        model_paths: List of paths to saved models
        model_names: List of names for each model

    Returns:
        Dictionary with comparison results
    """
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    results = {}

    for model_path, model_name in zip(model_paths, model_names):
        print(f"\nEvaluating: {model_name}")
        print("-" * 70)

        result = evaluate_model_on_test_data(model_path, visualize=False)

        if result is not None:
            results[model_name] = result
            print(f"✓ {model_name}: {result['accuracy'] * 100:.2f}% accuracy")

    # Compare results
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)

        for name, result in results.items():
            print(f"{name}: {result['accuracy'] * 100:.2f}%")

        best_model = max(results.items(), key=lambda x: x[1]["accuracy"])
        print(
            f"\n✓ Best Model: {best_model[0]} with {best_model[1]['accuracy'] * 100:.2f}% accuracy"
        )

    return results


def main():
    """Main entry point for evaluation script."""
    print("\n" + "🔍 " * 35)
    print("SIGN LANGUAGE MODEL EVALUATION")
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
