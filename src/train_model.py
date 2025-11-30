"""
Optimized training script with data augmentation and better callbacks.
Targets >90% accuracy with faster training time.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import (
    TensorBoard,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
import tensorflow as tf

from src.config import (
    PROCESSED_DATA_DIR,
    SAVED_MODELS_DIR,
    MODEL_PATH,
    LOGS_DIR,
    RESULTS_DIR,
    ACTIONS,
    NO_SEQUENCES,
    SEQUENCE_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    VALIDATION_SPLIT,
    RANDOM_STATE,
)

from src.models.transformer_model import create_transformer_model
from src.utils.data_utils import load_sequences, split_data, get_action_statistics


# ============================================================================
# DATA AUGMENTATION FOR BETTER GENERALIZATION
# ============================================================================
def augment_sequence(sequence, noise_factor=0.01, scale_factor=0.05):
    """
    Apply data augmentation to improve model generalization.

    Args:
        sequence: Input sequence (sequence_length, features)
        noise_factor: Amount of Gaussian noise to add
        scale_factor: Random scaling factor

    Returns:
        Augmented sequence
    """
    augmented = sequence.copy()

    # Add Gaussian noise
    noise = np.random.normal(0, noise_factor, augmented.shape)
    augmented += noise

    # Random scaling
    scale = 1.0 + np.random.uniform(-scale_factor, scale_factor)
    augmented *= scale

    return augmented


def augment_dataset(X, y, augmentation_factor=2):
    """
    Augment entire dataset by creating variations.

    Args:
        X: Input data
        y: Labels
        augmentation_factor: How many augmented copies per sample

    Returns:
        Augmented X and y
    """
    X_augmented = [X]
    y_augmented = [y]

    print(f"Augmenting dataset with factor {augmentation_factor}...")

    for i in range(augmentation_factor):
        X_aug = np.array([augment_sequence(seq) for seq in X])
        X_augmented.append(X_aug)
        y_augmented.append(y)

    X_final = np.concatenate(X_augmented, axis=0)
    y_final = np.concatenate(y_augmented, axis=0)

    print(f"✓ Dataset augmented: {len(X)} → {len(X_final)} samples")

    return X_final, y_final


# ============================================================================
# OPTIMIZED CALLBACKS
# ============================================================================
def setup_optimized_callbacks(log_dir):
    """
    Setup optimized callbacks for faster convergence.

    Args:
        log_dir: Directory for TensorBoard logs

    Returns:
        List of Keras callbacks
    """
    callbacks = []

    # TensorBoard callback
    tb_callback = TensorBoard(
        log_dir=log_dir, histogram_freq=1, write_graph=True, update_freq="epoch"
    )
    callbacks.append(tb_callback)

    # Model checkpoint - save best model
    checkpoint_path = os.path.join(SAVED_MODELS_DIR, "best_model.h5")
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_categorical_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
        save_weights_only=False,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping - more aggressive for faster training
    early_stop_callback = EarlyStopping(
        monitor="val_categorical_accuracy",
        patience=15,  # Reduced from 20
        restore_best_weights=True,
        mode="max",
        verbose=1,
        min_delta=0.001,  # Stop if improvement < 0.1%
    )
    callbacks.append(early_stop_callback)

    # Reduce learning rate on plateau
    reduce_lr_callback = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,  # More aggressive reduction
        patience=7,  # Reduced from 10
        min_lr=1e-7,
        verbose=1,
        mode="min",
    )
    callbacks.append(reduce_lr_callback)

    return callbacks


# ============================================================================
# TRAINING PIPELINE
# ============================================================================
class TrainingLogger:
    """Handles logging and visualization of training progress."""

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.history = None

    def save_training_history(self, history):
        """Save training history for later analysis."""
        self.history = history.history

        # Save as numpy file
        history_path = os.path.join(self.log_dir, "training_history.npy")
        np.save(history_path, self.history)
        print(f"✓ Training history saved to: {history_path}")

    def plot_training_curves(self):
        """Plot and save training curves."""
        if self.history is None:
            print("⚠ No training history available")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot loss
        axes[0].plot(self.history["loss"], label="Training Loss", linewidth=2)
        axes[0].plot(self.history["val_loss"], label="Validation Loss", linewidth=2)
        axes[0].set_title("Model Loss Over Epochs", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Epoch", fontsize=12)
        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Plot accuracy
        axes[1].plot(
            self.history["categorical_accuracy"], label="Training Accuracy", linewidth=2
        )
        axes[1].plot(
            self.history["val_categorical_accuracy"],
            label="Validation Accuracy",
            linewidth=2,
        )
        axes[1].set_title("Model Accuracy Over Epochs", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_ylabel("Accuracy", fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        # Add 90% threshold line
        axes[1].axhline(
            y=0.90, color="r", linestyle="--", label="90% Target", alpha=0.7
        )
        axes[1].legend(fontsize=10)

        plt.tight_layout()

        # Save figure
        plot_path = os.path.join(RESULTS_DIR, "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"✓ Training curves saved to: {plot_path}")
        plt.close()


class ModelEvaluator:
    """Handles model evaluation and metrics generation."""

    def __init__(self, model, X_test, y_test, actions):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.actions = actions
        self.y_pred_probs = None
        self.y_pred_labels = None
        self.y_true_labels = None

    def predict(self):
        """Generate predictions on test data."""
        print("\n" + "=" * 70)
        print("GENERATING PREDICTIONS ON TEST DATA")
        print("=" * 70)

        self.y_pred_probs = self.model.predict(self.X_test, verbose=1)
        self.y_pred_labels = np.argmax(self.y_pred_probs, axis=1)
        self.y_true_labels = np.argmax(self.y_test, axis=1)

        print(f"✓ Predictions completed for {len(self.X_test)} samples")

    def print_classification_report(self):
        """Print detailed classification report."""
        print("\n" + "=" * 70)
        print("CLASSIFICATION REPORT")
        print("=" * 70)

        report = classification_report(
            self.y_true_labels,
            self.y_pred_labels,
            target_names=self.actions,
            zero_division=0,
        )
        print(report)

        # Save report to file
        report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write("CLASSIFICATION REPORT\n")
            f.write("=" * 70 + "\n")
            f.write(report)
        print(f"✓ Report saved to: {report_path}")

    def plot_confusion_matrix(self):
        """Generate and save confusion matrix heatmap."""
        print("\n" + "=" * 70)
        print("GENERATING CONFUSION MATRIX")
        print("=" * 70)

        conf_matrix = confusion_matrix(self.y_true_labels, self.y_pred_labels)

        # Create figure
        plt.figure(figsize=(20, 16))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.actions,
            yticklabels=self.actions,
            cbar_kws={"label": "Count"},
        )

        plt.title(
            "Confusion Matrix - Sign Language Detection",
            fontsize=20,
            fontweight="bold",
            pad=20,
        )
        plt.ylabel("True Label", fontsize=16, fontweight="bold")
        plt.xlabel("Predicted Label", fontsize=16, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save figure
        matrix_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
        plt.savefig(matrix_path, dpi=300, bbox_inches="tight")
        print(f"✓ Confusion matrix saved to: {matrix_path}")
        plt.close()

    def calculate_metrics(self):
        """Calculate and display overall metrics."""
        accuracy = np.mean(self.y_pred_labels == self.y_true_labels)

        print("\n" + "=" * 70)
        print("OVERALL METRICS")
        print("=" * 70)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        print(
            f"Correct Predictions: {np.sum(self.y_pred_labels == self.y_true_labels)}/{len(self.y_true_labels)}"
        )

        if accuracy >= 0.90:
            print("✅ TARGET ACHIEVED: Accuracy >= 90%!")
        else:
            print(f"⚠️ Need {(0.90 - accuracy) * 100:.2f}% more for 90% target")

        print("=" * 70)

        return accuracy

    def evaluate_all(self):
        """Run complete evaluation pipeline."""
        self.predict()
        self.print_classification_report()
        self.plot_confusion_matrix()
        accuracy = self.calculate_metrics()
        return accuracy


def train_model(use_augmentation=True):
    """
    Optimized training pipeline with data augmentation.

    Args:
        use_augmentation: Whether to use data augmentation

    Returns:
        Trained model and evaluation metrics
    """
    print("\n" + "🚀 " * 35)
    print("OPTIMIZED SIGN LANGUAGE MODEL TRAINING")
    print("🚀 " * 35 + "\n")

    # Load data
    print("=" * 70)
    print("LOADING PREPROCESSED DATA")
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
        print("Please run data preprocessing first: python -m src.data_preprocessing")
        return None, None

    # Apply data augmentation
    if use_augmentation and len(X) > 0:
        print("\n" + "=" * 70)
        print("APPLYING DATA AUGMENTATION")
        print("=" * 70)
        X, y = augment_dataset(X, y, augmentation_factor=2)

    # Print data statistics
    print("\n" + "=" * 70)
    print("DATA STATISTICS")
    print("=" * 70)
    action_stats = get_action_statistics(y, ACTIONS)
    for action, count in action_stats.items():
        print(f"  '{action}': {count} sequences")
    print("=" * 70)

    # Split data
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE, stratify=True
    )

    # Create model
    model = create_transformer_model()

    # Display model summary
    print("\n" + "=" * 70)
    print("MODEL ARCHITECTURE")
    print("=" * 70)
    model.summary()

    # Setup callbacks
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(LOGS_DIR, f"training_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    callbacks = setup_optimized_callbacks(log_dir)

    # Training
    print("\n" + "=" * 70)
    print("STARTING OPTIMIZED TRAINING")
    print("=" * 70)
    print(f"Epochs: {EPOCHS} (with early stopping)")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_test)}")
    print(f"Target: >90% accuracy")
    print("=" * 70)

    start_time = time.time()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    training_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED! 🎉")
    print("=" * 70)
    print(
        f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)"
    )
    print(f"Actual epochs trained: {len(history.history['loss'])}")
    print("=" * 70)

    # Save training history and plots
    logger = TrainingLogger(log_dir)
    logger.save_training_history(history)
    logger.plot_training_curves()

    # Evaluate model
    evaluator = ModelEvaluator(model, X_test, y_test, ACTIONS)
    accuracy = evaluator.evaluate_all()

    # Save final model
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)
    model.save(MODEL_PATH)
    print(f"✓ Model saved to: {MODEL_PATH}")

    # Save model in Keras format as well
    keras_model_path = MODEL_PATH.replace(".h5", ".keras")
    model.save(keras_model_path)
    print(f"✓ Model also saved in Keras format: {keras_model_path}")
    print("=" * 70)

    # Final summary
    print("\n" + "✅ " * 35)
    print("TRAINING PIPELINE COMPLETED!")
    print("✅ " * 35)
    print(f"\n📊 Final Test Accuracy: {accuracy * 100:.2f}%")

    if accuracy >= 0.90:
        print("🎯 SUCCESS: Achieved >90% accuracy target!")
    else:
        print("⚠️ Consider:")
        print("  - Training for more epochs")
        print("  - Collecting more diverse data")
        print("  - Adjusting hyperparameters")

    print(f"\n📁 Model saved: {MODEL_PATH}")
    print(f"📁 Results saved: {RESULTS_DIR}")
    print(f"📁 Logs saved: {log_dir}")
    print(f"⏱️ Training time: {training_time/60:.1f} minutes")

    return model, accuracy


def main():
    """Main entry point for training script."""
    print("\n" + "=" * 70)
    print("OPTIMIZED SIGN LANGUAGE MODEL TRAINING")
    print("=" * 70)
    print(f"Target: >90% accuracy with faster training")
    print(f"Data location: {PROCESSED_DATA_DIR}")
    print(f"Max epochs: {EPOCHS} (with early stopping)")
    print(f"Batch size: {BATCH_SIZE}")
    print("=" * 70)

    response = input("\nProceed with training? (y/n): ").lower().strip()

    if response != "y":
        print("Training cancelled.")
        return

    # Ask about augmentation
    aug = input("\nUse data augmentation? (recommended) (y/n): ").lower().strip()
    use_augmentation = aug == "y"

    # Start training
    try:
        model, accuracy = train_model(use_augmentation=use_augmentation)

        if model is not None:
            print("\n✅ Training completed successfully!")
            if accuracy >= 0.90:
                print("🎯 TARGET ACHIEVED: >90% accuracy!")
        else:
            print("\n⚠ Training failed or was cancelled.")

    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user!")
        print("Partial results may have been saved.")
    except Exception as e:
        print(f"\n\n✗ Error during training: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
