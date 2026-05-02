"""
Training pipeline — transformer model for sign language detection.
Extractor contract enforced via runtime shape assertion on load.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import (
    TensorBoard,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from src.config import (
    EXTRACTOR_MODE,
    PROCESSED_DATA_DIR,
    SAVED_MODELS_DIR,
    MODEL_PATH,
    LOGS_DIR,
    RESULTS_DIR,
    ACTIONS,
    NO_SEQUENCES,
    SEQUENCE_LENGTH,
    TOTAL_KEYPOINTS,
    BATCH_SIZE,
    EPOCHS,
    VALIDATION_SPLIT,
    RANDOM_STATE,
)
from src.models.transformer_model import create_transformer_model
from src.utils.data_utils import load_sequences, split_data, get_action_statistics

tf.random.set_seed(RANDOM_STATE)


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------
def _assert_data_contract(X: np.ndarray):
    """Crash early if preprocessed data doesn't match extractor contract."""
    expected = (SEQUENCE_LENGTH, TOTAL_KEYPOINTS)
    if X.shape[1:] != expected:
        raise RuntimeError(
            f"Data shape mismatch: expected sequences of shape {expected}, "
            f"got {X.shape[1:]}.\n"
            f"  extractor_mode={EXTRACTOR_MODE}\n"
            f"  Re-run preprocessing with the correct extractor."
        )
    # For hands_normalized: dims [0:1536] must be zero in every frame
    zero_region = X[:, :, :1536]
    if not np.allclose(zero_region, 0.0):
        raise RuntimeError(
            "Pose/face dims [0:1536] are not zero. "
            "Data was preprocessed with extract_keypoints_holistic(), "
            f"but extractor_mode='{EXTRACTOR_MODE}' requires hands_normalized. "
            "Delete data/processed/ and re-run preprocessing."
        )
    print(
        f"✓ Data contract validated  |  extractor={EXTRACTOR_MODE}  |  shape={X.shape}"
    )


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def _augment_sequence(seq: np.ndarray) -> np.ndarray:
    aug = seq.copy()
    aug += np.random.normal(0, 0.01, aug.shape)
    aug *= 1.0 + np.random.uniform(-0.05, 0.05)
    return aug


def _augment(
    X: np.ndarray, y: np.ndarray, factor: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    copies_X = [X] + [
        np.array([_augment_sequence(s) for s in X]) for _ in range(factor)
    ]
    copies_y = [y] * (factor + 1)
    X_out = np.concatenate(copies_X)
    y_out = np.concatenate(copies_y)
    print(f"✓ Augmented: {len(X)} → {len(X_out)} sequences")
    return X_out, y_out


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def _callbacks(log_dir: str) -> list:
    return [
        TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq="epoch"),
        ModelCheckpoint(
            filepath=os.path.join(SAVED_MODELS_DIR, "best_model.keras"),
            monitor="val_categorical_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_categorical_accuracy",
            patience=15,
            restore_best_weights=True,
            mode="max",
            min_delta=0.001,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=7,
            min_lr=1e-7,
            mode="min",
            verbose=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(use_augmentation: bool = True) -> tuple:
    print(f"\n{'='*60}")
    print(f"  TRAINING  |  extractor={EXTRACTOR_MODE}")
    print(f"{'='*60}\n")

    X, y, _ = load_sequences(
        data_path=PROCESSED_DATA_DIR,
        actions=ACTIONS,
        no_sequences=NO_SEQUENCES,
        sequence_length=SEQUENCE_LENGTH,
    )

    _assert_data_contract(X)

    # Split BEFORE augmentation — prevents data leakage into test set
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE, stratify=True
    )

    if use_augmentation:
        X_train, y_train = _augment(X_train, y_train, factor=2)

    stats = get_action_statistics(y, ACTIONS)
    print("\nClass distribution (original):")
    for action, count in stats.items():
        print(f"  '{action}': {count}")

    model = create_transformer_model()
    model.summary()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(LOGS_DIR, f"run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}")
    print(f"  Train: {len(X_train)}  |  Val: {len(X_test)}")
    print(f"{'='*60}\n")

    t0 = time.time()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=_callbacks(log_dir),
        verbose=1,
    )
    elapsed = time.time() - t0

    np.save(os.path.join(log_dir, "history.npy"), history.history)
    _plot_curves(history, log_dir)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    accuracy = float(np.mean(y_pred == y_true))

    print(classification_report(y_true, y_pred, target_names=ACTIONS, zero_division=0))
    _plot_confusion(y_true, y_pred)

    model.save(MODEL_PATH)
    print(f"\n✓ Model saved → {MODEL_PATH}")
    print(f"✓ Accuracy: {accuracy*100:.2f}%  |  Time: {elapsed/60:.1f} min")

    return model, accuracy


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _plot_curves(history, log_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, val_metric, title in [
        (axes[0], "loss", "val_loss", "Loss"),
        (axes[1], "categorical_accuracy", "val_categorical_accuracy", "Accuracy"),
    ]:
        ax.plot(history.history[metric], label="train")
        ax.plot(history.history[val_metric], label="val")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
    axes[1].axhline(0.9, color="r", linestyle="--", alpha=0.5, label="90% target")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Training curves → {path}")


def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(18, 14))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=ACTIONS, yticklabels=ACTIONS
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Confusion matrix → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print(f"\n{'='*60}")
    print(f"  SIGN LANGUAGE — MODEL TRAINING")
    print(f"  extractor_mode : {EXTRACTOR_MODE}")
    print(f"  data           : {PROCESSED_DATA_DIR}")
    print(f"  max epochs     : {EPOCHS}")
    print(f"{'='*60}\n")

    if input("Proceed? (y/n): ").strip().lower() != "y":
        return

    use_aug = input("Use data augmentation? (y/n): ").strip().lower() == "y"

    try:
        model, acc = train(use_augmentation=use_aug)
        if model is not None:
            print(f"\n✅ Training complete  |  accuracy={acc*100:.2f}%")
    except RuntimeError as e:
        print(f"\n✗ Contract violation: {e}")
    except KeyboardInterrupt:
        print("\n⚠ Interrupted")


if __name__ == "__main__":
    main()
