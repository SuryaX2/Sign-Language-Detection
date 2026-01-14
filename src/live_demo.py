"""
Production-grade live sign language detection with robust prediction logic.
"""

import cv2
import numpy as np
import time
from collections import deque, Counter

from src.config import (
    MODEL_PATH,
    ACTIONS,
    SEQUENCE_LENGTH,
    PREDICTION_THRESHOLD,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)

from src.models.transformer_model import load_trained_model
from src.utils.mediapipe_utils import (
    HandsDetector,
    mediapipe_detection,
    draw_hand_landmarks,
    extract_keypoints_hands_normalized,
)


class SignLanguageDetector:
    """Production-grade real-time sign language detector with stability logic."""

    def __init__(self, model_path, actions, sequence_length=20, threshold=0.7):
        self.model_path = model_path
        self.actions = np.array(actions)
        self.sequence_length = sequence_length
        self.threshold = threshold

        # Enhanced thresholds
        self.confidence_threshold = 0.90
        self.stability_window = 15
        self.min_stable_frames = 14
        self.no_gesture_threshold = 0.60
        self.cooldown_frames = 15

        print("Loading model...")
        self.model = load_trained_model(model_path)
        print("✓ Model loaded")

        # Tracking
        self.sequence = deque(maxlen=sequence_length)
        self.sentence = deque(maxlen=5)
        self.predictions_buffer = deque(maxlen=self.stability_window)
        self.confidence_buffer = deque(maxlen=self.stability_window)

        # State management
        self.current_prediction = None
        self.current_confidence = 0.0
        self.frames_since_last_append = 0
        self.stable_prediction = None
        self.stable_count = 0

        # Performance
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

    def reset(self):
        self.sequence.clear()
        self.sentence.clear()
        self.predictions_buffer.clear()
        self.confidence_buffer.clear()
        self.current_prediction = None
        self.current_confidence = 0.0
        self.frames_since_last_append = 0
        self.stable_prediction = None
        self.stable_count = 0
        self.frame_count = 0
        self.start_time = time.time()

    def update_fps(self):
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

    def is_no_gesture(self, keypoints):
        """Detect if no meaningful gesture is present."""
        hand_keypoints = keypoints[-126:]
        non_zero = np.count_nonzero(hand_keypoints)
        total = len(hand_keypoints)
        ratio = non_zero / total
        return ratio < 0.1

    def process_frame(self, keypoints):
        """
        Process frame with robust prediction logic.

        Returns:
            (predicted_action, confidence, stability, is_new_letter)
        """
        self.frames_since_last_append += 1

        # Check for no gesture
        if self.is_no_gesture(keypoints):
            self.sequence.clear()
            self.predictions_buffer.clear()
            self.confidence_buffer.clear()
            self.stable_prediction = None
            self.stable_count = 0
            return None, 0.0, 0.0, False

        # Add to sequence
        self.sequence.append(keypoints)

        # Need full sequence
        if len(self.sequence) < self.sequence_length:
            return None, 0.0, 0.0, False

        # Make prediction
        seq_array = np.array(list(self.sequence))
        probs = self.model.predict(np.expand_dims(seq_array, axis=0), verbose=0)[0]

        pred_idx = np.argmax(probs)
        pred_action = self.actions[pred_idx]
        pred_conf = probs[pred_idx]

        # Calculate entropy (uncertainty measure)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = -np.log(1.0 / len(self.actions))
        normalized_entropy = entropy / max_entropy

        # Reject if too uncertain (high entropy)
        if normalized_entropy > 0.7:
            return pred_action, pred_conf, 0.0, False

        # Update buffers
        self.predictions_buffer.append(pred_idx)
        self.confidence_buffer.append(pred_conf)

        self.current_prediction = pred_action
        self.current_confidence = pred_conf

        # Calculate stability
        if len(self.predictions_buffer) >= self.stability_window:
            counter = Counter(self.predictions_buffer)
            most_common_idx, most_common_count = counter.most_common(1)[0]
            stability = most_common_count / self.stability_window

            avg_confidence = np.mean(list(self.confidence_buffer))

            # Check if prediction is stable
            if (
                most_common_idx == pred_idx
                and stability >= (self.min_stable_frames / self.stability_window)
                and avg_confidence >= self.confidence_threshold
                and self.frames_since_last_append >= self.cooldown_frames
            ):

                stable_action = self.actions[most_common_idx]

                # Append if different from last
                if len(self.sentence) == 0 or stable_action != self.sentence[-1]:
                    self.sentence.append(stable_action)
                    self.frames_since_last_append = 0
                    self.predictions_buffer.clear()
                    self.confidence_buffer.clear()
                    print(
                        f"✓ Added: '{stable_action}' (conf: {avg_confidence:.2f}, stability: {stability:.2f})"
                    )
                    return stable_action, avg_confidence, stability, True

            return pred_action, pred_conf, stability, False

        return pred_action, pred_conf, 0.0, False

    def draw_ui(self, image, pred_action, confidence, stability):
        """Draw modern UI with prediction stages."""
        h, w, _ = image.shape

        # Top bar - sentence
        cv2.rectangle(image, (0, 0), (w, 70), (40, 40, 40), -1)
        sentence_text = " ".join(list(self.sentence)) if self.sentence else "Waiting..."
        cv2.putText(
            image,
            sentence_text,
            (15, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (255, 255, 255),
            3,
        )

        # Status indicator
        status_x = w - 100
        if pred_action:
            color = (0, 255, 0) if stability > 0.7 else (0, 165, 255)
            cv2.circle(image, (status_x, 35), 15, color, -1)

        # Bottom panel - current prediction
        panel_height = 140
        cv2.rectangle(image, (0, h - panel_height), (w, h), (30, 30, 30), -1)

        if pred_action:
            # Current letter being detected
            cv2.putText(
                image,
                "Detecting:",
                (15, h - 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 180, 180),
                2,
            )

            letter_color = (
                (0, 255, 0)
                if confidence >= self.confidence_threshold
                else (100, 200, 255)
            )
            cv2.putText(
                image,
                pred_action,
                (15, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.8,
                letter_color,
                4,
            )

            # Confidence bar
            bar_x = 150
            bar_width = 300
            bar_y = h - 85

            cv2.rectangle(
                image, (bar_x, bar_y), (bar_x + bar_width, bar_y + 15), (60, 60, 60), -1
            )

            filled_width = int(bar_width * confidence)
            if confidence >= self.confidence_threshold:
                bar_color = (0, 255, 0)
            elif confidence >= 0.7:
                bar_color = (0, 200, 255)
            else:
                bar_color = (100, 100, 100)

            cv2.rectangle(
                image, (bar_x, bar_y), (bar_x + filled_width, bar_y + 15), bar_color, -1
            )

            cv2.putText(
                image,
                f"{confidence*100:.0f}%",
                (bar_x + bar_width + 10, bar_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # Stability bar
            if len(self.predictions_buffer) >= 5:
                bar_y2 = h - 55
                cv2.rectangle(
                    image,
                    (bar_x, bar_y2),
                    (bar_x + bar_width, bar_y2 + 15),
                    (60, 60, 60),
                    -1,
                )

                stability_filled = int(bar_width * stability)
                stability_color = (0, 255, 0) if stability > 0.7 else (150, 150, 0)
                cv2.rectangle(
                    image,
                    (bar_x, bar_y2),
                    (bar_x + stability_filled, bar_y2 + 15),
                    stability_color,
                    -1,
                )

                cv2.putText(
                    image,
                    f"Stable: {stability*100:.0f}%",
                    (bar_x + bar_width + 10, bar_y2 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            # Cooldown indicator
            if self.frames_since_last_append < self.cooldown_frames:
                cooldown_ratio = self.frames_since_last_append / self.cooldown_frames
                cooldown_text = f"Cooldown: {int((1-cooldown_ratio)*100)}%"
                cv2.putText(
                    image,
                    cooldown_text,
                    (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (100, 100, 255),
                    1,
                )
        else:
            cv2.putText(
                image,
                "No sign detected",
                (15, h - 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (150, 150, 150),
                2,
            )

        # FPS
        cv2.putText(
            image,
            f"FPS: {self.fps:.0f}",
            (w - 100, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        # Instructions
        instructions = ["Q: Quit | R: Reset | S: Save | SPACE: Add Space"]
        cv2.putText(
            image,
            instructions[0],
            (15, h - panel_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

    def add_space(self):
        """Add space to sentence."""
        if len(self.sentence) > 0:
            self.sentence.append(" ")
            print("✓ Space added")

    def save_sentence(self):
        """Save sentence to file."""
        if len(self.sentence) == 0:
            print("⚠ No sentence to save")
            return

        sentence_text = "".join(list(self.sentence))
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"sentence_{timestamp}.txt"

        with open(filename, "w") as f:
            f.write(sentence_text)

        print(f"✓ Saved: {filename}")
        print(f"  '{sentence_text}'")

    def run(self):
        """Run live detection."""
        print("\n" + "=" * 70)
        print("LIVE SIGN LANGUAGE DETECTION")
        print("=" * 70)
        print("Controls:")
        print("  Q - Quit")
        print("  R - Reset sentence")
        print("  S - Save sentence")
        print("  SPACE - Add space")
        print("=" * 70 + "\n")

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("✗ Cannot open webcam")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("✓ Camera ready\n")

        with HandsDetector(
            max_num_hands=2,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        ) as hands:

            pred_action = None
            confidence = 0.0
            stability = 0.0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                image, results = mediapipe_detection(frame, hands)
                draw_hand_landmarks(image, results)

                keypoints = extract_keypoints_hands_normalized(results)
                pred_action, confidence, stability, is_new = self.process_frame(
                    keypoints
                )

                self.update_fps()
                self.draw_ui(image, pred_action, confidence, stability)

                cv2.imshow("Sign Language Detection", image)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("\n✓ Exiting...")
                    break
                elif key == ord("r"):
                    print("\n✓ Reset")
                    self.reset()
                elif key == ord("s"):
                    self.save_sentence()
                elif key == ord(" "):
                    self.add_space()

        cap.release()
        cv2.destroyAllWindows()

        print("\n" + "=" * 70)
        print("SESSION ENDED")
        if len(self.sentence) > 0:
            print(f"Final: {''.join(list(self.sentence))}")
        print("=" * 70)


def main():
    import os

    print("\n" + "🎥 " * 35)
    print("SIGN LANGUAGE LIVE DETECTION")
    print("🎥 " * 35 + "\n")

    if not os.path.exists(MODEL_PATH):
        print(f"✗ Model not found: {MODEL_PATH}")
        print("  Train model first: python -m src.train_model")
        return

    print(f"Model: {MODEL_PATH}")
    print(f"Confidence threshold: 85%")
    print(f"Stability window: 15 frames")
    print(f"Min stable frames: 12")

    response = input("\nStart? (y/n): ").strip().lower()
    if response != "y":
        print("Cancelled")
        return

    try:
        detector = SignLanguageDetector(
            model_path=MODEL_PATH,
            actions=ACTIONS,
            sequence_length=SEQUENCE_LENGTH,
            threshold=PREDICTION_THRESHOLD,
        )
        detector.run()
        print("\n✅ Completed")
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted")
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
