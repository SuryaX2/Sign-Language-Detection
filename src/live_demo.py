"""
Live demo script for real-time sign language detection using webcam.
Uses MediaPipe Hands for hand tracking and trained transformer model for prediction.
"""

import cv2
import numpy as np
import time
from collections import deque

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
    """Real-time sign language detection from webcam."""

    def __init__(self, model_path, actions, sequence_length=20, threshold=0.6):
        """
        Initialize the detector.

        Args:
            model_path: Path to trained model
            actions: List of action labels
            sequence_length: Number of frames per sequence
            threshold: Confidence threshold for predictions
        """
        self.model_path = model_path
        self.actions = np.array(actions)
        self.sequence_length = sequence_length
        self.threshold = threshold

        # Load model
        print("Loading trained model...")
        self.model = load_trained_model(model_path)
        print("✓ Model loaded successfully!")

        # Initialize tracking variables
        self.sequence = deque(maxlen=sequence_length)
        self.sentence = deque(maxlen=5)  # Store last 5 predictions
        self.predictions = deque(maxlen=10)  # Store last 10 prediction indices

        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

    def reset(self):
        """Reset all tracking variables."""
        self.sequence.clear()
        self.sentence.clear()
        self.predictions.clear()
        self.frame_count = 0
        self.start_time = time.time()

    def update_fps(self):
        """Update FPS calculation."""
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:  # Update every second
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

    def process_frame(self, keypoints):
        """
        Process a single frame and make prediction.

        Args:
            keypoints: Extracted keypoints from current frame

        Returns:
            Tuple of (predicted_action, confidence, is_new_prediction)
        """
        # Add keypoints to sequence
        self.sequence.append(keypoints)

        # Need full sequence to make prediction
        if len(self.sequence) < self.sequence_length:
            return None, 0.0, False

        # Make prediction
        sequence_array = np.array(list(self.sequence))
        res = self.model.predict(np.expand_dims(sequence_array, axis=0), verbose=0)[0]

        prediction_index = np.argmax(res)
        predicted_action = self.actions[prediction_index]
        confidence = res[prediction_index]

        # Track predictions
        self.predictions.append(prediction_index)

        # Only add to sentence if:
        # 1. Confidence is above threshold
        # 2. Last 10 predictions are consistent
        # 3. Different from last prediction in sentence
        is_new_prediction = False

        if len(self.predictions) >= 10:
            # Check if last 10 predictions are mostly the same
            unique, counts = np.unique(list(self.predictions), return_counts=True)
            most_common_idx = unique[np.argmax(counts)]

            if most_common_idx == prediction_index and confidence > self.threshold:
                if len(self.sentence) == 0 or predicted_action != self.sentence[-1]:
                    self.sentence.append(predicted_action)
                    is_new_prediction = True

        return predicted_action, confidence, is_new_prediction

    def draw_ui(self, image, predicted_action, confidence):
        """
        Draw user interface on image.

        Args:
            image: Image to draw on
            predicted_action: Current predicted action
            confidence: Prediction confidence
        """
        h, w, _ = image.shape

        # Draw top bar with sentence
        cv2.rectangle(image, (0, 0), (w, 60), (245, 117, 16), -1)

        # Display sentence
        sentence_text = (
            " ".join(list(self.sentence))
            if len(self.sentence) > 0
            else "Waiting for sign..."
        )
        cv2.putText(
            image,
            sentence_text,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

        # Draw bottom bar with current prediction
        cv2.rectangle(image, (0, h - 80), (w, h), (50, 50, 50), -1)

        if predicted_action is not None:
            # Prediction text
            pred_text = f"Detecting: {predicted_action}"
            cv2.putText(
                image,
                pred_text,
                (10, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Confidence bar
            bar_width = int((w - 20) * confidence)
            bar_color = (0, 255, 0) if confidence > self.threshold else (0, 165, 255)
            cv2.rectangle(image, (10, h - 25), (10 + bar_width, h - 10), bar_color, -1)
            cv2.rectangle(image, (10, h - 25), (w - 10, h - 10), (100, 100, 100), 2)

            # Confidence percentage
            conf_text = f"{confidence * 100:.1f}%"
            cv2.putText(
                image,
                conf_text,
                (w - 100, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Draw FPS counter
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(
            image,
            fps_text,
            (w - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Draw instructions
        instructions = [
            "Press 'q' to quit",
            "Press 'r' to reset",
            "Press 's' to save sentence",
        ]
        y_offset = 100
        for instruction in instructions:
            cv2.putText(
                image,
                instruction,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y_offset += 25

    def save_sentence(self):
        """Save current sentence to file."""
        if len(self.sentence) == 0:
            print("⚠ No sentence to save!")
            return

        sentence_text = " ".join(list(self.sentence))
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"sentence_{timestamp}.txt"

        with open(filename, "w") as f:
            f.write(sentence_text)

        print(f"✓ Sentence saved to: {filename}")
        print(f"  Content: {sentence_text}")

    def run(self):
        """Run the live detection demo."""
        print("\n" + "=" * 70)
        print("STARTING LIVE SIGN LANGUAGE DETECTION")
        print("=" * 70)
        print("Instructions:")
        print("  - Show hand signs to the camera")
        print("  - Press 'q' to quit")
        print("  - Press 'r' to reset detected sentence")
        print("  - Press 's' to save current sentence")
        print("=" * 70 + "\n")

        # Open webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("✗ Error: Could not open webcam!")
            return

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("✓ Webcam opened successfully!")
        print("✓ Starting detection...\n")

        # Initialize MediaPipe Hands
        with HandsDetector(
            max_num_hands=2,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        ) as hands:

            predicted_action = None
            confidence = 0.0

            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    print("✗ Failed to grab frame")
                    break

                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)

                # Make detections
                image, results = mediapipe_detection(frame, hands)

                # Draw hand landmarks
                draw_hand_landmarks(image, results)

                # Extract keypoints
                keypoints = extract_keypoints_hands_normalized(results)

                # Process frame and get prediction
                predicted_action, confidence, is_new = self.process_frame(keypoints)

                if is_new:
                    print(
                        f"✓ New sign detected: '{predicted_action}' (Confidence: {confidence * 100:.1f}%)"
                    )

                # Update FPS
                self.update_fps()

                # Draw UI
                self.draw_ui(image, predicted_action, confidence)

                # Display frame
                cv2.imshow("Sign Language Detection", image)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("\n✓ Quitting...")
                    break
                elif key == ord("r"):
                    print("\n✓ Resetting sentence...")
                    self.reset()
                elif key == ord("s"):
                    print("\n✓ Saving sentence...")
                    self.save_sentence()

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        print("\n" + "=" * 70)
        print("DEMO ENDED")
        print("=" * 70)

        if len(self.sentence) > 0:
            final_sentence = " ".join(list(self.sentence))
            print(f"Final sentence: {final_sentence}")


def main():
    """Main entry point for live demo."""
    print("\n" + "🎥 " * 35)
    print("SIGN LANGUAGE LIVE DETECTION DEMO")
    print("🎥 " * 35 + "\n")

    # Check if model exists
    import os

    if not os.path.exists(MODEL_PATH):
        print(f"✗ Trained model not found at: {MODEL_PATH}")
        print("  Please train a model first: python -m src.train_model")
        return

    print(f"Using model: {MODEL_PATH}")
    print(f"Detection threshold: {PREDICTION_THRESHOLD * 100:.0f}%")
    print(f"Sequence length: {SEQUENCE_LENGTH} frames")

    response = input("\nStart live detection? (y/n): ").lower().strip()

    if response != "y":
        print("Demo cancelled.")
        return

    # Run demo
    try:
        detector = SignLanguageDetector(
            model_path=MODEL_PATH,
            actions=ACTIONS,
            sequence_length=SEQUENCE_LENGTH,
            threshold=PREDICTION_THRESHOLD,
        )

        detector.run()

        print("\n✅ Demo completed successfully!")

    except KeyboardInterrupt:
        print("\n\n⚠ Demo interrupted by user!")
    except Exception as e:
        print(f"\n\n✗ Error during demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
