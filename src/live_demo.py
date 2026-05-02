"""
Live sign language detection — webcam inference.
Extractor: hands_normalized (must match preprocessing extractor).
"""

import os
import cv2
import time
import numpy as np
from collections import deque, Counter

from src.config import (
    EXTRACTOR_MODE,
    MODEL_PATH,
    ACTIONS,
    SEQUENCE_LENGTH,
    TOTAL_KEYPOINTS,
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

assert (
    EXTRACTOR_MODE == "hands_normalized"
), f"live_demo only supports extractor='hands_normalized', got '{EXTRACTOR_MODE}'"

_STABILITY_WINDOW = 15
_MIN_STABLE_FRAMES = 14
_COOLDOWN_FRAMES = 15
_ENTROPY_THRESHOLD = 0.7


def _assert_model_contract(model):
    """Crash before the first frame if the model was trained on a different extractor."""
    expected = (SEQUENCE_LENGTH, TOTAL_KEYPOINTS)
    actual = tuple(model.input_shape[1:])
    if actual != expected:
        raise RuntimeError(
            f"Model input shape {actual} != expected {expected}.\n"
            f"  extractor_mode={EXTRACTOR_MODE}\n"
            f"  The saved model was likely trained with a different extractor.\n"
            f"  Re-run preprocessing + training to fix."
        )
    print(
        f"✓ Model contract validated  |  input={actual}  |  extractor={EXTRACTOR_MODE}"
    )


class SignLanguageDetector:
    def __init__(
        self, model_path: str, confidence_threshold: float = PREDICTION_THRESHOLD
    ):
        self.actions = np.array(ACTIONS)
        self.conf_thr = confidence_threshold
        self.model = load_trained_model(model_path)
        _assert_model_contract(self.model)

        self._seq = deque(maxlen=SEQUENCE_LENGTH)
        self.sentence = deque(maxlen=5)
        self._pred_buf = deque(maxlen=_STABILITY_WINDOW)
        self._conf_buf = deque(maxlen=_STABILITY_WINDOW)
        self._since_last = 0
        self._fps = 0
        self._frame_count = 0
        self._t0 = time.time()

    # ------------------------------------------------------------------
    def reset(self):
        self._seq.clear()
        self.sentence.clear()
        self._pred_buf.clear()
        self._conf_buf.clear()
        self._since_last = 0
        self._frame_count = 0
        self._t0 = time.time()

    # ------------------------------------------------------------------
    def _update_fps(self):
        self._frame_count += 1
        elapsed = time.time() - self._t0
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._t0 = time.time()

    # ------------------------------------------------------------------
    @staticmethod
    def _no_gesture(kp: np.ndarray) -> bool:
        return np.count_nonzero(kp[-126:]) / 126 < 0.1

    # ------------------------------------------------------------------
    def process(self, kp: np.ndarray) -> tuple:
        """Returns (action|None, confidence, stability, is_new)."""
        self._since_last += 1

        if self._no_gesture(kp):
            self._seq.clear()
            self._pred_buf.clear()
            self._conf_buf.clear()
            return None, 0.0, 0.0, False

        self._seq.append(kp)
        if len(self._seq) < SEQUENCE_LENGTH:
            return None, 0.0, 0.0, False

        probs = self.model.predict(np.expand_dims(list(self._seq), 0), verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        action = self.actions[idx]

        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
        if entropy / (-np.log(1.0 / len(self.actions))) > _ENTROPY_THRESHOLD:
            return action, conf, 0.0, False

        self._pred_buf.append(idx)
        self._conf_buf.append(conf)

        if len(self._pred_buf) < _STABILITY_WINDOW:
            return action, conf, 0.0, False

        top_idx, top_count = Counter(self._pred_buf).most_common(1)[0]
        stability = top_count / _STABILITY_WINDOW
        avg_conf = float(np.mean(self._conf_buf))

        if (
            top_idx == idx
            and stability >= _MIN_STABLE_FRAMES / _STABILITY_WINDOW
            and avg_conf >= self.conf_thr
            and self._since_last >= _COOLDOWN_FRAMES
        ):
            stable_action = self.actions[top_idx]
            if not self.sentence or stable_action != self.sentence[-1]:
                self.sentence.append(stable_action)
                self._since_last = 0
                self._pred_buf.clear()
                self._conf_buf.clear()
                print(
                    f"✓ '{stable_action}'  conf={avg_conf:.2f}  stability={stability:.2f}"
                )
                return stable_action, avg_conf, stability, True

        return action, conf, stability, False

    # ------------------------------------------------------------------
    def _draw(self, img: np.ndarray, action, conf: float, stability: float):
        h, w = img.shape[:2]

        cv2.rectangle(img, (0, 0), (w, 70), (40, 40, 40), -1)
        text = " ".join(self.sentence) if self.sentence else "Waiting..."
        cv2.putText(
            img, text, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3
        )

        if action:
            dot_color = (0, 255, 0) if stability > 0.7 else (0, 165, 255)
            cv2.circle(img, (w - 100, 35), 15, dot_color, -1)

        ph = 140
        cv2.rectangle(img, (0, h - ph), (w, h), (30, 30, 30), -1)
        cv2.putText(
            img,
            f"Q:quit R:reset S:save SPACE:space",
            (15, h - ph - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
        )

        if action:
            lc = (0, 255, 0) if conf >= self.conf_thr else (100, 200, 255)
            cv2.putText(
                img,
                "Detecting:",
                (15, h - 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 180, 180),
                2,
            )
            cv2.putText(
                img, str(action), (15, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.8, lc, 4
            )

            bx, bw, by = 160, 280, h - 85
            cv2.rectangle(img, (bx, by), (bx + bw, by + 14), (60, 60, 60), -1)
            cv2.rectangle(
                img,
                (bx, by),
                (bx + int(bw * conf), by + 14),
                (0, 255, 0) if conf >= self.conf_thr else (0, 200, 255),
                -1,
            )
            cv2.putText(
                img,
                f"{conf*100:.0f}%",
                (bx + bw + 8, by + 11),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
            )

            by2 = h - 55
            cv2.rectangle(img, (bx, by2), (bx + bw, by2 + 14), (60, 60, 60), -1)
            cv2.rectangle(
                img,
                (bx, by2),
                (bx + int(bw * stability), by2 + 14),
                (0, 255, 0) if stability > 0.7 else (140, 140, 0),
                -1,
            )
            cv2.putText(
                img,
                f"Stable {stability*100:.0f}%",
                (bx + bw + 8, by2 + 11),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
            )

            if self._since_last < _COOLDOWN_FRAMES:
                ratio = self._since_last / _COOLDOWN_FRAMES
                cv2.putText(
                    img,
                    f"Cooldown {int((1-ratio)*100)}%",
                    (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (100, 100, 255),
                    1,
                )
        else:
            cv2.putText(
                img,
                "No sign detected",
                (15, h - 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (150, 150, 150),
                2,
            )

        cv2.putText(
            img,
            f"FPS {self._fps:.0f}",
            (w - 90, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

    # ------------------------------------------------------------------
    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        print("✓ Camera ready\n")

        with HandsDetector(
            max_num_hands=2,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        ) as hands:
            action, conf, stability = None, 0.0, 0.0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                image, results = mediapipe_detection(frame, hands)
                draw_hand_landmarks(image, results)

                kp = extract_keypoints_hands_normalized(results)
                action, conf, stability, _ = self.process(kp)
                self._update_fps()
                self._draw(image, action, conf, stability)
                cv2.imshow("Sign Language Detection", image)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    self.reset()
                    print("✓ Reset")
                elif key == ord("s"):
                    self._save()
                elif key == ord(" "):
                    self.sentence.append(" ")
                    print("✓ Space")

        cap.release()
        cv2.destroyAllWindows()
        if self.sentence:
            print(f"\nFinal: {''.join(self.sentence)}")

    # ------------------------------------------------------------------
    def _save(self):
        if not self.sentence:
            return
        text = "".join(self.sentence)
        fname = f"sentence_{time.strftime('%Y%m%d-%H%M%S')}.txt"
        with open(fname, "w") as f:
            f.write(text)
        print(f"✓ Saved '{text}' → {fname}")


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"✗ Model not found: {MODEL_PATH}")
        print("  Run: python -m src.train_model")
        return

    print(f"\n{'='*60}")
    print(f"  LIVE DETECTION  |  extractor={EXTRACTOR_MODE}")
    print(f"  model           : {MODEL_PATH}")
    print(f"  conf_threshold  : {PREDICTION_THRESHOLD}")
    print(f"{'='*60}\n")

    if input("Start? (y/n): ").strip().lower() != "y":
        return

    try:
        detector = SignLanguageDetector(
            MODEL_PATH, confidence_threshold=PREDICTION_THRESHOLD
        )
        detector.run()
    except RuntimeError as e:
        print(f"\n✗ {e}")
    except KeyboardInterrupt:
        print("\n⚠ Interrupted")


if __name__ == "__main__":
    main()
