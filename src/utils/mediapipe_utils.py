"""
MediaPipe utilities for pose, face, and hand landmark detection.
Handles keypoint extraction with normalization for sign language recognition.
"""

import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe solutions
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def mediapipe_detection(image, model):
    """
    Perform MediaPipe detection on an image.
    
    Args:
        image: BGR image from OpenCV
        model: MediaPipe model (Holistic or Hands)
        
    Returns:
        image: Processed BGR image
        results: MediaPipe detection results
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Image is no longer writeable to improve performance
    image_rgb.flags.writeable = False
    
    # Make detection
    results = model.process(image_rgb)
    
    # Image is now writeable again
    image_rgb.flags.writeable = True
    
    # Convert back to BGR
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    return image_bgr, results


def draw_styled_landmarks(image, results):
    """
    Draw styled landmarks on the image for holistic detection.
    
    Args:
        image: Image to draw on
        results: MediaPipe Holistic results
    """
    # Draw face connections
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            results.face_landmarks, 
            mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        )
    
    # Draw pose connections
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )
    
    # Draw left hand connections
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            results.left_hand_landmarks, 
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )
    
    # Draw right hand connections
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            results.right_hand_landmarks, 
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )


def draw_hand_landmarks(image, results):
    """
    Draw hand landmarks for Hands model (used in live demo).
    
    Args:
        image: Image to draw on
        results: MediaPipe Hands results
    """
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )


def extract_keypoints_holistic(results):
    """
    Extract keypoints from MediaPipe Holistic results.
    Used for data preprocessing from images.
    
    Args:
        results: MediaPipe Holistic results
        
    Returns:
        numpy array of shape (1662,) containing all keypoints
    """
    # Extract pose landmarks (33 landmarks * 4 values = 132)
    pose = np.array([[res.x, res.y, res.z, res.visibility] 
                     for res in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33 * 4)
    
    # Extract face landmarks (468 landmarks * 3 values = 1404)
    face = np.array([[res.x, res.y, res.z] 
                     for res in results.face_landmarks.landmark]).flatten() \
           if results.face_landmarks else np.zeros(468 * 3)
    
    # Extract left hand landmarks (21 landmarks * 3 values = 63)
    lh = np.array([[res.x, res.y, res.z] 
                   for res in results.left_hand_landmarks.landmark]).flatten() \
         if results.left_hand_landmarks else np.zeros(21 * 3)
    
    # Extract right hand landmarks (21 landmarks * 3 values = 63)
    rh = np.array([[res.x, res.y, res.z] 
                   for res in results.right_hand_landmarks.landmark]).flatten() \
         if results.right_hand_landmarks else np.zeros(21 * 3)
    
    # Concatenate all keypoints: 132 + 1404 + 63 + 63 = 1662
    return np.concatenate([pose, face, lh, rh])


def extract_keypoints_hands_normalized(results):
    """
    Extract and normalize hand keypoints from MediaPipe Hands results.
    This version includes wrist normalization for position invariance.
    Used for live demo/inference.
    
    Args:
        results: MediaPipe Hands results
        
    Returns:
        numpy array of shape (1662,) with normalized hand keypoints
    """
    # Initialize all keypoint arrays with zeros
    pose = np.zeros(33 * 4)
    face = np.zeros(468 * 3)
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)
    
    # Process detected hands
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                              results.multi_handedness):
            # Determine which hand (Left or Right)
            hand_label = handedness.classification[0].label
            
            # Extract raw hand coordinates
            hand_points = np.array([[landmark.x, landmark.y, landmark.z] 
                                   for landmark in hand_landmarks.landmark])
            
            # --- NORMALIZATION LOGIC ---
            # 1. Get wrist coordinates (landmark 0)
            wrist_coords = hand_points[0]
            
            # 2. Subtract wrist coordinates to make position-invariant
            relative_coords = hand_points - wrist_coords
            
            # 3. Flatten to 1D array
            normalized_hand = relative_coords.flatten()
            
            # 4. Assign to correct hand based on label
            if hand_label == 'Left':
                lh = normalized_hand
            elif hand_label == 'Right':
                rh = normalized_hand
    
    # Concatenate all keypoints: 132 + 1404 + 63 + 63 = 1662
    return np.concatenate([pose, face, lh, rh])


def extract_keypoints_hands(results):
    """
    Extract hand keypoints WITHOUT normalization.
    Alternative version if normalization causes issues.
    
    Args:
        results: MediaPipe Hands results
        
    Returns:
        numpy array of shape (1662,)
    """
    # Initialize all keypoint arrays with zeros
    pose = np.zeros(33 * 4)
    face = np.zeros(468 * 3)
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)
    
    # Process detected hands
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                              results.multi_handedness):
            hand_label = handedness.classification[0].label
            
            # Extract hand coordinates (no normalization)
            hand_points = np.array([[landmark.x, landmark.y, landmark.z] 
                                   for landmark in hand_landmarks.landmark]).flatten()
            
            if hand_label == 'Left':
                lh = hand_points
            elif hand_label == 'Right':
                rh = hand_points
    
    return np.concatenate([pose, face, lh, rh])


class HolisticDetector:
    """Context manager for MediaPipe Holistic detection."""
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.holistic = None
    
    def __enter__(self):
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        return self.holistic
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.holistic:
            self.holistic.close()


class HandsDetector:
    """Context manager for MediaPipe Hands detection."""
    
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, 
                 min_tracking_confidence=0.5):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.hands = None
    
    def __enter__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        return self.hands
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hands:
            self.hands.close()


if __name__ == "__main__":
    print("MediaPipe Utils Module")
    print("=" * 50)
    print("Available functions:")
    print("  - mediapipe_detection()")
    print("  - draw_styled_landmarks()")
    print("  - draw_hand_landmarks()")
    print("  - extract_keypoints_holistic()")
    print("  - extract_keypoints_hands_normalized()")
    print("  - extract_keypoints_hands()")
    print("\nAvailable classes:")
    print("  - HolisticDetector (context manager)")
    print("  - HandsDetector (context manager)")