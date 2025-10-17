"""
Data utilities for loading, saving, and managing sequence data.
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from src.config import (
    PROCESSED_DATA_DIR, 
    ACTIONS, 
    NO_SEQUENCES, 
    SEQUENCE_LENGTH,
    VALIDATION_SPLIT,
    RANDOM_STATE
)


def load_sequences(data_path=None, actions=None, no_sequences=None, sequence_length=None):
    """
    Load processed sequence data from disk.
    
    Args:
        data_path: Path to processed data directory (default: from config)
        actions: List of action labels (default: from config)
        no_sequences: Number of sequences per action (default: from config)
        sequence_length: Length of each sequence (default: from config)
        
    Returns:
        X: numpy array of shape (n_samples, sequence_length, n_features)
        y: numpy array of one-hot encoded labels
        label_map: dictionary mapping action labels to indices
    """
    # Use defaults from config if not provided
    data_path = data_path or PROCESSED_DATA_DIR
    actions = actions or ACTIONS
    no_sequences = no_sequences or NO_SEQUENCES
    sequence_length = sequence_length or SEQUENCE_LENGTH
    
    # Create label mapping
    label_map = {label: num for num, label in enumerate(actions)}
    
    sequences = []
    labels = []
    
    print(f"Loading sequences from: {data_path}")
    print(f"Expected: {len(actions)} actions × {no_sequences} sequences = {len(actions) * no_sequences} total")
    print("-" * 70)
    
    # Load data for each action
    for action in actions:
        action_sequences_found = 0
        
        for sequence_idx in range(no_sequences):
            sequence_path = os.path.join(data_path, action, str(sequence_idx))
            
            # Check if sequence folder exists and has correct number of frames
            if not os.path.exists(sequence_path):
                continue
            
            frame_files = os.listdir(sequence_path)
            
            if len(frame_files) != sequence_length:
                print(f"  ⚠ '{action}' seq {sequence_idx}: Found {len(frame_files)} frames, expected {sequence_length}")
                continue
            
            # Load all frames in the sequence
            window = []
            for frame_num in range(sequence_length):
                frame_path = os.path.join(sequence_path, f"{frame_num}.npy")
                
                if not os.path.exists(frame_path):
                    print(f"  ⚠ '{action}' seq {sequence_idx}: Missing frame {frame_num}")
                    break
                
                frame_data = np.load(frame_path)
                window.append(frame_data)
            
            # Only add complete sequences
            if len(window) == sequence_length:
                sequences.append(window)
                labels.append(label_map[action])
                action_sequences_found += 1
        
        print(f"  ✓ '{action}': Loaded {action_sequences_found}/{no_sequences} sequences")
    
    print("-" * 70)
    print(f"Total sequences loaded: {len(sequences)}")
    
    if len(sequences) == 0:
        raise ValueError("No sequences found! Please run data preprocessing first.")
    
    # Convert to numpy arrays
    X = np.array(sequences)
    y = to_categorical(labels, num_classes=len(actions)).astype(int)
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    return X, y, label_map


def split_data(X, y, test_size=None, random_state=None, stratify=True):
    """
    Split data into training and testing sets.
    
    Args:
        X: Feature array
        y: Label array (one-hot encoded)
        test_size: Fraction of data to use for testing (default: from config)
        random_state: Random seed (default: from config)
        stratify: Whether to stratify split by labels
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    test_size = test_size or VALIDATION_SPLIT
    random_state = random_state or RANDOM_STATE
    
    # Convert one-hot back to labels for stratification
    y_labels = np.argmax(y, axis=1) if stratify else None
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y_labels
        )
        print(f"✓ Data split: {len(X_train)} train, {len(X_test)} test (stratified)")
    except ValueError as e:
        # If stratification fails (too few samples), split without stratification
        print(f"⚠ Stratification failed: {e}")
        print("  Splitting without stratification...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state
        )
        print(f"✓ Data split: {len(X_train)} train, {len(X_test)} test")
    
    return X_train, X_test, y_train, y_test


def verify_data_integrity(data_path=None, actions=None, no_sequences=None, sequence_length=None):
    """
    Verify the integrity of processed data.
    
    Args:
        data_path: Path to processed data directory
        actions: List of action labels
        no_sequences: Expected number of sequences per action
        sequence_length: Expected length of each sequence
        
    Returns:
        Dictionary with verification results
    """
    data_path = data_path or PROCESSED_DATA_DIR
    actions = actions or ACTIONS
    no_sequences = no_sequences or NO_SEQUENCES
    sequence_length = sequence_length or SEQUENCE_LENGTH
    
    print("=" * 70)
    print("VERIFYING DATA INTEGRITY")
    print("=" * 70)
    
    results = {
        'total_expected': len(actions) * no_sequences,
        'total_found': 0,
        'complete_sequences': 0,
        'incomplete_sequences': 0,
        'missing_sequences': 0,
        'action_details': {}
    }
    
    for action in actions:
        action_stats = {
            'complete': 0,
            'incomplete': 0,
            'missing': 0
        }
        
        for sequence_idx in range(no_sequences):
            sequence_path = os.path.join(data_path, action, str(sequence_idx))
            
            if not os.path.exists(sequence_path):
                action_stats['missing'] += 1
                results['missing_sequences'] += 1
                continue
            
            frame_files = os.listdir(sequence_path)
            
            if len(frame_files) == sequence_length:
                action_stats['complete'] += 1
                results['complete_sequences'] += 1
            else:
                action_stats['incomplete'] += 1
                results['incomplete_sequences'] += 1
            
            results['total_found'] += 1
        
        results['action_details'][action] = action_stats
        
        status = "✓" if action_stats['complete'] == no_sequences else "⚠"
        print(f"{status} '{action}': {action_stats['complete']}/{no_sequences} complete "
              f"({action_stats['incomplete']} incomplete, {action_stats['missing']} missing)")
    
    print("=" * 70)
    print(f"Summary:")
    print(f"  Total sequences: {results['total_found']}/{results['total_expected']}")
    print(f"  Complete: {results['complete_sequences']}")
    print(f"  Incomplete: {results['incomplete_sequences']}")
    print(f"  Missing: {results['missing_sequences']}")
    print("=" * 70)
    
    return results


def save_sequence(keypoints, action, sequence_idx, frame_num, data_path=None):
    """
    Save a single frame of keypoints to disk.
    
    Args:
        keypoints: Numpy array of keypoints
        action: Action label
        sequence_idx: Sequence index
        frame_num: Frame number within sequence
        data_path: Path to save data (default: from config)
    """
    data_path = data_path or PROCESSED_DATA_DIR
    
    # Create directory if it doesn't exist
    sequence_dir = os.path.join(data_path, action, str(sequence_idx))
    os.makedirs(sequence_dir, exist_ok=True)
    
    # Save keypoints
    file_path = os.path.join(sequence_dir, f"{frame_num}.npy")
    np.save(file_path, keypoints)


def get_action_statistics(y, actions=None):
    """
    Get statistics about action distribution in dataset.
    
    Args:
        y: One-hot encoded labels
        actions: List of action labels
        
    Returns:
        Dictionary with action counts
    """
    actions = actions or ACTIONS
    y_labels = np.argmax(y, axis=1)
    
    stats = {}
    for idx, action in enumerate(actions):
        count = np.sum(y_labels == idx)
        stats[action] = count
    
    return stats


if __name__ == "__main__":
    print("Data Utils Module")
    print("=" * 70)
    
    # Test data verification
    print("\nTesting data integrity verification...")
    try:
        results = verify_data_integrity()
        
        if results['complete_sequences'] > 0:
            print("\n✓ Data verification successful!")
        else:
            print("\n⚠ No complete sequences found. Run data preprocessing first.")
    except Exception as e:
        print(f"\n⚠ Could not verify data: {e}")
        print("This is normal if you haven't run preprocessing yet.")