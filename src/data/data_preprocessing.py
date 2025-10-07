"""
Data Preprocessing Utilities for SEED-VII Dataset
Implements feature normalization, augmentation, and preparation functions
"""

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import scipy.signal as signal


class EEGPreprocessor:
    """
    EEG signal preprocessing utilities for SEED-VII dataset
    
    Handles normalization, filtering, and feature engineering for EEG data
    """
    
    def __init__(self, method='standardize'):
        """
        Initialize EEG preprocessor
        
        Args:
            method (str): Normalization method - 'standardize', 'normalize', or 'robust'
        """
        self.method = method
        self.scaler = None
        self.fitted = False
    
    def fit(self, eeg_data):
        """
        Fit preprocessing parameters on training data
        
        Args:
            eeg_data (np.ndarray): EEG features [n_samples, n_features]
        """
        if self.method == 'standardize':
            self.scaler = StandardScaler()
        elif self.method == 'normalize':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        self.scaler.fit(eeg_data)
        self.fitted = True
    
    def transform(self, eeg_data):
        """
        Transform EEG data using fitted parameters
        
        Args:
            eeg_data (np.ndarray): EEG features to transform
            
        Returns:
            np.ndarray: Preprocessed EEG features
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        return self.scaler.transform(eeg_data)
    
    def fit_transform(self, eeg_data):
        """Fit and transform in one step"""
        self.fit(eeg_data)
        return self.transform(eeg_data)
    
    def apply_temporal_filtering(self, eeg_signal, fs=250, lowcut=0.5, highcut=50):
        """
        Apply bandpass filter to raw EEG signals
        
        Args:
            eeg_signal (np.ndarray): Raw EEG signal [n_channels, n_timepoints]
            fs (float): Sampling frequency
            lowcut (float): Low cutoff frequency
            highcut (float): High cutoff frequency
            
        Returns:
            np.ndarray: Filtered EEG signal
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_signal = signal.filtfilt(b, a, eeg_signal, axis=1)
        
        return filtered_signal


class EyeTrackingPreprocessor:
    """
    Eye tracking data preprocessing utilities
    
    Handles normalization and feature engineering for eye movement data
    """
    
    def __init__(self, method='standardize'):
        """
        Initialize eye tracking preprocessor
        
        Args:
            method (str): Normalization method
        """
        self.method = method
        self.scaler = None
        self.fitted = False
    
    def fit(self, eye_data):
        """Fit preprocessing parameters"""
        if self.method == 'standardize':
            self.scaler = StandardScaler()
        elif self.method == 'normalize':
            self.scaler = MinMaxScaler()
        
        self.scaler.fit(eye_data)
        self.fitted = True
    
    def transform(self, eye_data):
        """Transform eye tracking data"""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        return self.scaler.transform(eye_data)
    
    def fit_transform(self, eye_data):
        """Fit and transform in one step"""
        self.fit(eye_data)
        return self.transform(eye_data)
    
    def compute_fixation_features(self, gaze_x, gaze_y, timestamps):
        """
        Extract fixation-related features from gaze coordinates
        
        Args:
            gaze_x (np.ndarray): X-coordinate of gaze
            gaze_y (np.ndarray): Y-coordinate of gaze  
            timestamps (np.ndarray): Timestamp information
            
        Returns:
            dict: Dictionary of computed fixation features
        """
        features = {}
        
        # Compute gaze velocity
        dx = np.diff(gaze_x)
        dy = np.diff(gaze_y)
        dt = np.diff(timestamps)
        velocity = np.sqrt(dx**2 + dy**2) / dt
        
        features['mean_velocity'] = np.mean(velocity)
        features['std_velocity'] = np.std(velocity)
        features['max_velocity'] = np.max(velocity)
        
        # Compute gaze dispersion
        features['gaze_dispersion_x'] = np.std(gaze_x)
        features['gaze_dispersion_y'] = np.std(gaze_y)
        
        # Compute saccade count (velocity threshold-based)
        saccade_threshold = np.percentile(velocity, 90)
        features['saccade_count'] = np.sum(velocity > saccade_threshold)
        
        return features


class MultimodalPreprocessor:
    """
    Combined preprocessing for multimodal EEG-Eye data
    
    Coordinates preprocessing of both modalities and handles modality-specific operations
    """
    
    def __init__(self, eeg_method='standardize', eye_method='standardize'):
        """
        Initialize multimodal preprocessor
        
        Args:
            eeg_method (str): EEG normalization method
            eye_method (str): Eye tracking normalization method
        """
        self.eeg_preprocessor = EEGPreprocessor(eeg_method)
        self.eye_preprocessor = EyeTrackingPreprocessor(eye_method)
        self.fitted = False
    
    def fit(self, eeg_data, eye_data):
        """
        Fit preprocessors on both modalities
        
        Args:
            eeg_data (np.ndarray): EEG features
            eye_data (np.ndarray): Eye tracking features
        """
        self.eeg_preprocessor.fit(eeg_data)
        self.eye_preprocessor.fit(eye_data)
        self.fitted = True
    
    def transform(self, eeg_data, eye_data):
        """
        Transform both modalities
        
        Args:
            eeg_data (np.ndarray): EEG features to transform
            eye_data (np.ndarray): Eye features to transform
            
        Returns:
            tuple: (transformed_eeg, transformed_eye)
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        eeg_transformed = self.eeg_preprocessor.transform(eeg_data)
        eye_transformed = self.eye_preprocessor.transform(eye_data)
        
        return eeg_transformed, eye_transformed
    
    def fit_transform(self, eeg_data, eye_data):
        """Fit and transform both modalities"""
        self.fit(eeg_data, eye_data)
        return self.transform(eeg_data, eye_data)


class DataAugmentation:
    """
    Data augmentation techniques for EEG and eye tracking data
    """
    
    @staticmethod
    def add_gaussian_noise(data, noise_level=0.01):
        """
        Add Gaussian noise to features for data augmentation
        
        Args:
            data (np.ndarray): Input features
            noise_level (float): Noise standard deviation relative to data std
            
        Returns:
            np.ndarray: Augmented data
        """
        noise = np.random.normal(0, noise_level * np.std(data), data.shape)
        return data + noise
    
    @staticmethod
    def temporal_jittering(features, jitter_ratio=0.1):
        """
        Apply temporal jittering by randomly shifting time windows
        
        Args:
            features (np.ndarray): Temporal features
            jitter_ratio (float): Maximum jitter as fraction of sequence length
            
        Returns:
            np.ndarray: Jittered features
        """
        seq_len = features.shape[0]
        max_jitter = int(seq_len * jitter_ratio)
        jitter = np.random.randint(-max_jitter, max_jitter + 1)
        
        if jitter > 0:
            return np.concatenate([features[jitter:], features[:jitter]], axis=0)
        elif jitter < 0:
            return np.concatenate([features[jitter:], features[:jitter]], axis=0)
        else:
            return features
    
    @staticmethod
    def feature_dropout(features, dropout_ratio=0.1):
        """
        Randomly set some features to zero (feature dropout)
        
        Args:
            features (np.ndarray): Input features
            dropout_ratio (float): Fraction of features to drop
            
        Returns:
            np.ndarray: Features with dropout applied
        """
        dropout_mask = np.random.random(features.shape) > dropout_ratio
        return features * dropout_mask


def create_cross_subject_splits(subject_labels, test_subject):
    """
    Create train/test splits for cross-subject validation
    
    Args:
        subject_labels (np.ndarray): Subject IDs for each sample
        test_subject (int): Subject ID to use as test set
        
    Returns:
        tuple: (train_indices, test_indices)
    """
    train_mask = subject_labels != test_subject
    test_mask = subject_labels == test_subject
    
    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]
    
    return train_indices, test_indices


def balance_dataset(features, labels, method='oversample'):
    """
    Balance dataset to handle class imbalance
    
    Args:
        features (np.ndarray): Feature matrix
        labels (np.ndarray): Class labels
        method (str): Balancing method - 'oversample' or 'undersample'
        
    Returns:
        tuple: (balanced_features, balanced_labels)
    """
    from collections import Counter
    
    class_counts = Counter(labels)
    
    if method == 'oversample':
        max_count = max(class_counts.values())
        target_count = max_count
    else:  # undersample
        min_count = min(class_counts.values())
        target_count = min_count
    
    balanced_features = []
    balanced_labels = []
    
    for class_label in class_counts.keys():
        class_mask = labels == class_label
        class_features = features[class_mask]
        class_size = len(class_features)
        
        if class_size < target_count:
            # Oversample
            indices = np.random.choice(class_size, target_count, replace=True)
        else:
            # Undersample
            indices = np.random.choice(class_size, target_count, replace=False)
        
        balanced_features.append(class_features[indices])
        balanced_labels.extend([class_label] * target_count)
    
    return np.vstack(balanced_features), np.array(balanced_labels)
```
