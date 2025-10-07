"""
SEED-VII Dataset Loader for Multimodal EEG-Eye Emotion Recognition
Handles loading and preprocessing of SEED-VII dataset with EEG and eye tracking data
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split


class SEEDVII_Dataset(Dataset):
    """
    SEED-VII dataset loader for multimodal emotion recognition.
    
    The SEED-VII dataset contains EEG and eye tracking data from 20 subjects
    watching 80 emotional video clips across 7 emotion categories.
    
    Args:
        data_dir (str): Path to dataset directory containing EEG_features and EYE_features folders
        modality (str): Data modality - 'eeg', 'eye', or 'multimodal' (default: 'multimodal')
        subset_ratio (float): Fraction of dataset to use (default: 0.01 for quick testing)
    """
    
    def __init__(self, data_dir: str = ".", modality: str = 'multimodal', 
                 subset_ratio: float = 0.01):
        self.data_dir = data_dir
        self.modality = modality
        self.subset_ratio = subset_ratio
        
        # Dataset specifications based on SEED-VII paper
        self.num_classes = 7  # Seven emotion categories
        self.num_subjects = 20
        self.eeg_feature_dim = 310  # EEG differential entropy features
        self.eye_feature_dim = 33   # Eye tracking features
        
        # Emotion labels mapping
        self.emotion_labels = {
            0: 'neutral', 1: 'sad', 2: 'fear', 3: 'happy',
            4: 'disgust', 5: 'surprise', 6: 'angry'
        }
        
        # Load and process data
        self._load_data()
        if self.subset_ratio < 1.0:
            self._create_subset()
    
    def _load_data(self):
        """Load EEG and eye movement features from .mat files"""
        print(f"Loading SEED-VII dataset from {self.data_dir}")
        
        eeg_data, eye_data = [], []
        emotion_labels, subject_labels = [], []
        
        # Get feature files
        eeg_dir = os.path.join(self.data_dir, 'EEG_features')
        eye_dir = os.path.join(self.data_dir, 'EYE_features')
        
        eeg_files = sorted(glob.glob(os.path.join(eeg_dir, '*.mat')))
        eye_files = sorted(glob.glob(os.path.join(eye_dir, '*.mat')))
        
        print(f"Found {len(eeg_files)} EEG files and {len(eye_files)} eye files")
        
        # Get emotion mapping for video clips
        emotion_map = self._get_emotion_mapping()
        
        # Process each subject
        for subject_idx, eeg_file in enumerate(eeg_files[:self.num_subjects]):
            subject_name = os.path.basename(eeg_file).replace('.mat', '')
            print(f"Loading subject {subject_idx + 1}: {subject_name}")
            
            # Load corresponding files
            try:
                eeg_mat = sio.loadmat(eeg_file)
                eye_file = self._find_matching_eye_file(subject_name, eye_files)
                if eye_file:
                    eye_mat = sio.loadmat(eye_file)
                else:
                    continue
            except Exception as e:
                print(f"Error loading files for {subject_name}: {e}")
                continue
            
            # Process all 80 video clips
            for video_id in range(1, 81):
                eeg_features, eye_features = self._extract_features(
                    eeg_mat, eye_mat, video_id)
                
                if eeg_features is not None and eye_features is not None:
                    min_windows = min(len(eeg_features), len(eye_features))
                    if min_windows > 0:
                        eeg_data.append(eeg_features[:min_windows])
                        eye_data.append(eye_features[:min_windows])
                        
                        emotion_label = emotion_map.get(video_id, 6)
                        emotion_labels.extend([emotion_label] * min_windows)
                        subject_labels.extend([subject_idx] * min_windows)
        
        # Convert to numpy arrays
        self.eeg_features = np.vstack(eeg_data)
        self.eye_features = np.vstack(eye_data)
        self.emotion_labels = np.array(emotion_labels)
        self.subject_labels = np.array(subject_labels)
        
        print(f"Dataset loaded: {len(self.emotion_labels)} samples")
        print(f"EEG shape: {self.eeg_features.shape}, Eye shape: {self.eye_features.shape}")
    
    def _get_emotion_mapping(self):
        """Create emotion mapping for video clips based on SEED-VII protocol"""
        emotion_map = {}
        
        # Simplified mapping based on experimental protocol
        # 4 sessions × 20 clips per session = 80 total clips
        for session in range(4):
            # Emotion sequence varies by session
            emotions = [0, 6, 3, 1, 5, 2, 4][:7] if session % 2 == 0 else [5, 1, 2, 6, 0, 4, 3]
            for i, emotion in enumerate(emotions):
                for video in range(4):  # 4 videos per emotion per session
                    video_id = session * 20 + i * 4 + video + 1
                    if video_id <= 80:
                        emotion_map[video_id] = emotion % 7
        
        return emotion_map
    
    def _find_matching_eye_file(self, subject_name, eye_files):
        """Find the matching eye tracking file for a given subject"""
        for eye_file in eye_files:
            if subject_name in os.path.basename(eye_file):
                return eye_file
        return None
    
    def _extract_features(self, eeg_mat, eye_mat, video_id):
        """Extract features for a specific video ID from loaded .mat files"""
        video_key = str(video_id)
        
        # Try different key formats for EEG features
        eeg_features = None
        for key in [f'de_LDS_{video_id}', f'de_{video_id}', video_key]:
            if key in eeg_mat:
                eeg_features = eeg_mat[key]
                break
        
        # Try different key formats for Eye features
        eye_features = None
        for key in [video_key, str(video_id)]:
            if key in eye_mat:
                eye_features = eye_mat[key]
                break
        
        # Process EEG features
        if eeg_features is not None:
            if eeg_features.ndim == 3:
                eeg_features = eeg_features.reshape(eeg_features.shape[0], -1)
            if eeg_features.shape[1] != self.eeg_feature_dim:
                eeg_features = None
        
        # Process eye features
        if eye_features is not None and eye_features.shape[1] != self.eye_feature_dim:
            eye_features = None
        
        return eeg_features, eye_features
    
    def _create_subset(self):
        """Create a subset of the dataset for faster training/testing"""
        n_samples = len(self.emotion_labels)
        subset_size = max(1, int(n_samples * self.subset_ratio))
        
        try:
            # Stratified sampling to maintain class distribution
            indices = np.arange(n_samples)
            subset_indices, _ = train_test_split(
                indices, train_size=subset_size, 
                stratify=self.emotion_labels, random_state=42)
        except:
            # Fallback to random sampling if stratification fails
            subset_indices = np.random.choice(n_samples, subset_size, replace=False)
        
        # Apply subset selection
        self.eeg_features = self.eeg_features[subset_indices]
        self.eye_features = self.eye_features[subset_indices]
        self.emotion_labels = self.emotion_labels[subset_indices]
        self.subject_labels = self.subject_labels[subset_indices]
        
        print(f"Created {self.subset_ratio*100:.1f}% subset: {len(self.emotion_labels)} samples")
    
    def __len__(self):
        """Return the total number of samples"""
        return len(self.emotion_labels)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset
        
        Args:
            idx (int): Sample index
            
        Returns:
            dict: Sample containing modality features, emotion label, and subject ID
        """
        sample = {}
        
        # Add features based on modality
        if self.modality in ['eeg', 'multimodal']:
            sample['eeg'] = torch.FloatTensor(self.eeg_features[idx])
        if self.modality in ['eye', 'multimodal']:
            sample['eye'] = torch.FloatTensor(self.eye_features[idx])
        
        # Add labels
        sample['label'] = torch.LongTensor([self.emotion_labels[idx]])[0]
        sample['subject'] = torch.LongTensor([self.subject_labels[idx]])[0]
        
        return sample
    
    def get_class_weights(self):
        """Calculate class weights for handling imbalanced data"""
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(self.emotion_labels), 
            y=self.emotion_labels
        )
        return torch.FloatTensor(class_weights)
    
    def get_subject_info(self):
        """Get information about subjects in the dataset"""
        unique_subjects = np.unique(self.subject_labels)
        subject_counts = {sub: np.sum(self.subject_labels == sub) for sub in unique_subjects}
        return subject_counts
```
