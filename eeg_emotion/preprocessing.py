"""
EEG Signal Preprocessing Module.

This module handles subject-specific normalization and sliding window segmentation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional


class EEGPreprocessor:
    """EEG signal preprocessing utilities."""

    def __init__(self, 
                 sampling_rate: int = 250,
                 window_size: int = 500,      # 2 seconds
                 step_size: int = 250,        # 1 second overlap
                 zscore_per_subject: bool = True,
                 zscore_per_channel: bool = True):
        """
        Args:
            sampling_rate (int): Sampling rate of EEG data (Hz).
            window_size (int): Sliding window length in samples.
            step_size (int): Step size between windows.
            zscore_per_subject (bool): Whether to normalize per subject.
            zscore_per_channel (bool): Whether to normalize per channel.
        """
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.step_size = step_size
        self.zscore_per_subject = zscore_per_subject
        self.zscore_per_channel = zscore_per_channel

    def zscore_normalize(self, signal: np.ndarray, 
                        mean: Optional[np.ndarray] = None, 
                        std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform subject-level or channel-level z-score normalization.

        Args:
            signal: EEG data of shape (n_channels, n_samples)
            mean, std: Pre-computed statistics (for test set consistency)

        Returns:
            normalized_signal, mean, std
        """
        signal = signal.astype(np.float32)
        
        if self.zscore_per_channel:
            # Per channel normalization
            if mean is None:
                mean = np.mean(signal, axis=1, keepdims=True)
                std = np.std(signal, axis=1, keepdims=True) + 1e-8
            normalized = (signal - mean) / std
        else:
            # Global normalization
            if mean is None:
                mean = np.mean(signal)
                std = np.std(signal) + 1e-8
            normalized = (signal - mean) / std
            
        return normalized, mean, std

    def create_sliding_windows(self, signal: np.ndarray, label: int) -> List[Tuple[np.ndarray, int]]:
        """
        Create sliding windows from a long EEG segment.

        Args:
            signal: EEG data of shape (n_channels, n_samples)
            label: Emotion label (0 or 1)

        Returns:
            List of (window, label) tuples
        """
        windows = []
        n_samples = signal.shape[1]
        
        for start in range(0, n_samples - self.window_size + 1, self.step_size):
            window = signal[:, start:start + self.window_size]   # (30, window_size)
            windows.append((window, label))
        
        return windows


class EEGDataset(Dataset):
    """PyTorch Dataset for EEG emotion recognition with sliding windows."""

    def __init__(self, 
                 data_loader,
                 subject_files: List[str],
                 preprocessor: EEGPreprocessor,
                 is_test: bool = False):
        """
        Args:
            data_loader: EEGDataLoader instance
            subject_files: List of .mat filenames
            preprocessor: EEGPreprocessor instance
            is_test: Whether this is test set
        """
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.is_test = is_test
        self.samples = []          # List of (eeg_window, label, subject_id, is_depressed)
        
        self._build_dataset(subject_files)

    def _build_dataset(self, subject_files: List[str]):
        """Build dataset by loading subjects and creating sliding windows."""
        for filename in subject_files:
            if self.is_test:
                # Test set
                data = self.data_loader.load_test_subject(filename)
                eeg = data['eeg']                    # (30, 20000)
                user_id = data['user_id']
                
                # For test set, we don't know label yet, will predict
                windows = self.preprocessor.create_sliding_windows(eeg, label=-1)
                for window, _ in windows:
                    self.samples.append((window, -1, user_id, -1))
            else:
                # Training set
                data = self.data_loader.load_train_subject(filename)
                is_depressed = data['is_depressed']
                subject_id = data['subject_id']
                
                # Neutral trials (label = 0)
                for neu_trial in self._split_trials(data['neu']):
                    windows = self.preprocessor.create_sliding_windows(neu_trial, label=0)
                    for window, label in windows:
                        self.samples.append((window, label, subject_id, is_depressed))
                
                # Positive trials (label = 1)
                for pos_trial in self._split_trials(data['pos']):
                    windows = self.preprocessor.create_sliding_windows(pos_trial, label=1)
                    for window, label in windows:
                        self.samples.append((window, label, subject_id, is_depressed))

    def _split_trials(self, long_signal: np.ndarray) -> List[np.ndarray]:
        """Split 50s signal into 4 trials (for training set)."""
        trial_length = 12500  # 50s * 250Hz
        return [long_signal[:, i*trial_length:(i+1)*trial_length] 
                for i in range(4)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window, label, subject_id, is_depressed = self.samples[idx]
        
        # Convert to torch tensor
        x = torch.from_numpy(window).float()          # (30, window_size)
        y = torch.tensor(label, dtype=torch.long)
        domain = torch.tensor(is_depressed, dtype=torch.long)
        
        return {
            'eeg': x,
            'label': y,
            'domain': domain,
            'subject_id': subject_id
        }