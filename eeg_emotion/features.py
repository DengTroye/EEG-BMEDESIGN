"""
EEG Feature Extraction Module.

This module provides PSD (Power Spectral Density) and DE (Differential Entropy)
feature extraction, which are classic and effective features for EEG emotion recognition.
"""

import numpy as np
from scipy import signal
from typing import Dict


class EEGFeatureExtractor:
    """EEG feature extraction utilities."""

    def __init__(self, sampling_rate: int = 250):
        """
        Args:
            sampling_rate (int): Sampling rate of the EEG signal in Hz.
        """
        self.sampling_rate = sampling_rate
        
        # Standard EEG frequency bands
        self.freq_bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }

    def extract_psd(self, 
                    eeg_signal: np.ndarray, 
                    nperseg: int = 500, 
                    noverlap: int = None) -> np.ndarray:
        """
        Extract PSD features for 5 frequency bands using Welch's method.

        Args:
            eeg_signal: EEG signal of shape (n_channels, n_samples)

        Returns:
            PSD features of shape (n_channels, 5)
        """
        if noverlap is None:
            noverlap = nperseg // 2

        n_channels = eeg_signal.shape[0]
        psd_features = np.zeros((n_channels, len(self.freq_bands)))

        for ch in range(n_channels):
            freqs, psd = signal.welch(
                eeg_signal[ch], 
                fs=self.sampling_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                scaling='density'
            )
            
            for band_idx, (low, high) in enumerate(self.freq_bands.values()):
                idx = np.logical_and(freqs >= low, freqs <= high)
                if np.any(idx):
                    psd_features[ch, band_idx] = np.mean(psd[idx])

        return psd_features

    def extract_de(self, eeg_signal: np.ndarray, window_size: int = 500) -> np.ndarray:
        """
        Extract Differential Entropy (DE) features.

        Args:
            eeg_signal: EEG signal of shape (n_channels, n_samples)

        Returns:
            DE features of shape (n_channels, n_windows)
        """
        n_channels, n_samples = eeg_signal.shape
        n_windows = (n_samples - window_size) // window_size + 1
        de_features = np.zeros((n_channels, n_windows))

        for ch in range(n_channels):
            for i in range(n_windows):
                start = i * window_size
                segment = eeg_signal[ch, start:start + window_size]
                variance = np.var(segment) + 1e-8
                de_features[ch, i] = 0.5 * np.log(2 * np.pi * np.e * variance)

        return de_features

    def extract_features(self, eeg_signal: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract both PSD and DE features.

        Args:
            eeg_signal: EEG signal of shape (n_channels, n_samples)

        Returns:
            Dictionary containing 'psd', 'de', and 'combined' features.
        """
        psd = self.extract_psd(eeg_signal)
        de = self.extract_de(eeg_signal, window_size=500)
        
        # Average DE across windows to match PSD shape
        de_mean = np.mean(de, axis=1, keepdims=True)
        combined = np.hstack([psd, de_mean])

        return {
            'psd': psd,           # (30, 5)
            'de': de,             # (30, n_windows)
            'combined': combined  # (30, 6)
        }