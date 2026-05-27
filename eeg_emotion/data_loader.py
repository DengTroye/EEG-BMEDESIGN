"""
EEG Data Loader for Emotion Recognition Competition.

This module handles loading of .mat files from both training and test sets.
It supports MATLAB v7 and v7.3 formats and uses robust key detection.
"""

import os
from typing import Dict, List

import numpy as np
from scipy.io import loadmat


class EEGDataLoader:
    """Data loader for the EEG emotion recognition dataset."""

    def __init__(self, data_root: str = "data"):
        """
        Args:
            data_root (str): Root directory containing 'train' and 'test_public' folders.
        """
        self.data_root = data_root
        self.sampling_rate = 250
        self.n_channels = 30

    def _load_mat(self, filepath: str) -> dict:
        """Load .mat file, supporting both v7 and v7.3 formats."""
        try:
            # Try standard scipy loader first
            return loadmat(filepath, squeeze_me=True)
        except NotImplementedError:
            # Fallback for MATLAB v7.3 (HDF5) files
            import h5py
            with h5py.File(filepath, 'r') as f:
                data = {}
                for key in f.keys():
                    if not key.startswith('#'):
                        dset = f[key]
                        data[key] = np.array(dset).T if isinstance(dset, h5py.Dataset) else dset
                return data

    def _find_key(self, mat_dict: dict, candidates: List[str], filepath: str) -> str:
        """Find the correct key from candidate list."""
        for candidate in candidates:
            if candidate in mat_dict:
                return candidate
        
        raise KeyError(
            f"None of the expected keys {candidates} found in {filepath}. "
            f"Available keys: {list(mat_dict.keys())}"
        )

    def load_train_subject(self, filename: str) -> Dict:
        """
        Load a single training subject.

        Args:
            filename (str): Filename like "HC1001timedata.mat" or "DEP1003timedata.mat"

        Returns:
            Dict containing neu, pos, is_depressed, subject_id, etc.
        """
        filepath = os.path.join(self.data_root, "train", filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Training file not found: {filepath}")

        mat = self._load_mat(filepath)

        neu_key = self._find_key(mat, ['EEG_data_neu', 'neu'], filepath)
        pos_key = self._find_key(mat, ['EEG_data_pos', 'pos'], filepath)

        is_depressed = 1 if filename.startswith("DEP") else 0
        subject_id = filename.replace("timedata.mat", "").replace(".mat", "")

        return {
            'neu': mat[neu_key],
            'pos': mat[pos_key],
            'is_depressed': is_depressed,
            'subject_id': subject_id,
            'filename': filename
        }

    def load_test_subject(self, filename: str) -> Dict:
        """
        Load a single test subject.

        Args:
            filename (str): Filename like "P_test1.mat"

        Returns:
            Dict containing eeg data and user_id.
        """
        filepath = os.path.join(self.data_root, "test_public", filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Test file not found: {filepath}")

        mat = self._load_mat(filepath)

        # Test set key is 'test_eeg_c'
        eeg_key = self._find_key(mat, ['test_eeg_c', 'EEG_data', 'eeg', 'data'], filepath)
        user_id = filename.replace(".mat", "")

        return {
            'eeg': mat[eeg_key],
            'user_id': user_id,
            'filename': filename
        }

    def get_all_train_files(self) -> List[str]:
        """Return sorted list of all training .mat filenames."""
        train_dir = os.path.join(self.data_root, "train")
        return sorted([f for f in os.listdir(train_dir) if f.endswith('.mat')])

    def get_all_test_files(self) -> List[str]:
        """Return sorted list of all test .mat filenames."""
        test_dir = os.path.join(self.data_root, "test_public")
        return sorted([f for f in os.listdir(test_dir) if f.endswith('.mat')])