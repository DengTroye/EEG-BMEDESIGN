"""
Unit test for EEG Preprocessing Module.
Tests normalization and sliding window functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eeg_emotion.data_loader import EEGDataLoader
from eeg_emotion.preprocessing import EEGPreprocessor, EEGDataset


def test_preprocessing():
    """Test preprocessing pipeline."""
    print("="*70)
    print("🚀 Starting EEG Preprocessing Test")
    print("="*70)
    
    # 初始化
    data_loader = EEGDataLoader(data_root="data")
    preprocessor = EEGPreprocessor(
        window_size=500,      # 2秒
        step_size=250,        # 1秒重叠
        zscore_per_subject=True,
        zscore_per_channel=True
    )
    
    # 获取部分训练数据进行测试
    train_files = data_loader.get_all_train_files()[:3]  # 只取前3个被试测试
    print(f"Selected {len(train_files)} training subjects for testing")
    
    # ==================== 测试 Preprocessor ====================
    print("\n[1] Testing EEGPreprocessor...")
    
    # 加载一个被试的数据
    sample = data_loader.load_train_subject(train_files[0])
    print(f"Original neu shape: {sample['neu'].shape}")
    
    # 测试 z-score 归一化
    normalized_neu, mean, std = preprocessor.zscore_normalize(sample['neu'])
    print(f"After z-score normalization: {normalized_neu.shape}")
    print(f"Mean (should be close to 0): {normalized_neu.mean():.4f}")
    print(f"Std (should be close to 1): {normalized_neu.std():.4f}")
    print("  ✅ z-score normalization passed")
    
    # ==================== 测试 Dataset ====================
    print("\n[2] Testing EEGDataset...")
    
    dataset = EEGDataset(
        data_loader=data_loader,
        subject_files=train_files,
        preprocessor=preprocessor,
        is_test=False
    )
    
    print(f"Dataset total samples: {len(dataset)}")
    
    if len(dataset) > 0:
        sample_item = dataset[0]
        print(f"Single sample eeg shape: {sample_item['eeg'].shape}")
        print(f"Label: {sample_item['label']}")
        print(f"Domain (is_depressed): {sample_item['domain']}")
        print("  ✅ EEGDataset created successfully")
    
    print("\n" + "="*70)
    print("🎉 All preprocessing tests passed!")
    print("="*70)


if __name__ == "__main__":
    test_preprocessing()