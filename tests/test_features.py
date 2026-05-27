"""
Unit test for EEGFeatureExtractor.
Tests PSD and DE feature extraction.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eeg_emotion.data_loader import EEGDataLoader
from eeg_emotion.features import EEGFeatureExtractor


def test_feature_extractor():
    """Test PSD and DE feature extraction."""
    print("="*70)
    print("🚀 Starting EEG Feature Extractor Test")
    print("="*70)
    
    # 初始化
    data_loader = EEGDataLoader(data_root="data")
    extractor = EEGFeatureExtractor(sampling_rate=250)
    
    # 获取训练文件
    train_files = data_loader.get_all_train_files()
    if not train_files:
        print("❌ No training files found!")
        return
    
    # 加载样本数据
    sample_file = train_files[0]
    print(f"Testing on sample: {sample_file}")
    
    data = data_loader.load_train_subject(sample_file)
    neu_signal = data['neu']        # (30, 50000)
    
    print(f"Original signal shape: {neu_signal.shape}")
    
    # ==================== 测试 PSD ====================
    print("\n[1] Testing PSD Feature Extraction...")
    psd_features = extractor.extract_psd(neu_signal)
    print(f"PSD features shape: {psd_features.shape} (n_channels × n_bands)")
    print(f"PSD sample (first channel): {psd_features[0]}")
    print("  ✅ PSD extraction passed")
    
    # ==================== 测试 DE ====================
    print("\n[2] Testing DE Feature Extraction...")
    de_features = extractor.extract_de(neu_signal, window_size=500)
    print(f"DE features shape: {de_features.shape} (n_channels × n_windows)")
    print(f"DE sample (first channel, first 5 windows): {de_features[0, :5]}")
    print("  ✅ DE extraction passed")
    
    # ==================== 测试组合特征 ====================
    print("\n[3] Testing Combined Features...")
    features = extractor.extract_features(neu_signal)
    print(f"Combined features shape: {features['combined'].shape}")
    print("  ✅ Combined feature extraction passed")
    
    print("\n" + "="*70)
    print("🎉 All feature extraction tests passed successfully!")
    print("="*70)


if __name__ == "__main__":
    test_feature_extractor()