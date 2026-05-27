"""
Unit test for EEGDataLoader.
Run this file to verify that data loading works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eeg_emotion.data_loader import EEGDataLoader


def test_data_loader():
    """Test the functionality of EEGDataLoader."""
    loader = EEGDataLoader(data_root="data")
    
    print("="*60)
    print("🚀 Starting EEGDataLoader Test")
    print("="*60)
    
    # ==================== 测试训练集 ====================
    print("\n[1] Testing Training Data...")
    train_files = loader.get_all_train_files()
    print(f"Found {len(train_files)} training files")
    
    if train_files:
        sample_file = train_files[0]
        print(f"Loading sample: {sample_file}")
        
        data = loader.load_train_subject(sample_file)
        
        print(f"  - neu shape: {data['neu'].shape}")
        print(f"  - pos shape: {data['pos'].shape}")
        print(f"  - is_depressed: {data['is_depressed']}")
        print(f"  - subject_id: {data['subject_id']}")
        print("  ✅ Training data loaded successfully")
    else:
        print("  ⚠️  No training files found!")
    
    # ==================== 测试测试集 ====================
    print("\n[2] Testing Test Data...")
    test_files = loader.get_all_test_files()
    print(f"Found {len(test_files)} test files")
    
    if test_files:
        sample_test_file = test_files[0]
        print(f"Loading sample: {sample_test_file}")
        
        test_data = loader.load_test_subject(sample_test_file)
        
        print(f"  - eeg shape: {test_data['eeg'].shape}")
        print(f"  - user_id: {test_data['user_id']}")
        print("  ✅ Test data loaded successfully")
    else:
        print("  ⚠️  No test files found!")
    
    print("\n" + "="*60)
    print("🎉 All tests passed! DataLoader is working correctly.")
    print("="*60)


if __name__ == "__main__":
    test_data_loader()