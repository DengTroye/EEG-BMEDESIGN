import sys
import os
import joblib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from eeg_emotion.data_loader import EEGDataLoader
from eeg_emotion.preprocessing import EEGPreprocessor
from eeg_emotion.features import EEGFeatureExtractor
from xgboost import XGBClassifier

def main():
    print("程序开始运行...")
    
    data_root = "data"
    loader = EEGDataLoader(data_root=data_root)
    preprocessor = EEGPreprocessor(window_size=500, step_size=250)
    extractor = EEGFeatureExtractor(sampling_rate=250)
    
    Path("results/models").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    
    train_files = loader.get_all_train_files()
    print(f"✅ 共找到 {len(train_files)} 个训练 subject")
    
    test_files = train_files
    accuracies = []
    
    print("开始 LOSO 交叉验证 (全部 60 个 subject)...")
    print("这可能需要几分钟时间，请耐心等待...\n")
    
    for i, test_file in enumerate(tqdm(test_files)):
        X_train_list, y_train_list = [], []
        X_test = None
        y_test = None
        
        for train_file in test_files:
            data_dict = loader.load_train_subject(train_file)
            
            signal = None
            if isinstance(data_dict, dict):
                if 'neu' in data_dict:
                    signal = data_dict['neu']
                elif 'pos' in data_dict:
                    signal = data_dict['pos']
                
                label = int(data_dict.get('is_depressed', 0))
            
            if signal is None:
                continue
            
            if len(signal.shape) == 1:
                signal = signal.reshape(1, -1)
            elif signal.shape[0] > signal.shape[1]:
                signal = signal.T
            
            normalized, _, _ = preprocessor.zscore_normalize(signal)
            features_dict = extractor.extract_features(normalized)
            feat = features_dict['combined'].flatten()
            
            if train_file == test_file:
                X_test = feat.reshape(1, -1)
                y_test = label
            else:
                X_train_list.append(feat)
                y_train_list.append(label)
        
        if X_test is None or len(X_train_list) < 10:
            continue
        
        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)
        
        unique_labels = np.unique(y_train)
        if len(unique_labels) < 2:
            continue
        
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)[0]
        acc = 1.0 if y_pred == y_test else 0.0
        accuracies.append(acc)
    
    # ================== 最终结果 ==================
    if accuracies:
        mean_acc = np.mean(accuracies)
        print("\n" + "="*70)
        print("🎉 **方案1 (PSD + DE + XGBoost) LOSO Baseline 完成**")
        print("="*70)
        print(f"总 subject 数: {len(train_files)}")
        print(f"成功评估折数: {len(accuracies)}")
        print(f"**平均准确率: {mean_acc:.4f} ({mean_acc*100:.2f}%)**")
        print("="*70)
        
        # 保存最终模型
        joblib.dump(model, "results/models/traditional_xgb_loso_final.pkl")
        print("✅ 模型已保存: results/models/traditional_xgb_loso_final.pkl")
        
                # ================== 新增：特征重要性分析 ==================
        print("\n正在生成特征重要性图（用于报告）...")
        try:
            # 创建 features 文件夹
            Path("results/features").mkdir(parents=True, exist_ok=True)
            
            # XGBoost 自带特征重要性
            importance = model.feature_importances_
            
            # 绘制柱状图
            plt.figure(figsize=(12, 8))
            plt.bar(range(len(importance)), importance)
            plt.title("XGBoost Feature Importance (PSD + DE Features)")
            plt.xlabel("Feature Index (0-179)")
            plt.ylabel("Importance Score")
            plt.savefig("results/figures/feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 保存重要性数值
            np.save("results/features/feature_importance.npy", importance)
            
            # 打印 Top 10 重要特征
            top10_idx = np.argsort(importance)[-10:][::-1]
            print("Top 10 重要特征索引:", top10_idx.tolist())
            print("对应重要性分数:", [round(float(x), 4) for x in importance[top10_idx].tolist()])
            
            print("✅ 特征重要性图已保存:")
            print("   📊 results/figures/feature_importance.png")
            print("   📁 results/features/feature_importance.npy")
            
        except Exception as e:
            print(f"⚠️ 特征重要性生成失败: {e}")
    
    else:
        print("\n❌ 未能完成评估")

if __name__ == "__main__":
    main()