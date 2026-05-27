# EEG Emotion Recognition

**第十一届全国大学生生物医学工程创新设计竞赛** —— 脑机接口赛道（赛题四）

基于脑电信号的跨被试情绪识别（积极 vs 中性），重点解决健康与抑郁被试的域偏移问题。

## 项目特点
- 三种方案系统对比（传统特征工程 vs 端到端 vs 混合架构）
- 多源域适应（DANN） + 多任务学习（情绪 + 健康/抑郁）
- 支持 MATLAB v7 & v7.3 格式
- 模块化设计，易于扩展

## 项目结构
<pre>
EEG_Emotion_Recognition/
├── data/                    # 原始数据（.gitignore）
├── eeg_emotion/             # 主代码包
├── configs/                 # 配置
├── experiments/             # 实验记录
├── results/                 # 结果与提交
├── notebooks/               # 探索性分析
├── tests/                   # 测试
├── train.py                 # 训练入口
├── predict.py               # 预测入口
├── README.md
├── requirements.txt
└── .gitignore
</pre>

## 快速开始

```bash
conda create -n eeg python=3.10
conda activate eeg

# PyTorch（根据环境选择）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu   # CPU版
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # GPU版

pip install -r requirements.txt

# 测试数据加载
python -m tests.test_data_loader
