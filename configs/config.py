"""
Configuration file for EEG Emotion Recognition Project.
Centralized management of paths, hyperparameters, and experiment settings.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    """Data related configuration."""
    data_root: str = "data"
    train_dir: str = "train"
    test_dir: str = "test_public"
    sampling_rate: int = 250
    n_channels: int = 30


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration."""
    window_size: int = 500      # 2 seconds (250Hz * 2)
    step_size: int = 250        # 1 second overlap
    zscore_per_subject: bool = True
    zscore_per_channel: bool = True


@dataclass
class ModelConfig:
    """Model hyperparameters."""
    num_classes: int = 2
    dropout: float = 0.3
    # CNN-Transformer specific
    cnn_out_channels: int = 64
    transformer_layers: int = 4
    transformer_heads: int = 8


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 50
    use_dann: bool = True
    use_multi_task: bool = True
    lambda_dann: float = 0.5      # Weight for domain adversarial loss


class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.data = DataConfig()
        self.preprocessing = PreprocessingConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        
        # Paths
        self.project_root = Path(__file__).parent.parent
        self.data_root = self.project_root / self.data.data_root


# Global config instance
config = Config()