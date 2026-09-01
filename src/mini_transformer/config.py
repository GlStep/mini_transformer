from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    pairs: list[str]
    raw_dir: str
    processed_dir: str
    download_sample_size: int | None


@dataclass
class TokenizerConfig:
    model_path: str
    vocab_size: int


@dataclass
class ModelConfig:
    d_model: int
    n_heads: int
    d_ff: int
    n_layers: int
    dropout: float
    max_len: int


@dataclass
class TrainingConfig:
    batch_size: int
    num_epochs: int
    grad_clip: float
    warmup_steps: int
    checkpoint_dir: str


@dataclass
class Config:
    data: DataConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    training: TrainingConfig


def load_config(path: str | Path = "config.yaml") -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(
        data=DataConfig(**raw["data"]),
        tokenizer=TokenizerConfig(**raw["tokenizer"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
    )