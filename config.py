from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_name: str = ""


@dataclass
class TrainerConfig:
    augment: bool = False
    pretrained: bool = False
    batch_size: int = 32
    num_workers: int = 4
    num_classes: int = 10
    max_epochs: int = 20
    seed: int = 42
    optim_name: str = "SGD"
    momentum: float = 0.9
    lr: float = 0.01
    num_folds: int = 5
    early_stopping_monitor: str = "loss/val_loss"
    early_stopping_mode: str = "min"
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.01
    logger: str = "wandb"
    save_dir: str = "models"


@dataclass
class DatasetConfig:
    name: str = ""
    train_path: str = ""
    test_path: str = ""
    aug_path: str | None = None


@dataclass
class WandbConfig:
    project: str = "pl-classifier"
    job_type: str = "eval"


@dataclass
class Config:
    wandb: WandbConfig = WandbConfig()
    dataset: DatasetConfig = DatasetConfig()
    models: ModelConfig = ModelConfig()
    trainer: TrainerConfig = TrainerConfig()
