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
    max_epochs: int = 20
    num_classes: int = 10
    seed: int = 42
    save_dir: str = "models"
    optim_name: str = "SGD"
    lr: float = 0.01
    momentum: float = 0.9


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