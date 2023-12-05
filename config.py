from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_name: str



@dataclass
class TrainerConfig:
    augment: bool
    pretrained: bool
    batch_size: int
    num_workers: int
    max_epochs: int
    num_classes: int
    seed: int
    save_dir: str
    optim_name: str
    lr: float
    momentum: float


@dataclass
class DatasetConfig:
    name: str
    train_path: str
    test_path: str
    aug_path: str

@dataclass
class WandbConfig:
    project: str
    job_type: str


@dataclass
class Config:
    wandb: WandbConfig = WandbConfig()
    dataset: DatasetConfig = DatasetConfig()
    models: ModelConfig = ModelConfig()
    trainer: TrainerConfig = TrainerConfig()