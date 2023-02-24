import os
from sklearn.model_selection import KFold, StratifiedKFold
import torch
import pytorch_lightning as pl
from torchvision.datasets import ImageFolder

from torch.utils import data as udata
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    RichProgressBar,
    RichModelSummary,
)

import numpy as np
import wandb
from modules.classifier import ClassifierModel
import hydra
from omegaconf import DictConfig

from datetime import datetime
from rich import print


data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.4782, 0.5025, 0.4018], [0.2095, 0.1745, 0.2315]),
        ]
    ),
    "valid": transforms.Compose(
        [
            # transforms.Resize(224),
            # transforms.CenterCrop(224),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.4782, 0.5025, 0.4018], [0.2095, 0.1745, 0.2315]),
        ]
    ),
}


class MySubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.indices)


def setup_dataloader(dataset, train_idx, val_idx, batch_size, generator):
    train_sampler = udata.SubsetRandomSampler(train_idx, generator=generator)
    val_sampler = udata.SubsetRandomSampler(val_idx, generator=generator)
    train_loader = DataLoader(
        dataset,
        batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        dataset,
        batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        sampler=val_sampler,
    )
    return train_loader, val_loader


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.trainer.seed)
    kf = KFold(n_splits=5, shuffle=True, random_state=cfg.trainer.seed)

    print("split dataset load")
    train_all_dataset = ImageFolder(
        cfg.dataset.path.train, transform=data_transforms["train"]
    )

    if cfg.dataset.aug is not None:
        print("augment dataset concatting")
        augment_dataset = ImageFolder(
            cfg.dataset.aug.path, transform=data_transforms["train"]
        )
        train_all_dataset = udata.ConcatDataset([train_all_dataset, augment_dataset])

    current_time = datetime.now()
    save_dir = f"{cfg.trainer.save_dir}/{current_time:%Y-%m-%d}/{current_time:%H-%M-%S}"

    # K fold cross-validation (K=5)
    for fold, (train_idx, val_idx) in enumerate(
        kf.split(np.arange(len(train_all_dataset)))
    ):
        # データセットの設定
        test_dataset = ImageFolder(
            cfg.dataset.path.test, transform=data_transforms["valid"]
        )

        # データローダーの設定
        train_loader, val_loader = setup_dataloader(
            train_all_dataset,
            train_idx,
            val_idx,
            cfg.models.params.batch_size,
            generator=torch.Generator().manual_seed(cfg.trainer.seed),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.models.params.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
        )

        # 保存先の設定
        experiment_name = f"{cfg.models.params.model_name}: fold-{fold}"
        save_path = f"{save_dir}/{experiment_name}"
        os.makedirs(save_path, exist_ok=True)

        # モデルの構築
        net = ClassifierModel(
            cfg,
            cfg.trainer.num_class,
            cfg.models.params.model_name,
        )

        # setting callbacks
        ckptCallback = ModelCheckpoint(
            monitor="loss/val_loss",
            mode="min",
            dirpath=save_path,
            filename="{epoch}--{val_loss:.3f}",
        )
        earlyStoppingCallback = EarlyStopping(
            monitor="loss/val_loss",
            patience=3,
            mode="min",
            min_delta=0.01,
        )

        # setting logger
        wandb_logger = WandbLogger(
            save_dir=save_path,
            project=cfg.wandb.project,
            group=f"{current_time:%m-%d}--model--{cfg.models.params.model_name}--dataset--{cfg.dataset.name}",
            job_type=cfg.wandb.job_type,
            tags=[cfg.models.params.model_name, cfg.dataset.name],
            name=experiment_name,
        )

        # trainerの設定
        trainer = pl.Trainer(
            max_epochs=cfg.trainer.max_epochs,
            logger=[wandb_logger],
            callbacks=[
                earlyStoppingCallback,
                ckptCallback,
                RichModelSummary(),
                RichProgressBar(),
            ],
            devices="auto",
            accelerator="gpu",
            # gradient_clip_val=0.5,
        )
        # 学習
        trainer.fit(
            model=net,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        trainer.test(net, test_loader, "best")
        del net
        del trainer
        wandb.finish()
    return None


if __name__ == "__main__":
    print("🐬evaluation start.🐬")
    main()
    print("evaluation finished.😂")
