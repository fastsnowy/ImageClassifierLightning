import os
from datetime import datetime

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from rich import print
from sklearn.model_selection import KFold
from torch.utils import data as udata
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from config import Config
from modules.classifier import ClassifierModel

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


def setup_dataloader(dataset, train_idx, val_idx, batch_size, generator, num_workers):
    train_sampler = udata.SubsetRandomSampler(train_idx, generator=generator)
    val_sampler = udata.SubsetRandomSampler(val_idx, generator=generator)
    train_loader = DataLoader(
        dataset,
        batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        dataset,
        batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        sampler=val_sampler,
    )
    return train_loader, val_loader


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config) -> None:
    pl.seed_everything(cfg.trainer.seed)
    kf = KFold(
        n_splits=cfg.trainer.num_folds,
        shuffle=True,
        random_state=cfg.trainer.seed,
    )

    print("split dataset load")
    train_all_dataset = ImageFolder(
        cfg.dataset.train_path,
        transform=data_transforms["train"],
    )

    num_classes = len(train_all_dataset.class_to_idx.keys())

    if cfg.trainer.augment:
        print("augment dataset concatting")
        augment_dataset = ImageFolder(
            cfg.dataset.aug_path,
            transform=data_transforms["train"],
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
            cfg.dataset.test_path,
            transform=data_transforms["valid"],
        )

        # データローダーの設定
        train_loader, val_loader = setup_dataloader(
            train_all_dataset,
            train_idx,
            val_idx,
            cfg.trainer.batch_size,
            generator=torch.Generator().manual_seed(cfg.trainer.seed),
            num_workers=cfg.trainer.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.trainer.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=cfg.trainer.num_workers,
        )

        # 保存先の設定
        experiment_name = f"{cfg.models.model_name}: fold-{fold}"
        save_path = f"{save_dir}/{experiment_name}"
        os.makedirs(save_path, exist_ok=True)

        # モデルの構築
        net = ClassifierModel(
            cfg,
            num_classes,
            cfg.models.model_name,
        )

        # setting callbacks
        ckptCallback = ModelCheckpoint(
            monitor=cfg.trainer.early_stopping_monitor,
            mode=cfg.trainer.early_stopping_mode,
            dirpath=save_path,
            filename="{epoch}--{val_loss:.3f}",
        )
        earlyStoppingCallback = EarlyStopping(
            monitor=cfg.trainer.early_stopping_monitor,
            patience=cfg.trainer.early_stopping_patience,
            mode=cfg.trainer.early_stopping_mode,
            min_delta=cfg.trainer.early_stopping_min_delta,
        )

        # setting logger
        wandb_logger = WandbLogger(
            save_dir=save_path,
            project=cfg.wandb.project,
            group=f"model--{cfg.models.model_name}--dataset--{cfg.dataset.name}",
            job_type=cfg.wandb.job_type,
            tags=[cfg.models.model_name, cfg.dataset.name],
            name=experiment_name,
        )

        csv_logger = CSVLogger(save_dir=save_path, name=experiment_name)

        if cfg.trainer.logger == "wandb":
            logger = wandb_logger
        elif cfg.trainer.logger == "csv":
            logger = csv_logger
        else:
            raise ValueError("logger must be 'wandb' or 'csv'.")
        
        callback_list = [earlyStoppingCallback, RichModelSummary(), RichProgressBar()]
        if cfg.trainer.checkpoint_callback:
            callback_list.append(ckptCallback)

        # trainerの設定
        trainer = pl.Trainer(
            max_epochs=cfg.trainer.max_epochs,
            logger=[logger],
            callbacks=callback_list,
            devices="auto",
            accelerator="gpu",
        )
        # training
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
