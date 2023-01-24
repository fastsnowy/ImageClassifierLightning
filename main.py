import os
from sklearn.model_selection import KFold
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

import wandb
from modules.classifier import mymodel
import hydra
from omegaconf import DictConfig

from datetime import datetime

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "valid": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.models.params.seed)
    kf = KFold(n_splits=5, shuffle=True, random_state=2023)
    full_dataset = ImageFolder(cfg.dataset.path.full)
    train_dataset, test_dataset = udata.random_split(
        full_dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(2023)
    )

    if cfg.trainer.augment:
        print("augment dataset concatting")
        augment_dataset = ImageFolder(cfg.dataset.aug.path)
        train_dataset = udata.ConcatDataset([train_dataset, augment_dataset])

    test_dataset = MySubset(
        test_dataset,
        list(range(0, len(test_dataset))),
        transform=data_transforms["valid"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.models.params.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
    )
    current_time = datetime.now()
    save_dir = f"{cfg.trainer.save_dir}/{current_time:%Y-%m-%d}/{current_time:%H-%M-%S}"
    # K fold cross-validation (K=5)
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):

        experiment_name = f"{cfg.models.params.model_name}: fold-{fold}"
        save_path = f"{save_dir}/{experiment_name}"
        os.makedirs(save_path, exist_ok=True)
        net = mymodel(
            batch_size=cfg.models.params.batch_size,
            num_class=cfg.trainer.num_class,
            m_name=cfg.models.params.model_name,
            optim_name=cfg.models.params.optim_name,
            optim_hparams=cfg.models.optim_params,
        )

        # setting callbacks
        ckptCallback = ModelCheckpoint(
            monitor="loss/val_loss",
            mode="min",
            dirpath=save_path,
            filename="{epoch}--{val_loss:.3f}",
        )
        earlyStoppingCallback = EarlyStopping(monitor="loss/val_loss", patience=5)

        # setting logger
        wandb_logger = WandbLogger(
            save_dir=save_path,
            project=cfg.wandb.project,
            group=f"model--{cfg.models.params.model_name}--dataset--{cfg.dataset.name}",
            job_type=cfg.wandb.job_type,
            tags=[cfg.models.params.model_name, cfg.dataset.name],
            name=experiment_name,
        )
        d_train = MySubset(train_dataset, train_idx, transform=data_transforms["train"])
        d_val = MySubset(train_dataset, val_idx, transform=data_transforms["valid"])
        train_loader = DataLoader(
            d_train,
            cfg.models.params.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
        )
        val_loader = DataLoader(
            d_val,
            cfg.models.params.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
        )

        trainer = pl.Trainer(
            max_epochs=cfg.models.params.epochs,
            logger=[wandb_logger],
            callbacks=[
                earlyStoppingCallback,
                ckptCallback,
                RichModelSummary(),
                RichProgressBar(),
            ],
            devices="auto",
            accelerator="gpu",
        )
        trainer.fit(net, train_loader, val_loader)
        trainer.test(net, test_loader, "best")
        del net
        del trainer
        wandb.finish()
    return None


if __name__ == "__main__":
    print("🐬evaluation start.🐬")
    main()
    print("evaluation finished.😂")
