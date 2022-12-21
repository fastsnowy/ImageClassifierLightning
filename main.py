import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import torch
import torchmetrics
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from classifier import mymodel

import hydra
from omegaconf import DictConfig

from datetime import datetime

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]),
    'valid': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]),
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

@hydra.main(config_path='config', config_name='config')
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.models.params.seed)
    sum_matrix = torch.zeros(
        cfg.models.params.num_class,
        cfg.models.params.num_class
        ).cuda()
    kf = KFold(n_splits=5, shuffle=True, random_state=2022)
    cv = 0.0
    cvt = 0.0
    train_set = ImageFolder(cfg.dataset.path.train)

    test_set = ImageFolder(cfg.dataset.path.test, transform=data_transforms['valid'])
    test_loader = DataLoader(test_set, cfg.models.params.batch_size, True, pin_memory=True, num_workers=8)
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_set)):
        net = mymodel(
            batch_size=cfg.models.params.batch_size,
            num_class=cfg.models.params.num_class,
            m_name=cfg.models.params.model_name,
            optim_name=cfg.models.params.optim_name,
            optim_hparams=cfg.models.optim_params
            )
        name =f"{cfg.models.params.model_name}: fold-{fold}"
        save_path = name
        os.makedirs(save_path, exist_ok=True)
        ckpt = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=save_path,
        filename="{epoch}--{val_loss:.3f}",
        )

        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            group=f"model:{cfg.models.params.model_name}, dataset:{cfg.dataset.name}-{datetime.now():%H-%M-%S}",
            job_type=cfg.wandb.job_type,
            tags=[cfg.models.params.model_name, cfg.dataset.name],
            name=name,
            )
        early = EarlyStopping(monitor="val_loss", patience=5)
        d_train = MySubset(train_set, train_idx, data_transforms['train'])
        d_val = MySubset(train_set, val_idx, data_transforms['valid'])
        train_loader = DataLoader(d_train, cfg.models.params.batch_size, shuffle=True, pin_memory=True, num_workers=8)
        val_loader = DataLoader(d_val, cfg.models.params.batch_size, shuffle=False, pin_memory=True, num_workers=8)
        
        trainer = pl.Trainer(
            gpus=1, 
            max_epochs=cfg.models.params.epochs, 
            logger=wandb_logger, 
            callbacks=[early, ckpt]
            )
        trainer.fit(net, train_loader, val_loader)
        cv += trainer.callback_metrics["avg_acc"].mean().item() /kf.n_splits
        trainer.test(net, test_loader, "best")
        # trainer.test(net, ckpt_path="best", test_dataloaders=test_loader)
        test_acc = trainer.callback_metrics["test_acc"].item()
        cvt += test_acc / kf.n_splits
        del net
        # wandb.finish()
        if fold != 4:
            wandb.finish()
        else:
            wandb.log({"CV_test": cvt, "CV": cv})
            wandb.finish()
        print(f"fold:{fold},test_acc:{test_acc}")
    
    return None

if __name__ == '__main__':
    print("🐬evaluation start.🐬")
    main()
    print("evaluation finished.😂")