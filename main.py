import os
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
import torch
import torchmetrics
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader
from torchvision import transforms, utils, models
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import wandb
import hydra
from omegaconf import DictConfig
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        # transforms.CenterCrop(224),
        # transforms.RandomHorizontalFlip(p=0.1),
        # transforms.RandomVerticalFlip(p=0.1),
        # transforms.RandomAutocontrast(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'valid': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

class mymodel(pl.LightningModule):

    def __init__(self,
    batch_size, 
    num_class, 
    optim_hparams: dict,
    model_name,
    optim_name,
    d_train=None,
    d_val=None,
    d_test=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.num_class = num_class
        self.model_name = model_name
        self.optim_name = optim_name
        self.conf_matrix = torchmetrics.ConfusionMatrix(num_class)
        # self.d_train = d_train
        # self.d_val = d_val
        # self.d_test = d_test

        if self.model_name == "vgg16":
            #vgg16
            self.model = models.vgg16(pretrained=True)
            for param in self.model.parameters():
                param.requires_grad=False
            num_feat = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_feat, self.hparams.num_class)
            print(self.model)
        
        elif self.model_name == 'resnet18':
            # resnet18
            self.model = models.resnet18(pretrained=True)
            for param in self.model.parameters():
                param.requires_grad=False
            num_feat = self.model.fc.in_features
            self.model.fc = nn.Linear(num_feat, self.hparams.num_class)
        
        else:
            assert False, f'不明なmodelです: "{self.model_name}"'

        # loss
        self.loss_module = nn.CrossEntropyLoss()
    
    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        val_loss = F.cross_entropy(out, y)
        out_label = torch.argmax(out, dim=1)
        acc = torch.sum(y==out_label) *1.0 /len(y)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': val_loss, 'val_acc': acc, 'out': out, 'targets': y}
    

    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        self.log('avg_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('avg_acc', avg_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'avg_val_loss': avg_loss, 'val_acc': avg_acc}
    
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        test_loss = F.cross_entropy(out, y)
        out_label = torch.argmax(out, dim=1)
        acc = torch.sum(y==out_label) *1.0 /len(y)
        self.log('test_loss', test_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'test_loss': test_loss, 'test_acc': acc, 'preds':out, 'targets': y}
    
    def test_epoch_end(self, outputs) -> None:
        global sum_matrix
        preds = torch.cat([x["preds"] for x in outputs])
        targets = torch.cat([x["targets"] for x in outputs])
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        # mat = self.conf_matrix(preds, targets)
        # sum_matrix += mat
        # names = ["Bacterialblight", "Blast", "Brownspot", "Tungro"]
        # df = pd.DataFrame(mat.cpu().numpy(), index=names, columns=names)
        # plt.figure(figsize = (10,7))
        # sns.set(font_scale=1.3)
        # # plt.xlabel('Predicted label')
        # # plt.ylabel('True label')
        # fig_ = sns.heatmap(df, annot=True, cmap='Blues', fmt="g")
        # fig_.set(xlabel='Predicited label', ylabel='True label')
        # fig_.get_figure()
        # # plt.close(fig_)
        # wandb.log({"confusion matrix(testset)": [wandb.Image(fig_)]})
        # df_sum = pd.DataFrame(sum_matrix.cpu().numpy(), index=names, columns=names)
        # plt.figure(figsize = (10,7))
        # sns.set(font_scale=1.3)
        # # plt.xlabel('Predicted label')
        # # plt.ylabel('True label')
        # fig_s = sns.heatmap(df_sum, annot=True, cmap='Blues', fmt="g")
        # fig_s.set(xlabel='Predicited label', ylabel='True label')
        # fig_s.get_figure()
        # # plt.close(fig_s)
        # wandb.log({"confusion matrix(sum)": [wandb.Image(fig_s)]})

        return {'avg_test_loss': avg_loss, 'test_acc': avg_acc, 'preds': preds, 'targets': targets}

    
    def configure_optimizers(self):
        if self.optim_name == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), **self.hparams.optim_hparams)
        elif self.optim_name == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), **self.hparams.optim_hparams)
        else:
            assert False, f'不明なOptimizerです: "{self.hparams.optim_name}"'
        return optimizer
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

@hydra.main("config", "config")
def main(cfg: DictConfig) -> None:
    pass
if __name__ == '__main__':
    main()