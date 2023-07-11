import torch
from torch import Tensor
import torchmetrics
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import timm
import wandb


class ClassifierModel(pl.LightningModule):
    def __init__(self, cfg, num_classes, model_name="vgg16") -> None:
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.save_hyperparameters()
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=False,
            num_classes=num_classes,
        )
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        train_loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat, y)
        self.log(
            "loss/train_loss",
            train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "accuracy/train_acc",
            self.train_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = F.cross_entropy(y_hat, y)
        self.val_acc(y_hat, y)
        self.log(
            "loss/val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "accuracy/val_acc",
            self.val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {
            "loss/val_loss": val_loss,
            "accuracy/val_acc": self.val_acc,
            "raw/out": y_hat,
            "raw/targets": y,
        }

    def validation_epoch_end(self, outputs) -> None:
        return None

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        test_loss = F.cross_entropy(y_hat, y)
        self.test_acc(y_hat, y)
        self.log_dict({"loss/test_loss": test_loss, "accuracy/test_acc": self.test_acc})
        return {
            "raw/preds": y_hat,
            "raw/targets": y,
        }

    def test_epoch_end(self, outputs) -> None:
        preds = torch.cat([x["raw/preds"] for x in outputs])
        targets = torch.cat([x["raw/targets"] for x in outputs])

        preds_label = torch.argmax(preds, dim=1)
        self.logger.experiment.log(
            {
                "metrics/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=targets.cpu().numpy(),
                    preds=preds_label.cpu().numpy(),
                ),
            }
        )

        return {
            "raw/preds": preds,
            "raw/targets": targets,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        # optimizer = torch.optim.Adam(self.parameters(), lr=0.02)
        return optimizer
