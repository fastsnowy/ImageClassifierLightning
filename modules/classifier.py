import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as F
from config import Config
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


class ClassifierModel(pl.LightningModule):
    def __init__(self, cfg, num_classes, model_name="vgg16") -> None:
        super().__init__()
        self.cfg: Config = cfg
        self.save_hyperparameters()
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=cfg.trainer.pretrained,
            num_classes=num_classes,
        )
        metrics = MetricCollection(
            {
                "accuracy": MulticlassAccuracy(num_classes=num_classes),
                "precision": MulticlassPrecision(num_classes=num_classes, average="macro"),
                "recall": MulticlassRecall(num_classes=num_classes, average="macro"),
                "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
            }
        )
        self.train_metrics = metrics.clone(prefix="metrics/train_")
        self.val_metrics = metrics.clone(prefix="metrics/val_")
        self.test_metrics = metrics.clone(prefix="metrics/test_")

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        train_output = self.train_metrics(y_hat, y)
        train_loss = F.cross_entropy(y_hat, y)
        self.log(
            "loss/train_loss",
            train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log_dict(train_output)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = F.cross_entropy(y_hat, y)
        self.val_metrics.update(y_hat, y)
        self.log(
            "loss/val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {
            "raw/targets": y,
            "raw/out": y_hat,
        }

    def on_validation_epoch_end(self):
        val_output = self.val_metrics.compute()
        self.log_dict(val_output)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        test_loss = F.cross_entropy(y_hat, y)
        self.test_metrics.update(y_hat, y)
        self.log("loss/test_loss", test_loss)
        return {
            "raw/preds": y_hat,
            "raw/targets": y,
        }

    def on_test_epoch_end(self):
        test_output = self.test_metrics.compute()
        self.log_dict(test_output)
        self.test_metrics.reset()

    def configure_optimizers(self):
        if self.cfg.trainer.optim_name == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.cfg.trainer.lr,
                momentum=self.cfg.trainer.momentum,
            )
        elif self.cfg.trainer.optim_name == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.trainer.lr)
        else:
            raise NotImplementedError
        return optimizer
