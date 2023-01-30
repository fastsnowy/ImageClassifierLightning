import torch
from torch import Tensor
import torchmetrics
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import timm
import wandb


class mymodel(pl.LightningModule):
    def __init__(
        self,
        batch_size=32,
        num_class=None,
        optim_hparams: dict = None,
        m_name=None,
        optim_name=None,
        pretrained=True,
        cfg=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_class = num_class
        self.m_name = m_name
        self.optim_name = optim_name
        self.pretrained = pretrained
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_class)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_class)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_class)

        if self.m_name == "vgg-16":
            # vgg16
            self.model = timm.create_model(
                "vgg16",
                pretrained=self.pretrained,
                num_classes=self.hparams.num_class,
            )

        elif self.m_name == "resnet-18":
            # resnet18
            self.model = timm.create_model(
                "resnet18",
                pretrained=self.pretrained,
                num_classes=self.hparams.num_class,
            )

        elif self.m_name == "resnet-50":
            self.model = timm.create_model(
                "resnet50",
                pretrained=self.pretrained,
                num_classes=self.hparams.num_class,
            )

        elif self.m_name == "mobilenetv2":
            self.model = timm.create_model(
                "mobilenetv2_100",
                pretrained=self.pretrained,
                num_classes=self.hparams.num_class,
            )
        elif self.m_name == "efficientnet_b0":
            self.model = timm.create_model(
                "efficientnet_b0",
                pretrained=self.pretrained,
                num_classes=self.hparams.num_class,
            )
        elif self.m_name == "densenet121":
            self.model = timm.create_model(
                "densenet121",
                pretrained=self.pretrained,
                num_classes=self.hparams.num_class,
            )
        else:
            assert False, f'不明なmodelです: "{self.m_name}"'

    def cross_entropy_loss(self, y_hat, y):
        loss = F.cross_entropy(y_hat, y)
        return loss

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.formard(x)
        train_loss = self.cross_entropy_loss(y_hat, y)
        self.train_acc(y_hat, y)
        self.log_dict(
            {
                "accuracy/train_acc": self.train_acc,
                "loss/train_loss": train_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = self.cross_entropy_loss(y_hat, y)
        self.val_acc(y_hat, y)
        self.log_dict(
            {
                "loss/val_loss": val_loss,
                "accuracy/val_acc": self.val_acc,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {
            "loss/val_loss": val_loss,
            "accuracy/val_acc": self.val_acc,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss/val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy/val_acc"] for x in outputs]).mean()
        self.log_dict(
            {
                "loss/avg_loss": avg_loss,
                "accuracy/avg_acc": avg_acc,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss/avg_val_loss": avg_loss, "accuracy/val_acc": avg_acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        test_loss = self.cross_entropy_loss(y_hat, y)
        self.test_acc(y_hat, y)
        self.log_dict(
            {
                "loss/test_loss": test_loss,
                "accuracy/test_acc": self.test_acc,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return {
            "loss/test_loss": test_loss,
            "accuracy/test_acc": self.test_acc,
            "raw/preds": y_hat,
            "raw/targets": y,
        }

    def test_epoch_end(self, outputs) -> None:
        preds = torch.cat([x["raw/preds"] for x in outputs])
        targets = torch.cat([x["raw/targets"] for x in outputs])
        avg_loss = torch.stack([x["loss/test_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy/test_acc"] for x in outputs]).mean()

        preds_label = torch.argmax(preds, dim=1)
        print("target", targets.size())
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
            "loss/avg_test_loss": avg_loss,
            "accuracy/test_acc": avg_acc,
            "raw/preds": preds,
            "raw/targets": targets,
        }

    def configure_optimizers(self):
        if self.optim_name == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), **self.hparams.optim_hparams)
        elif self.optim_name == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(), **self.hparams.optim_hparams
            )
        else:
            assert False, f'不明なOptimizerです: "{self.hparams.optim_name}"'
        return optimizer
