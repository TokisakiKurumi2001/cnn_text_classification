import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List
from cnn_text_classification import CNNForSeqClassifier
import torchmetrics

class LitCNNForSeqClassifier(pl.LightningModule):
    def __init__(
        self, num_classes: int, embed_dim: int, kernel_sizes: List[int],
        num_channels: List[int], dropout: float
    ):
        super(LitCNNForSeqClassifier, self).__init__()
        self.cnn_cls = CNNForSeqClassifier(
            num_classes, embed_dim, kernel_sizes, num_channels, dropout
        )
        self.num_classes = num_classes
        self.main_loss = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        logits = self.cnn_cls(batch)
        loss = self.main_loss(logits.view(-1, self.num_classes), labels.view(-1))
        self.log("train/loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        preds = torch.argmax(logits, dim=-1)
        self.train_acc.update(preds, labels)
        return loss

    def training_epoch_end(self, outputs):
        self.log('train/acc_epoch', self.train_acc.compute(), on_epoch=True, sync_dist=True)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        logits = self.cnn_cls(batch)
        loss = self.main_loss(logits.view(-1, self.num_classes), labels.view(-1))
        self.log("valid/loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        preds = torch.argmax(logits, dim=-1)
        self.valid_acc.update(preds, labels)
        return loss

    def validation_step_end(self, outputs):
        self.log('valid/acc_epoch', self.valid_acc.compute(), on_epoch=True, sync_dist=True)
        self.valid_acc.reset()

    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        logits = self.cnn_cls(batch)
        preds = torch.argmax(logits, dim=-1)
        self.test_acc.update(preds, labels)
        return None
        
    def test_step_end(self, outputs):
        self.log('test/acc_epoch', self.test_acc.compute(), on_epoch=True, sync_dist=True)
        self.test_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        return optimizer