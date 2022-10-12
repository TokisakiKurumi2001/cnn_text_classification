from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from cnn_text_classification import LitCNNForSeqClassifier, train_dataloader, valid_dataloader, test_dataloader

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="proj_cnn_cls")

    # model
    lit_cnn_cls = LitCNNForSeqClassifier(
        num_classes=5, embed_dim=512, kernel_sizes=[3, 4, 5],
        num_channels=[512, 512, 512], dropout=0.1
    )

    # train model
    trainer = pl.Trainer(
        max_epochs=20, logger=wandb_logger, devices=2, accelerator="gpu", strategy="ddp",
        callbacks=[EarlyStopping(monitor="valid/acc_epoch", min_delta=0.00, patience=5, verbose=False, mode="max")]
    )
    trainer.fit(model=lit_cnn_cls, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    trainer.test(model=lit_cnn_cls, dataloaders=test_dataloader)
