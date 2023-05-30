"""Trains a simple pytorch linear model
"""
import warnings
import mlflow
import numpy as np
import pandas as pd
import torch
import click
import logging
import lightning.pytorch as pl
import mlflow.pytorch
import torchmetrics

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# from tqdm import tqdm
import torch.nn.functional as F

# from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import Callback


class ComputeMetrics(Callback):
    def __init__(self, mlflow_run, num_classes=5):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.accuracy_individual = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, average="none"
        )
        self.accuracy_macro = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.f1_macro = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.auroc = torchmetrics.AUROC(
            task="multiclass", num_classes=num_classes, average="none"
        )
        self.auroc_macro = torchmetrics.AUROC(
            task="multiclass", num_classes=num_classes, average="macro"
        )

    def on_train_epoch_end(self, trainer, pl_module):
        preds = torch.concat(pl_module.epoch_train_preds).cpu()
        targets = torch.concat(pl_module.epoch_train_targets).cpu()

        mean_loss = torch.concat(pl_module.epoch_train_preds).mean()
        mean_loss = mean_loss.cpu().item()
        mlflow.log_metric("train_loss", mean_loss, step=trainer.current_epoch)

        accuracy = self.accuracy_individual(preds, targets)
        accuracy = {
            f"val_class{i}_acc": a for i, a in enumerate(accuracy.numpy())
        }
        mlflow.log_metrics(accuracy, step=trainer.current_epoch)

        accuracy = self.accuracy_macro(preds, targets).cpu().item()
        mlflow.log_metric(
            "train_accuracy", accuracy, step=trainer.current_epoch
        )
        self.logger.info(f"Train loss: {mean_loss:.3f}, Accuracy: {accuracy}")

        # Reset for next epoch
        pl_module.epoch_train_preds = []
        pl_module.epoch_train_targets = []

    def on_validation_epoch_end(self, trainer, pl_module):
        preds = torch.concat(pl_module.epoch_validation_preds).cpu()
        targets = torch.concat(pl_module.epoch_validation_targets).cpu()

        accuracy = self.accuracy_macro(preds, targets).cpu().item()
        mlflow.log_metric("val_accuracy", accuracy, step=trainer.current_epoch)
        self.logger.info(f"Validation accuracy: {accuracy}")

        pl_module.epoch_validation_preds = []
        pl_module.epoch_validation_targets = []

    def on_test_epoch_end(self, trainer, pl_module):
        preds = torch.concat(pl_module.epoch_test_preds).cpu()
        targets = torch.concat(pl_module.epoch_test_targets).cpu()

        accuracy = self.accuracy_macro(preds, targets).cpu().item()
        mlflow.log_metric(
            "test_accuracy", accuracy, step=trainer.current_epoch
        )
        self.logger.info(f"Test accuracy: {accuracy}")

        pl_module.epoch_test_preds = []
        pl_module.epoch_test_targets = []


class EcgDataset(Dataset):
    """Ecg Dataset with no runtime augmentation"""

    def __init__(self, df):
        self.df = df
        self.targets = df.target.astype(int)
        self.inputs = df[df.columns.drop("target")]

    def __getitem__(self, idx):
        x = self.inputs.iloc[idx].values.astype(np.float32)
        y = self.targets.iloc[idx].astype(int)
        return x, y

    def __len__(self):
        return len(self.df)


class LitAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        input_size=187,
        learning_rate=1e-3,
        momentum=0.9,
        weight_decay=1e-3,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        self._model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 5),
        )

        # Use
        self.epoch_train_losses = []
        self.epoch_train_preds = []
        self.epoch_train_targets = []
        self.epoch_validation_preds = []
        self.epoch_validation_targets = []
        self.epoch_test_preds = []
        self.epoch_test_targets = []

    def forward(self, x):
        return self._model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.forward(x)
        loss = F.cross_entropy(x_hat, y)
        self.epoch_train_preds.append(x_hat.detach())
        self.epoch_train_targets.append(y.detach())
        self.epoch_train_losses.append(loss.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.forward(x)
        self.epoch_validation_preds.append(x_hat.detach())
        self.epoch_validation_targets.append(y.detach())

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.forward(x)
        self.epoch_test_preds.append(x_hat.detach())
        self.epoch_test_targets.append(y.detach())

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        return optimizer


@click.command(help="Train a model on the ECG Dataset")
@click.option("--training_data", default="data/processed", type=click.Path())
@click.option(
    "--epochs",
    type=click.INT,
    default=10,
    help="Maximum number of epochs to evaluate.",
)
@click.option(
    "--batch-size",
    type=click.INT,
    default=64,
    help="Batch size used to train with",
)
@click.option(
    "--learning_rate", type=click.FLOAT, default=1e-2, help="Learning rate."
)
@click.option(
    "--momentum", type=click.FLOAT, default=0.9, help="SGD momentum."
)
@click.option(
    "--weight_decay",
    type=click.FLOAT,
    default=1e-4,
    help="L2 weight normalization",
)
@click.option(
    "--seed",
    type=click.INT,
    default=97531,
    help="Seed for the random generator.",
)
def run(
    training_data,
    epochs,
    batch_size,
    learning_rate,
    momentum,
    weight_decay,
    seed,
):
    """Train a small Linear model

    Parameters
    ----------
    training_data : str
    epochs : int
    batch_size : int
    learning_rate : float
    momentum : float
    seed : int
    """
    logger = logging.getLogger(__name__)

    # Seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    # Load the training data
    # NOTE: training and validation sets are not completely separate as
    # they were generated from the same sample (data augmentation
    # occurred earlier)
    train_datafile = Path(training_data) / "mitbih_train.csv"
    train_df = pd.read_csv(train_datafile)

    val_datafile = Path(training_data) / "mitbih_val.csv"
    val_df = pd.read_csv(val_datafile)

    test_datafile = Path(training_data) / "mitbih_test.csv"
    test_df = pd.read_csv(test_datafile)

    # Create the training data
    train_dataset = EcgDataset(train_df)
    # train_class_weights =
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=g
    )
    validation_dataset = EcgDataset(val_df)
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        generator=g,
    )
    test_dataset = EcgDataset(test_df)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        generator=g,
    )

    model = LitAutoEncoder(
        input_size=train_dataset[0][0].shape[0],
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
    )

    with mlflow.start_run() as active_run:
        warnings.filterwarnings("ignore")
        trainer = pl.Trainer(
            max_epochs=epochs,
            enable_progress_bar=False,
            callbacks=[
                # EarlyStopping('val_loss'),
                ComputeMetrics(active_run)
            ],
        )
        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=[validation_dataloader],
        )
        trainer.test(model=model, dataloaders=test_dataloader)
        logger.info("run complete")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    run()
