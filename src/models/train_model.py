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
import os
from pathlib import Path
from torch.utils.data import DataLoader

import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from src.data.dataset import EcgDataset


class MetricsCallback(Callback):
    """Callback created to log mlflow metrics and to manage early stopping"""

    def __init__(self, mlflow_run, num_classes=5, early_stopping_patience=3):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Define metrics
        self.accuracy_individual = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, average="none"
        )
        self.accuracy_macro = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.f1_macro = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.auroc_macro = torchmetrics.AUROC(
            task="multiclass", num_classes=num_classes, average="macro"
        )

        # Early stopping
        self.stop_on_next_train_epoch_end = False
        self.early_stopping_last_value = None
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_count = early_stopping_patience
        self.early_stopping_checkpoint = f"{mlflow_run.info.run_id}_best.pt"

    def on_train_start(self, trainer, pl_module):
        self.early_stopping_metric_history = []

    def log_per_class_accuracies(self, preds, targets, prefix, epoch=None):
        score = self.accuracy_individual(preds, targets)
        score = {
            f"{prefix}_accuracy_class{i}": a
            for i, a in enumerate(score.numpy())
        }
        mlflow.log_metrics(score, step=epoch)
        return score

    def log_macro_accuracy(self, preds, targets, prefix, epoch=None):
        score = self.accuracy_macro(preds, targets).cpu().item()
        mlflow.log_metric(f"{prefix}_accuracy", score, step=epoch)
        return score

    def log_macro_f1(self, preds, targets, prefix, epoch=None):
        score = self.f1_macro(preds, targets).cpu().item()
        mlflow.log_metric(f"{prefix}_f1", score, step=epoch)
        return score

    def log_macro_auroc(self, preds, targets, prefix, epoch=None):
        score = self.auroc_macro(preds, targets).cpu().item()
        mlflow.log_metric(f"{prefix}_auroc", score, step=epoch)
        return score

    def on_train_epoch_end(self, trainer, pl_module):
        preds = torch.concat(pl_module.epoch_train_preds).cpu()
        targets = torch.concat(pl_module.epoch_train_targets).cpu()

        # Report training loss
        mean_loss = torch.concat(pl_module.epoch_train_preds).mean()
        mean_loss = mean_loss.cpu().item()
        mlflow.log_metric("train_loss", mean_loss, step=trainer.current_epoch)

        # Report metrics
        accuracy = self.log_macro_accuracy(
            preds, targets, "train", trainer.current_epoch
        )
        f1 = self.log_macro_f1(preds, targets, "train", trainer.current_epoch)
        auroc = self.log_macro_auroc(
            preds, targets, "train", trainer.current_epoch
        )

        self.logger.info(
            f"TRAIN epoch: {trainer.current_epoch}, "
            f"acc: {accuracy:.3f}, "
            f"auroc: {auroc:.3f}, "
            f"f1: {f1:.3f}, "
            f"loss: {mean_loss:.3f}"
        )

        # Reset for next epoch
        pl_module.epoch_train_preds = []
        pl_module.epoch_train_targets = []

        # Early stopping
        if self.stop_on_next_train_epoch_end:
            trainer.should_stop = True
            pl_module._model = torch.load(self.early_stopping_checkpoint)
            os.remove(self.early_stopping_checkpoint)
            self.logger.info(
                "early stopping triggered, returning best checkpoint"
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        preds = torch.concat(pl_module.epoch_validation_preds).cpu()
        targets = torch.concat(pl_module.epoch_validation_targets).cpu()

        # Metrics
        accuracy = self.log_macro_accuracy(
            preds, targets, "val", trainer.current_epoch
        )
        f1 = self.log_macro_f1(preds, targets, "val", trainer.current_epoch)
        auroc = self.log_macro_auroc(
            preds, targets, "val", trainer.current_epoch
        )

        # Early stopping
        if (
            self.early_stopping_last_value is None
            or auroc > self.early_stopping_last_value
        ):
            self.early_stopping_last_value = auroc
            self.early_stopping_count = self.early_stopping_patience

            # Save best checkpoint
            torch.save(pl_module._model, self.early_stopping_checkpoint)
        else:
            self.early_stopping_count -= 1

        if self.early_stopping_count == 0:
            self.stop_on_next_train_epoch_end = True

        # Update metrics
        self.logger.info(
            f"VAL epoch: {trainer.current_epoch}, "
            f"acc: {accuracy:.3f}, "
            f"auroc: {auroc:.3f}, "
            f"f1: {f1:.3f}"
        )

        pl_module.epoch_validation_preds = []
        pl_module.epoch_validation_targets = []

    def on_test_epoch_end(self, trainer, pl_module):
        preds = torch.concat(pl_module.epoch_test_preds).cpu()
        targets = torch.concat(pl_module.epoch_test_targets).cpu()

        class_accuracies = self.log_per_class_accuracies(
            preds, targets, "test"
        )
        accuracy = self.log_macro_accuracy(preds, targets, "test")
        f1 = self.log_macro_f1(preds, targets, "test")
        auroc = self.log_macro_auroc(preds, targets, "test")

        self.logger.info(
            f"TEST acc: {accuracy:.3f}, "
            + f"auroc: {auroc:.3f}, "
            + (
                ", ".join(
                    [f"{k}: {v:.3f}" for k, v in class_accuracies.items()]
                )
            )
            + f", f1: {f1:.3f}"
        )

        pl_module.epoch_test_preds = []
        pl_module.epoch_test_targets = []


class LinearModel(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-3,
        momentum=0.9,
        weight_decay=1e-3,
        input_size=187,
        output_size=5,
        num_hidden_layers=1,
        hidden_layer_size=2048,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        hidden_layers = []
        for _ in range(num_hidden_layers):
            hidden_layers.append(
                torch.nn.Linear(hidden_layer_size, hidden_layer_size)
            )
            hidden_layers.append(torch.nn.LeakyReLU())
            torch.nn.Dropout(),

        self._model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_layer_size),
            torch.nn.LeakyReLU(),
            *hidden_layers,
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_layer_size, output_size),
        )

        self.epoch_train_losses = []
        self.epoch_train_preds = []
        self.epoch_train_targets = []
        self.epoch_validation_preds = []
        self.epoch_validation_targets = []
        self.epoch_test_preds = []
        self.epoch_test_targets = []

    def forward(self, x):
        return self._model(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.forward(x)
        loss = F.cross_entropy(x_hat, y)
        self.epoch_train_preds.append(x_hat.detach())
        self.epoch_train_targets.append(y.detach())
        self.epoch_train_losses.append(loss.detach())
        return loss

    def eval_step(self, batch, batch_idx, mode):
        x, y = batch
        x_hat = self.forward(x)
        if mode == "val":
            self.epoch_validation_preds.append(x_hat.detach())
            self.epoch_validation_targets.append(y.detach())
        elif mode == "test":
            self.epoch_test_preds.append(x_hat.detach())
            self.epoch_test_targets.append(y.detach())

    def validation_step(self, batch, batch_idx):
        self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self.eval_step(batch, batch_idx, "test")

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
    "--hidden_layers",
    type=click.INT,
    default=1,
    help="Number of hidden layers in the model",
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
    hidden_layers,
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

    # Disable warning when using RTX4090
    torch.set_float32_matmul_precision("high")

    # Seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    with mlflow.start_run() as active_run:
        # Save parameters
        # mlflow.log_param('learning_rate', learning_rate)
        # mlflow.log_param('momentum', momentum)
        # mlflow.log_param('weight_decay', weight_decay)

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

        model = LinearModel(
            input_size=train_dataset[0][0].shape[0],
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            num_hidden_layers=hidden_layers,
        )

        warnings.filterwarnings("ignore")
        trainer = pl.Trainer(
            max_epochs=epochs,
            enable_progress_bar=False,
            callbacks=[MetricsCallback(active_run, early_stopping_patience=5)],
            enable_checkpointing=False,
            logger=False,
        )
        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=[validation_dataloader],
        )

        trainer.test(model=model, dataloaders=test_dataloader)
        preds = trainer.predict(
            model, test_dataloader, return_predictions=True
        )
        preds = torch.concat(preds).numpy()

        signature = mlflow.models.signature.infer_signature(
            test_dataset.df, preds
        )
        mlflow.pytorch.log_model(model, "linear_model", signature=signature)
        logger.info("run complete")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    run()
