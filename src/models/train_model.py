"""Trains a simple pytorch linear model
"""
import warnings
import mlflow
import numpy as np
import pandas as pd
import torch
import click
import logging

import mlflow.pytorch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class SimpleLinearModel(torch.nn.Module):
    def __init__(self, input_size=187):
        """Creates a simple model FC model"""
        super().__init__()

        self._model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 5),
        )

    def forward(self, x):
        return self._model(x)


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


@click.command(help="Train a model on the ECG Dataset")
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
    "--seed",
    type=click.INT,
    default=97531,
    help="Seed for the random generator.",
)
@click.option(
    "--train-val-ratio",
    type=click.FLOAT,
    default=0.95,
    help="Ratio of training data of the whole dataset.",
)
@click.argument("training_data", type=click.Path())
def run(
    training_data,
    epochs,
    batch_size,
    learning_rate,
    momentum,
    seed,
    train_val_ratio,
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
    train_val_ratio : float
    """
    logger = logging.getLogger(__name__)
    with mlflow.start_run():
        warnings.filterwarnings("ignore")

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

        # Initialize model and optimizer
        model = SimpleLinearModel(input_size=train_dataset[0][0].shape[0])
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=momentum
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            # Training loop
            train_losses = []
            train_preds = None
            train_targets = None
            train_loop = tqdm(
                enumerate(train_dataloader),
                total=len(train_dataloader),
                leave=False,
            )
            for batch_idx, batch in train_loop:
                x, target = batch
                pred = model(x)
                loss = loss_fn(pred, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_losses.append(loss.item())
                train_targets = (
                    target
                    if train_targets is None
                    else torch.concat([train_targets, target])
                )
                train_preds = (
                    pred
                    if train_preds is None
                    else torch.concat([train_preds, pred])
                )
                train_loop.set_description(
                    f"Loss: {np.array(train_losses).mean():.04f}"
                )

            # Validation loop
            val_losses = []
            val_preds = None
            val_targets = None
            val_loop = tqdm(
                enumerate(validation_dataloader),
                total=len(validation_dataloader),
                leave=False,
            )
            for batch_idx, batch in val_loop:
                with torch.no_grad():
                    x, target = batch
                    pred = model(x)
                    loss = loss_fn(pred, target)

                val_losses.append(loss.item())
                val_targets = (
                    target
                    if val_targets is None
                    else torch.concat([val_targets, target])
                )
                val_preds = (
                    pred
                    if val_preds is None
                    else torch.concat([val_preds, pred])
                )

            val_loss = np.array(val_losses).mean()
            train_loss = np.array(train_losses).mean()
            mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)
            mlflow.log_metric(key="val_loss", value=val_loss, step=epoch)
            logger.info(
                f"Epoch: {epoch}, "
                f"Training loss: {train_loss:.3f}, "
                f"Validation Loss: {val_loss:.3f}"
            )

        # Testing loop
        test_losses = []
        test_preds = None
        test_targets = None
        test_loop = tqdm(
            enumerate(test_dataloader),
            total=len(test_dataloader),
            leave=False,
        )
        for batch_idx, batch in test_loop:
            with torch.no_grad():
                x, target = batch
                pred = model(x)
                loss = loss_fn(pred, target)

            test_losses.append(loss.item())
            test_targets = (
                target
                if test_targets is None
                else torch.concat([test_targets, target])
            )
            test_preds = (
                pred
                if test_preds is None
                else torch.concat([test_preds, pred])
            )

        test_loss = np.array(test_losses).mean()
        logger.info(f"Test loss: {test_loss:.3f}")
        mlflow.log_metric(key="test_loss", value=test_loss)
        logger.info("run complete")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    run()
