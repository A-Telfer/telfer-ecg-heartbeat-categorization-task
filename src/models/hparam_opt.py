"""Run a simple hyperparameter optimization.

Modified based off the MLFlow Documentation:
https://github.com/mlflow/mlflow/blob/master/examples/hyperparam/search_hyperopt.py
"""
import mlflow
import numpy as np
import click
import logging
import optuna
import pandas as pd
import torch
import torchmetrics
from torch.utils.data import DataLoader
from src.data.dataset import EcgDataset
from mlflow.tracking import MlflowClient
from pathlib import Path

_inf = np.finfo(np.float64).max


@click.command(help="Simple hyperparameter optimization")
@click.option("--training_data", default="data/processed", type=click.Path())
@click.option(
    "--seed",
    type=click.INT,
    default=97531,
    help="Seed for the random generator.",
)
@click.option(
    "--max_runs",
    type=click.INT,
    default=3,
    help="Maximum number of runs to perform hparam optimization with.",
)
@click.option(
    "--epochs",
    type=click.INT,
    default=3,
    help="How many epochs to train each run for.",
)
def hparam_optimize(training_data, seed, max_runs, epochs):
    """Perform a simple hyper parameter optimization

    Parameters
    ----------
    seed : int
      Seed random generators for reproducibility.
    max_runs : int
      The number of runs used to search the parameter space.
    epochs : int
      How many epochs in each run.
    """
    logger = logging.getLogger(__name__)

    np.random.seed(seed)
    tracking_client = MlflowClient()

    def new_eval(epochs, experiment_id):
        """Function wrapper for training a mdoel"""

        def eval(trial):
            lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
            weight_decay = trial.suggest_float(
                "weight_decay", 1e-5, 1e-2, log=True
            )
            momentum = trial.suggest_float("momentum", 0, 0.99)
            hidden_layers = trial.suggest_int("num_layers", 1, 4)

            with mlflow.start_run(nested=True) as child_run:
                p = mlflow.projects.run(
                    run_id=child_run.info.run_id,
                    uri=".",
                    entry_point="train",
                    parameters={
                        "epochs": str(epochs),
                        "learning_rate": str(lr),
                        "momentum": str(momentum),
                        "weight_decay": str(weight_decay),
                        "hidden_layers": str(hidden_layers),
                        "seed": str(seed),
                    },
                    experiment_id=experiment_id,
                    synchronous=False,
                )
                succeeded = p.wait()
                mlflow.log_params(
                    {
                        "epochs": epochs,
                        "learning_rate": lr,
                        "momentum": momentum,
                        "weight_decay": weight_decay,
                        "hidden_layers": hidden_layers,
                        "seed": seed,
                    }
                )

            if succeeded:
                training_run = tracking_client.get_run(p.run_id)
                metrics = training_run.data.metrics
                score = metrics["val_auroc"]
            else:
                tracking_client.set_terminated(p.run_id, "FAILED")
                score = 0

            return score

        return eval

    with mlflow.start_run() as run:
        experiment_id = run.info.experiment_id
        logger.info("Starting null run")

        # Define search space for optimization and optimize
        logger.info("Beginning hparam optimization")
        study = optuna.create_study(direction="maximize")
        study.optimize(new_eval(epochs, experiment_id), n_trials=max_runs)
        mlflow.set_tags(study.best_params)

        # Find and report the best run
        client = MlflowClient()
        runs = client.search_runs(
            [experiment_id],
            "tags.mlflow.parentRunId = '{run_id}' ".format(
                run_id=run.info.run_id
            ),
        )
        best_val_train = 0
        best_val_valid = 0
        best_run = None
        for r in runs:
            if r.data.metrics["test_auroc"] > best_val_valid:
                best_run = r
                best_val_train = r.data.metrics["train_loss"]
                best_val_valid = r.data.metrics["test_auroc"]
        mlflow.set_tag("best_run", best_run.info.run_id)
        mlflow.log_metrics(
            {
                "best_train_loss": best_val_train,
                "best_test_auroc": best_val_valid,
            }
        )

        # Run the holdout set on the best model
        model_uri = "runs:/{}/linear_model".format(best_run.info.run_id)
        model = mlflow.pytorch.load_model(model_uri)

        # Load the holdout dataset
        holdout_datafile = Path(training_data) / "mitbih_holdout.csv"
        holdout_df = pd.read_csv(holdout_datafile)
        holdout_dataset = EcgDataset(holdout_df)
        holdout_dataloader = DataLoader(
            holdout_dataset,
            batch_size=64,
        )

        # Make predictions
        x_hat = []
        target = []
        for batch in holdout_dataloader:
            x, y = batch
            x_hat.append(model(x))
            target.append(y)

        x_hat = torch.concat(x_hat)
        target = torch.concat(target)

        # Get metrics
        auroc_score = torchmetrics.AUROC(
            task="multiclass", num_classes=x_hat.shape[1], average="macro"
        )(x_hat, target)

        accuracy_score = torchmetrics.Accuracy(
            task="multiclass", num_classes=x_hat.shape[1], average="macro"
        )(x_hat, target)

        # Save to mlflow
        mlflow.set_tag("holdout_accuracy", accuracy_score)
        mlflow.set_tag("holdout_auroc", auroc_score)
        logger.info(
            f"Holdout set results, acc: {accuracy_score:.3f}, "
            f"auroc: {auroc_score:.3f}"
        )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    hparam_optimize()
