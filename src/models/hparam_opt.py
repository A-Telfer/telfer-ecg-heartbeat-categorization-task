"""Run a simple hyperparameter optimization.

Modified based off the MLFlow Documentation:
https://github.com/mlflow/mlflow/blob/master/examples/hyperparam/search_hyperopt.py
"""
import mlflow
import numpy as np
import click
import logging
from mlflow.tracking import MlflowClient
from hyperopt import fmin, hp, rand

_inf = np.finfo(np.float64).max


@click.command(help="Simple hyperparameter optimization")
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
def hparam_optimize(seed, max_runs, epochs):
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

    def new_eval(
        epochs,
        experiment_id,
        null_train_loss,
        return_all=False,
    ):
        """Function wrapper for training a mdoel"""

        def eval(params):
            lr, momentum, weight_decay = params
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
                        "seed": str(seed),
                    },
                    experiment_id=experiment_id,
                    synchronous=False,
                )
                succeeded = p.wait()
                mlflow.log_params(
                    {
                        "lr": lr,
                        "momentum": momentum,
                        "weight_decay": weight_decay,
                        "epochs": epochs,
                        "seed": seed,
                    }
                )

            if succeeded:
                training_run = tracking_client.get_run(p.run_id)
                metrics = training_run.data.metrics
                train_loss = min(null_train_loss, metrics["train_loss"])
                val_accuracy = metrics["val_accuracy"]
            else:
                tracking_client.set_terminated(p.run_id, "FAILED")
                train_loss = null_train_loss
                val_accuracy = 0

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_accuracy": val_accuracy}
            )

            if return_all:
                return train_loss, val_accuracy
            else:
                return val_accuracy

        return eval

    with mlflow.start_run() as run:
        experiment_id = run.info.experiment_id

        # Create a null run to determine default values when errors occur
        logger.info("Starting null run")
        train_null_loss, _ = new_eval(1, experiment_id, _inf, True)(
            params=[0, 0, 0]
        )

        # Define search space for optimization
        space = [
            hp.loguniform("lr", np.log(1e-4), np.log(1e-1)),
            hp.uniform("momentum", 0, 0.99),
            hp.loguniform("weight_decay", np.log(1e-5), np.log(1e-2)),
        ]

        # Search parameters and optimize loss
        logger.info("Beginning hparam optimization")
        best = fmin(
            fn=new_eval(epochs, experiment_id, train_null_loss),
            space=space,
            algo=rand.suggest,
            max_evals=max_runs,
        )
        mlflow.set_tag("best params", str(best))

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
            if r.data.metrics["val_accuracy"] > best_val_valid:
                best_run = r
                best_val_train = r.data.metrics["train_loss"]
                best_val_valid = r.data.metrics["val_accuracy"]
        mlflow.set_tag("best_run", best_run.info.run_id)
        mlflow.log_metrics(
            {
                "best_train_loss": best_val_train,
                "best_val_accuracy": best_val_valid,
            }
        )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    hparam_optimize()
