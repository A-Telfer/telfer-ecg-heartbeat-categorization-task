"""Commands to process raw data
"""
import click
import logging
import pandas as pd
import numpy as np
import random

from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split


def graft_signals(signal1, signal2):
    """Combines two signals at a random point to generate a new signal"""
    split_point = random.randint(0, len(signal1) - 1)
    return np.concatenate([signal1[:split_point], signal2[split_point:]])


def temporal_shift_signal(signal, max=10):
    """Applies a random shift along the x-axis"""
    s = pd.Series(signal)
    shift = random.randint(-max, max)
    s = s.shift(shift).ffill().bfill()
    return s.values


def amplitude_shift_signal(signal, max=0.1):
    """Applies a random shift along the y-axis"""
    signal = signal + random.random() * max * 2 - max
    return np.clip(signal, 0, 1)


def add_noise(signal, scale=0.05):
    """Applies multiplicative noise generated using a normal distribution"""
    noise = np.random.normal(loc=1.0, scale=scale, size=len(signal))
    signal = signal * noise
    return np.clip(signal, 0, 1)


@click.command()
@click.option("--class_sample_size", default=100000, type=click.INT)
@click.option("--test-holdout-ratio", default=0.5, type=click.FLOAT)
@click.option("--seed", default=42, type=click.INT)
def main(class_sample_size, test_holdout_ratio, seed):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # Seed for consistency in creating augmented datasets
    np.random.seed(seed)

    # Load the dataframe
    project_dir = Path(__file__).resolve().parents[2]
    logger.info("loading raw data from mitbih_train.csv")
    df = pd.read_csv(project_dir / "data/raw/mitbih_train.csv", header=None)

    # Rename the target column
    df = df.rename(columns={df.columns[-1]: "target"})
    df.target = df.target.astype(int)

    # For each class, generate new signals using the graft augmentation
    generated_dfs = []
    for c, class_df in df.groupby("target"):
        logger.info(f"generating {class_sample_size} samples for class: {c}")
        indices = np.arange(len(class_df))
        values = class_df.values[:, :-1]
        signals = []
        for _ in tqdm(range(class_sample_size), leave=False):
            # Randomly sample two signals
            i1, i2 = np.random.choice(indices, size=2)
            signal1 = values[i1]
            signal2 = values[i2]

            # Apply augmentations
            signal = graft_signals(signal1, signal2)
            signals.append(signal)

        # Add the class column
        signals = pd.DataFrame(signals)
        signals["target"] = c
        generated_dfs.append(signals)

    # Merge and save the separate class_dataframes
    logger.info("saving processed data to mitbih_train.csv")
    output_df = pd.concat(generated_dfs)
    output_df.to_csv(
        project_dir / "data/processed/mitbih_train.csv", index=False
    )

    # Rename last column as 'target' in the test set
    logger.info("reading raw data from mitbih_test.csv")
    df = pd.read_csv(project_dir / "data/raw/mitbih_test.csv", header=None)
    df = df.rename(columns={df.columns[-1]: "target"})
    df.target = df.target.astype(int)

    # Split test dataset into test and holdout
    test_indices, holdout_indices = train_test_split(
        df.index, test_size=test_holdout_ratio, stratify=df.target
    )

    logger.info("saving test and holdout datasets")
    df.loc[test_indices].to_csv(
        project_dir / "data/processed/mitbih_test.csv", index=False
    )
    df.loc[holdout_indices].to_csv(
        project_dir / "data/processed/mitbih_holdout.csv", index=False
    )
    logger.info("completed data processing")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
