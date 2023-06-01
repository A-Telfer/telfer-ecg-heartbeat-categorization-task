import numpy as np

from torch.utils.data import Dataset


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
