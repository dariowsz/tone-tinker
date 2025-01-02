import os
from collections import namedtuple
from pathlib import Path
from typing import Literal, Union

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

PresetSample = namedtuple("PresetSample", ["id", "sample_path", "label"])


class PresetSamplesDataset(Dataset):
    """A dataset loader for generated synth sounds with associated preset metadata."""

    def __init__(
        self,
        data_path: Union[str, Path],
        split: Literal["train", "val"] = "train",
        preprocessed: bool = False,
        transforms=None,
    ):
        if split in ["train", "val"]:
            self.split = split
        else:
            print(f"'{split}' is not a valid split. Defaulting to 'train'.")
            self.split = "train"

        self.data_path = Path(data_path)
        self.preprocessed = preprocessed
        self.transforms = transforms
        self.samples = self._load_list_of_samples(preprocessed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        _, sample_path, label = self.samples[idx]
        if self.preprocessed:
            waveform = torch.tensor(
                np.load(sample_path), dtype=torch.float32
            ).unsqueeze(0)
            sample_rate = -1
        else:
            waveform, sample_rate = torchaudio.load(sample_path)
        label = torch.tensor(label, dtype=torch.float32)
        # First two values are between 0-100, normalize to 0-1
        label[0:2] = label[0:2] / 100.0
        if self.transforms:
            waveform = self.transforms(waveform)

        return waveform, sample_path, label

    def _load_list_of_samples(self, preprocessed: bool) -> list[PresetSample]:
        data_df = pd.read_csv(self.data_path / f"{self.split}.csv")
        dummies = pd.get_dummies(data_df["osc1_wavetable"], dtype=int)
        data_df = pd.concat([data_df, dummies], axis="columns")
        data_df.drop(["osc1_wavetable"], axis="columns", inplace=True)
        data_df["samples"] = data_df.apply(
            lambda row: PresetSample(
                row["audio_path"],
                (
                    os.path.join(self.data_path, self.split, row["audio_path"])
                    if not preprocessed
                    else os.path.join(
                        self.data_path,
                        f"{self.split}_preprocessed",
                        "spectrograms",
                        f"{row['audio_path']}.npy",
                    )
                ),
                row.drop("audio_path").values.astype(int),
            ),
            axis=1,
        )
        return data_df["samples"].tolist()
