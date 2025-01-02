# Based on the following preprocessing pipeline code: https://github.com/musikalkemist/generating-sound-with-neural-networks/blob/main/12%20Preprocessing%20pipeline/preprocess.py

"""
1- load a file
2- pad the signal (if necessary)
3- extracting log spectrogram from signal
4- normalise spectrogram
5- save the normalised spectrogram

PreprocessingPipeline
"""

import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, Literal

import librosa
import numpy as np


@dataclass
class Loader:
    """Loader is responsible for loading an audio file."""

    sample_rate: int
    duration_seconds: float
    mono: bool

    def load(self, file_path):
        signal = librosa.load(
            file_path,
            sr=self.sample_rate,
            duration=self.duration_seconds,
            mono=self.mono,
        )[0]
        return signal


@dataclass
class Padder:
    """Padder is responsible to apply padding to an array."""

    mode: Literal["constant"] = "constant"

    def left_pad(self, array, num_missing_items):
        padded_array = np.pad(array, (num_missing_items, 0), mode=self.mode)  # type: ignore
        return padded_array

    def right_pad(self, array, num_missing_items):
        padded_array = np.pad(array, (0, num_missing_items), mode=self.mode)  # type: ignore
        return padded_array


@dataclass
class LogSpectrogramExtractor:
    """LogSpectrogramExtractor extracts log spectrograms (in dB) from a
    time-series signal.
    """

    frame_size: int
    hop_length: int

    def extract(self, signal):
        # This is better than mels for audio reconstruction
        stft = librosa.stft(signal, n_fft=self.frame_size, hop_length=self.hop_length)[
            :-1
        ]
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram


@dataclass
class MinMaxNormaliser:
    """MinMaxNormaliser applies min max normalisation to an array."""

    min: float
    max: float

    def normalise(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array

    def denormalise(self, norm_array, original_min, original_max):
        array = (norm_array - self.min) / (self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return array


@dataclass
class Saver:
    """Saver is responsible to save features, and the min max values."""

    feature_save_dir: str
    min_max_values_save_dir: str

    def save_feature(self, feature, file_path):
        save_path = self._generate_save_path(file_path)
        np.save(save_path, feature)
        return save_path

    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.min_max_values_save_dir, "min_max_values.pkl")
        self._save(min_max_values, save_path)

    @staticmethod
    def _save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
        return save_path


@dataclass
class PreprocessingPipeline:
    """PreprocessingPipeline processes audio files in a directory, applying
    the following steps to each file:
        1- load a file
        2- pad the signal (if necessary)
        3- extracting log spectrogram from signal
        4- normalise spectrogram
        5- save the normalised spectrogram

    Storing the min max values for all the log spectrograms.
    """

    padder: Padder
    extractor: LogSpectrogramExtractor
    normaliser: MinMaxNormaliser
    saver: Saver | None = None
    _loader: Loader | None = field(default=None)
    min_max_values: Dict[str, Dict[str, float]] = field(default_factory=dict)
    _num_expected_samples: int = field(init=False, default=0)

    @property
    def loader(self):
        return self._loader

    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)

    def process_dir(self, audio_files_dir):
        if self.saver is None:
            raise ValueError("Saver is not set")
        for root, _, files in os.walk(audio_files_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self.process_file(file_path)
                print(f"Processed file {file_path}")
        self.saver.save_min_max_values(self.min_max_values)

    def process_file(self, file_path):
        if self.loader is None:
            raise ValueError("Loader is not set")
        if self.saver is None:
            raise ValueError("Saver is not set")
        signal = self.loader.load(file_path)
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        feature = self.extractor.extract(signal)
        norm_feature = self.normaliser.normalise(feature)
        save_path = self.saver.save_feature(norm_feature, file_path)
        self._store_min_max_value(save_path, feature.min(), feature.max())

    def process_audio_bytes(self, signal):
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        feature = self.extractor.extract(signal)
        norm_feature = self.normaliser.normalise(feature)
        return norm_feature

    def _is_padding_necessary(self, signal):
        return len(signal) < self._num_expected_samples

    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal

    def _store_min_max_value(self, save_path, min_val, max_val):
        self.min_max_values[save_path] = {"min": min_val, "max": max_val}


if __name__ == "__main__":
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION_SECONDS = 2
    SAMPLE_RATE = 48000
    MONO = True
    SPLIT = "train"

    SPECTROGRAMS_SAVE_DIR = f"data/{SPLIT}_preprocessed/spectrograms/"
    MIN_MAX_VALUES_SAVE_DIR = f"data/{SPLIT}_preprocessed/"
    FILES_DIR = f"data/{SPLIT}/"

    # instantiate all objects
    loader = Loader(SAMPLE_RATE, DURATION_SECONDS, MONO)
    padder = Padder()
    log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    min_max_normaliser = MinMaxNormaliser(0, 1)
    saver = Saver(SPECTROGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)

    preprocessing_pipeline = PreprocessingPipeline(
        padder, log_spectrogram_extractor, min_max_normaliser, saver, loader
    )

    preprocessing_pipeline.process_dir(FILES_DIR)
