import os
from typing import Tuple

import numpy as np
import torch

from src.ml_models import Autoencoder, SoundDesigner
from src.preprocess import (
    Loader,
    LogSpectrogramExtractor,
    MinMaxNormaliser,
    Padder,
    PreprocessingPipeline,
    Saver,
)
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.common import get_device

IDX_TO_OSC1_WAVETABLE = {
    0: "A.I.",
    1: "Esca II",
    2: "Plysaw II",
    3: "Sin-Tri",
    4: "Squ-Sw I",
}


class SoundDesignerInference:
    def __init__(self, model_path: str, config: dict):
        self.device = get_device()
        self.model = self._load_model(model_path, config)
        self.model.to(self.device)
        self.model.eval()
        self.config = config

    def _load_model(self, weights_path: str, config: dict) -> SoundDesigner:
        autoencoder = Autoencoder(
            input_shape=config["autoencoder"]["input_shape"],
            conv_filters=config["autoencoder"]["conv_filters"],
            conv_kernels=config["autoencoder"]["conv_kernels"],
            conv_strides=config["autoencoder"]["conv_strides"],
            decoder_output_padding=config["autoencoder"]["decoder_output_padding"],
            latent_space_dim=config["autoencoder"]["latent_space_dim"],
        )

        autoencoder, _ = CheckpointManager.load_checkpoint(
            config["autoencoder"]["pretrained_weights_path"], autoencoder
        )

        model = SoundDesigner(encoder=autoencoder.encoder)
        model, _ = CheckpointManager.load_checkpoint(weights_path, model)
        return model

    def _preprocess_audio(self, audio_data: np.ndarray) -> torch.Tensor:
        padder = Padder()
        log_spectrogram_extractor = LogSpectrogramExtractor(
            self.config["preprocessing"]["frame_size"],
            self.config["preprocessing"]["hop_length"],
        )
        min_max_normaliser = MinMaxNormaliser(0, 1)
        preprocessing_pipeline = PreprocessingPipeline(
            padder, log_spectrogram_extractor, min_max_normaliser
        )
        spectrogram = preprocessing_pipeline.process_audio_bytes(audio_data)
        return torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def predict(self, audio_data: np.ndarray) -> Tuple[float, float, str]:
        with torch.no_grad():
            inputs = self._preprocess_audio(audio_data)
            inputs = inputs.to(self.device)
            outputs = self.model(inputs).detach()
            osc1_position = float(outputs[0, 0].item())
            osc1_param2 = float(outputs[0, 1].item())
            _, max_idx = torch.max(outputs[0, -5:], dim=0)
            osc1_wavetable = IDX_TO_OSC1_WAVETABLE[int(max_idx.item())]
            return osc1_position, osc1_param2, osc1_wavetable
