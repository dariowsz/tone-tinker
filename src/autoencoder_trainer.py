import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ml_models import Autoencoder
from utils import (
    CheckpointManager,
    TensorBoardLogger,
    get_device,
    load_config,
    parse_args,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_fsdd(spectrograms_path: str) -> torch.Tensor:
    X = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)  # (n_bins, n_frames, 1)
            X.append(spectrogram)
    X = np.array(X)
    X = X[..., np.newaxis]  # -> (B, H, W, C)
    return torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)


def train(
    autoencoder: nn.Module,
    x_train: torch.Tensor,
    x_val: torch.Tensor,
    config: dict,
    optimizer: optim.Optimizer,
):
    device = get_device()
    logger = TensorBoardLogger(config["log_dir"])
    checkpoint_manager = CheckpointManager(
        save_dir=f"checkpoints/autoencoder/{autoencoder.latent_space_dim}/{time.strftime('%Y-%m-%d_%H:%M:%S')}",
        model_name="autoencoder",
        config=config,
    )

    autoencoder = autoencoder.to(device)
    x_train = x_train.to(device)
    x_val = x_val.to(device)

    train_dataset = TensorDataset(x_train)
    val_dataset = TensorDataset(x_val)
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    criterion = torch.nn.MSELoss()

    logger.log_model_graph(autoencoder, next(iter(train_loader))[0])

    for epoch in range(config["epochs"]):
        autoencoder.train()
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch[0]
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if batch_idx % 10 == 0:
                logger.log_training_step(
                    loss.item(), epoch * len(train_loader) + batch_idx
                )

        train_loss = epoch_loss / len(train_loader)
        logger.log_training_epoch(train_loss, epoch)
        logging.info(f"Epoch {epoch + 1}/{config['epochs']}, Loss: {train_loss}")

        # Validation phase every n epochs
        if (epoch + 1) % config["validation_interval"] == 0:
            autoencoder.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch[0]
                    outputs = autoencoder(inputs)
                    # logger.log_spectrograms(inputs, outputs, epoch)
                    loss = criterion(outputs, inputs)
                    val_loss += loss.item()

            val_loss = val_loss / len(val_loader)
            logger.log_validation(val_loss, epoch)
            logging.info(f"Validation Loss: {val_loss}")
            checkpoint_manager.save_checkpoint(
                autoencoder, optimizer, epoch, val_loss, train_loss
            )

    logger.close()
    return autoencoder


if __name__ == "__main__":
    args = parse_args(
        description="Train autoencoder model",
        default_config_path="configs/autoencoder_training_config.yaml",
    )
    config = load_config(args.config)

    autoencoder = Autoencoder(
        input_shape=config["model"]["input_shape"],
        conv_filters=config["model"]["conv_filters"],
        conv_kernels=config["model"]["conv_kernels"],
        conv_strides=config["model"]["conv_strides"],
        decoder_output_padding=config["model"]["decoder_output_padding"],
        latent_space_dim=config["model"]["latent_space_dim"],
    )

    optimizer = optim.Adam(
        autoencoder.parameters(), lr=config["training"]["learning_rate"]
    )

    if args.checkpoint:
        autoencoder, optimizer = CheckpointManager.load_checkpoint(
            args.checkpoint, autoencoder, optimizer
        )

    x_train = load_fsdd(config["data"]["train_spectrograms_path"])
    x_val = load_fsdd(config["data"]["val_spectrograms_path"])

    autoencoder = train(autoencoder, x_train, x_val, config["training"], optimizer)  # type: ignore
