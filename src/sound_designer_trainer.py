import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import PresetSamplesDataset
from ml_models import Autoencoder, SoundDesigner
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


def train(
    model: nn.Module,
    data_path: str,
    config: dict,
    optimizer: optim.Optimizer,
):
    device = get_device()
    logger = TensorBoardLogger(config["log_dir"])
    checkpoint_manager = CheckpointManager(
        save_dir=f"checkpoints/sound_designer/{model.latent_space_dim}/{time.strftime('%Y-%m-%d_%H:%M:%S')}",
        model_name="sound_designer",
        config=config,
    )

    model = model.to(device)

    train_dataset = PresetSamplesDataset(data_path, split="train", preprocessed=True)
    val_dataset = PresetSamplesDataset(data_path, split="val", preprocessed=True)
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Two loss functions: classification for the class labels and regression for the continuous values
    classification_criterion = nn.BCELoss()
    regression_criterion = nn.MSELoss()

    logger.log_model_graph(model, next(iter(train_loader))[0].to(device))

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0
        for batch_idx, (inputs, _, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            # Split outputs and targets
            class_outputs = outputs[:, :5]
            reg_outputs = outputs[:, 5:]
            class_targets = targets[:, :5]
            reg_targets = targets[:, 5:]

            class_loss = classification_criterion(class_outputs, class_targets)
            reg_loss = regression_criterion(reg_outputs, reg_targets)

            # REVIEW: Introduce parameter to balance the loss between classification and regression if needed
            total_loss = class_loss + reg_loss

            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

            if batch_idx % 10 == 0:
                logger.log_training_step(
                    total_loss.item(), epoch * len(train_loader) + batch_idx
                )

        train_loss = epoch_loss / len(train_loader)
        logger.log_training_epoch(train_loss, epoch)
        logging.info(f"Epoch {epoch + 1}/{config['epochs']}, Loss: {train_loss}")

        # Validation phase every n epochs
        if (epoch + 1) % config["validation_interval"] == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, _, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)

                    # Split outputs and targets
                    class_outputs = outputs[:, :5]
                    reg_outputs = outputs[:, 5:]
                    class_targets = targets[:, :5]
                    reg_targets = targets[:, 5:]

                    # Calculate losses
                    class_loss = classification_criterion(class_outputs, class_targets)
                    reg_loss = regression_criterion(reg_outputs, reg_targets)
                    total_loss = class_loss + reg_loss

                    val_loss += total_loss.item()

            avg_val_loss = val_loss / len(val_loader)
            logger.log_validation(avg_val_loss, epoch)
            logging.info(f"Validation Loss: {avg_val_loss}")
            checkpoint_manager.save_checkpoint(
                model, optimizer, epoch, avg_val_loss, train_loss
            )

    logger.close()
    return model


if __name__ == "__main__":
    args = parse_args(
        description="Train SoundDesigner model",
        default_config_path="configs/sound_designer_training_config.yaml",
    )
    config = load_config(args.config)

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

    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    if args.checkpoint:
        model, optimizer = CheckpointManager.load_checkpoint(
            args.checkpoint, model, optimizer
        )

    model = train(
        model, data_path="data", config=config["training"], optimizer=optimizer  # type: ignore
    )
