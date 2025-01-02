import time

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter


class TensorBoardLogger:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(f"{log_dir}/{time.strftime('%Y-%m-%d_%H:%M:%S')}")

    def log_training_step(self, loss, global_step):
        self.writer.add_scalar("Loss/train_batch", loss, global_step)

    def log_training_epoch(self, loss, epoch):
        self.writer.add_scalar("Loss/train_epoch", loss, epoch)

    def log_validation(self, loss, epoch):
        self.writer.add_scalar("Loss/validation", loss, epoch)

    # For debugging purposes only
    # FIXME: This method is not working correctly
    def log_spectrograms(self, originals, reconstructed, epoch):
        """Log spectrograms as images to tensorboard.

        Args:
            originals (torch.Tensor): Original spectrograms [B, C, F, T]
            reconstructed (torch.Tensor): Reconstructed spectrograms [B, C, F, T]
            epoch (int): Current epoch
        """
        # Convert to numpy and remove channel dimension if it's 1
        orig_specs = originals.cpu().detach().numpy()
        recon_specs = reconstructed.cpu().detach().numpy()
        if orig_specs.shape[1] == 1:
            orig_specs = orig_specs[:, 0]
            recon_specs = recon_specs[:, 0]

        # Create figure with subplots for each pair of spectrograms
        fig, axes = plt.subplots(
            2, orig_specs.shape[0], figsize=(4 * orig_specs.shape[0], 8)
        )

        for i in range(orig_specs.shape[0]):
            # Plot original
            im = axes[0, i].imshow(
                orig_specs[i], aspect="auto", origin="lower", cmap="viridis"
            )
            axes[0, i].set_title(f"Original {i+1}")
            plt.colorbar(im, ax=axes[0, i])

            # Plot reconstruction
            im = axes[1, i].imshow(
                recon_specs[i], aspect="auto", origin="lower", cmap="viridis"
            )
            axes[1, i].set_title(f"Reconstructed {i+1}")
            plt.colorbar(im, ax=axes[1, i])

        plt.tight_layout()

        # Log the figure to tensorboard
        self.writer.add_figure("Spectrograms", fig, epoch)
        plt.close(fig)

        # Optionally, also log the difference between original and reconstructed
        diff = np.abs(orig_specs - recon_specs)
        fig, axes = plt.subplots(
            1, orig_specs.shape[0], figsize=(4 * orig_specs.shape[0], 4)
        )

        for i in range(diff.shape[0]):
            im = axes[i].imshow(diff[i], aspect="auto", origin="lower", cmap="magma")
            axes[i].set_title(f"Difference {i+1}")
            plt.colorbar(im, ax=axes[i])

        plt.tight_layout()
        self.writer.add_figure("Reconstruction Difference", fig, epoch)
        plt.close(fig)

        # Log reconstruction error distribution
        self.writer.add_histogram("Reconstruction Error", diff.flatten(), epoch)

    def log_model_graph(self, model, input_to_model):
        self.writer.add_graph(model, input_to_model)

    # For debugging purposes only
    def log_model_parameters(self, model, epoch):
        for name, param in model.named_parameters():
            self.writer.add_histogram(f"Parameters/{name}", param.data, epoch)
            if param.grad is not None:
                self.writer.add_histogram(f"Gradients/{name}", param.grad, epoch)

    def close(self):
        self.writer.close()
