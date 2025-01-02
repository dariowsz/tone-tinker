import torch.nn as nn


class SoundDesigner(nn.Module):
    def __init__(self, encoder):
        super(SoundDesigner, self).__init__()
        self.latent_space_dim = encoder[-1].out_features

        self.encoder = encoder
        # Freeze encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.designer = self._build_designer()

    def forward(self, x):
        latent_space = self.encoder(x)
        designer_output = self.designer(latent_space)
        return designer_output

    # REVIEW: Other designer architectures could be explored
    def _build_designer(self):
        return nn.Sequential(
            nn.Linear(self.latent_space_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 7),
            nn.Sigmoid(),
        )
