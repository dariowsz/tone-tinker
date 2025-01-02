# Based on the following Autoencoder implementation: https://github.com/musikalkemist/generating-sound-with-neural-networks/blob/main/06%20Training%20an%20autoencoder/autoencoder.py

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_shape,
        conv_filters,
        conv_kernels,
        conv_strides,
        latent_space_dim,
        decoder_output_padding,
    ):
        super(Autoencoder, self).__init__()
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim
        self.decoder_output_padding = decoder_output_padding

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def forward(self, x):
        latent_space = self.encoder(x)
        reconstructed = self.decoder(latent_space)
        return reconstructed

    def _build_encoder(self):
        layers = []
        in_channels = self.input_shape[0]

        for i in range(self._num_conv_layers):
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.conv_filters[i],
                    kernel_size=self.conv_kernels[i],
                    stride=self.conv_strides[i],
                    padding=1,
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(self.conv_filters[i]))
            in_channels = self.conv_filters[i]

        encoder = nn.Sequential(*layers)
        self._shape_before_bottleneck = self._calculate_shape_before_bottleneck(encoder)

        return nn.Sequential(
            *encoder,
            nn.Flatten(),
            nn.Linear(self._shape_before_bottleneck, self.latent_space_dim),
        )

    def _build_decoder(self):
        layers = [
            nn.Linear(self.latent_space_dim, self._shape_before_bottleneck),  # type: ignore
            nn.Unflatten(1, self._shape_before_bottleneck_shape),
        ]

        in_channels = self.conv_filters[-1]
        for i in range(self._num_conv_layers - 1, 0, -1):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=self.conv_filters[i - 1],
                    kernel_size=self.conv_kernels[i],
                    stride=self.conv_strides[i],
                    padding=1,
                    output_padding=self.decoder_output_padding[i],
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(self.conv_filters[i - 1]))
            in_channels = self.conv_filters[i - 1]

        layers.append(
            nn.ConvTranspose2d(
                in_channels=self.conv_filters[0],
                out_channels=1,
                kernel_size=self.conv_kernels[0],
                stride=self.conv_strides[0],
                padding=1,
                output_padding=self.decoder_output_padding[0],
            )
        )
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def _calculate_shape_before_bottleneck(self, encoder):
        with torch.no_grad():
            x = torch.zeros(1, *self.input_shape)
            x = encoder(x)
            self._shape_before_bottleneck_shape = x.shape[1:]
            return int(torch.prod(torch.tensor(x.shape[1:])))
