# %%
import start  # noqa isort:skip

# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram
from torchsummary import summary

from src.dataset import PresetSamplesDataset
from src.ml_models import Autoencoder

# %%
# Hyperparameters
INPUT_SHAPE = (1, 28, 28)
NUM_EPOCHS = 10
LEARNING_RATE = 0.0005

# %%
model = Autoencoder(
    input_shape=INPUT_SHAPE,
    conv_filters=[32, 64, 64, 64],
    conv_kernels=[3, 3, 3, 3],
    conv_strides=[1, 2, 2, 1],
    latent_space_dim=512,
    decoder_output_padding=[0, 1, 1, 0],
)
summary(model, input_size=INPUT_SHAPE)

# %%
dataset = PresetSamplesDataset("data", split="train")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# %%
# Example of compiling and training the autoencoder
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# %%
# Training loop
for batch in dataloader:
    print(batch)
    break
    # optimizer.zero_grad()
    # outputs = model(x_train)
    # loss = criterion(outputs, x_train)
    # loss.backward()
    # optimizer.step()
    # print(f"Epoch [{epoch + 1}/10], Loss: {loss.item():.4f}")

# %%
# Configuration of MelSpectrogram transformation
sample_rate = 16000  # Change this to match the sample rate of your waveform
n_fft = 400  # Number of FFT bins
win_length = None  # Window length (can be set to the same as n_fft)
hop_length = 160  # Hop length (shift between windows)
n_mels = 128  # Number of Mel filterbanks

# Initialize the MelSpectrogram transform
mel_spectrogram_transform = MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    n_mels=n_mels,
)

mel_spectrogram = mel_spectrogram_transform(batch[0][0])

# %%
print(batch[0][0].shape)
mel_spectrogram.shape

# %%
