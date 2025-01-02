# %%
import start  # noqa isort:skip

# %%
import os
import pickle

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import Audio, display

from src.ml_models.autoencoder import Autoencoder
from src.preprocess import MinMaxNormaliser

# %%
# Constants
FILE_PATH = "output_0.wav"
HOP_LENGTH = 256
MIN_MAX_VALUES_PATH = "data/val_preprocessed/min_max_values.pkl"
MODEL_PATH = (
    "checkpoints/older_trains/autoencoder-val-loss-0.00305-log-specs-dim-64.pth"
)
SPECTROGRAMS_PATH = "data/val_preprocessed/spectrograms"
SR = 48000


# %%
# Helper functions
min_max_normaliser = MinMaxNormaliser(0, 1)


def load_fsdd(spectrograms_path):
    x_train = []
    file_paths = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)
            x_train.append(spectrogram)
            file_paths.append(file_path)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis]
    return x_train, file_paths


def select_spectrograms(spectrograms, file_paths, min_max_values, num_spectrograms=2):
    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms)
    sampled_spectrogrmas = spectrograms[sampled_indexes]
    file_paths = [file_paths[index] for index in sampled_indexes]
    sampled_min_max_values = [min_max_values[file_path] for file_path in file_paths]
    print(file_paths)
    print(sampled_min_max_values)
    return sampled_spectrogrmas, sampled_min_max_values


def convert_spectrograms_to_audio(spectrograms, min_max_values):
    signals = []
    for spectrogram, min_max_value in zip(spectrograms, min_max_values):
        # reshape the log spectrogram
        log_spectrogram = spectrogram[:, :, 0]
        # apply denormalisation
        denorm_log_spec = min_max_normaliser.denormalise(
            log_spectrogram, min_max_value["min"], min_max_value["max"]
        )
        # log spectrogram -> spectrogram
        spec = librosa.db_to_amplitude(denorm_log_spec)
        # apply Griffin-Lim
        signal = librosa.istft(spec, hop_length=HOP_LENGTH)
        # append signal to "signals"
        signals.append(signal)
    return signals


# %%
model = Autoencoder(
    input_shape=(1, 256, 376),
    conv_filters=(32, 64, 128, 256),
    conv_kernels=(3, 3, 3, 3),
    conv_strides=(2, 2, 2, 2),
    decoder_output_padding=(1, 1, 1, (1, 0)),
    latent_space_dim=64,
)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# %%
with open(MIN_MAX_VALUES_PATH, "rb") as f:
    min_max_values = pickle.load(f)

specs, file_paths = load_fsdd(SPECTROGRAMS_PATH)

sampled_specs, sampled_min_max_values = select_spectrograms(
    specs, file_paths, min_max_values, 5
)

original_signals = convert_spectrograms_to_audio(sampled_specs, sampled_min_max_values)

# %%
# FIXME: The original audio is not the same as the one after converting from the spectrogram
print("Original signals")
for signal in original_signals:
    display(Audio(signal, rate=SR))

# %%
# Convert list of numpy arrays to a batched tensor
preprocessed_audio = [torch.tensor(spec, dtype=torch.float32) for spec in sampled_specs]
preprocessed_audio = torch.stack(preprocessed_audio)
preprocessed_audio = preprocessed_audio.permute(0, 3, 1, 2)
preprocessed_audio.shape

# %%
reconstructed_specs = model(preprocessed_audio)
reconstructed_specs.shape

# %%
# Convert batched tensor back to list of numpy arrays
reconstructed_specs = reconstructed_specs.permute(0, 2, 3, 1).detach().numpy()
reconstructed_specs = [spec for spec in reconstructed_specs]
reconstructed_signals = convert_spectrograms_to_audio(
    reconstructed_specs, sampled_min_max_values
)

# %%
# FIXME: The original audio is not the same as the one after converting from the spectrogram
print("Reconstructued signals")
for signal in reconstructed_signals:
    display(Audio(signal, rate=SR))


# %%
# Plot the original mel spectrogram
example = 4

plt.figure(figsize=(10, 4))
librosa.display.specshow(
    sampled_specs[example][:, :, 0], sr=SR, x_axis="time", y_axis="mel"
)
plt.colorbar(format="%+2.0f dB")
plt.title("Original Mel Spectrogram")
plt.tight_layout()
plt.show()

# Plot the reconstructed mel spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(
    reconstructed_specs[example][:, :, 0],
    sr=SR,
    x_axis="time",
    y_axis="mel",
)
plt.colorbar(format="%+2.0f dB")
plt.title("Reconstructed Mel Spectrogram")
plt.tight_layout()
plt.show()

# %%
