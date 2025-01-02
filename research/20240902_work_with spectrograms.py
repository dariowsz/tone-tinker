# %%
import start  # noqa isort:skip

# %%
import numpy as np

# %%
file_path = "data/train_preprocessed/spectrograms/output_418.wav.npy"
spectrogram = np.load(file_path)
spectrogram.shape

# %%
