# %%
import start  # noqa isort:skip

# %%
import numpy as np
import torch
from mido import Message  # not part of Pedalboard, but convenient!
from pedalboard._pedalboard import load_plugin
from scipy.io.wavfile import write
from torch.utils.data import DataLoader

from src.dataset import PresetSamplesDataset
from src.ml_models import Autoencoder, SoundDesigner
from src.utils.checkpoint_manager import CheckpointManager

# %%
# Constants
AUTOENCODER_PRETRAINED_PATH = (
    "checkpoints/autoencoder/64/2024-07-05_00:41:16/autoencoder_best.pth"
)
MODEL_PATH = "checkpoints/sound_designer/64/2024-07-05_00:59:27/sound_designer_best.pth"
DATA_PATH = "data"
SAMPLE_RATE = 48000
IDX_TO_OSC1_WAVETABLE = {
    0: "A.I.",
    1: "Esca II",
    2: "Plysaw II",
    3: "Sin-Tri",
    4: "Squ-Sw I",
}

# %%
instrument = load_plugin("/Library/Audio/Plug-Ins/VST3/Massive.vst3")

# %%
autoencoder = Autoencoder(
    input_shape=(1, 256, 376),
    conv_filters=(32, 64, 128, 256),
    conv_kernels=(3, 3, 3, 3),
    conv_strides=(2, 2, 2, 2),
    decoder_output_padding=(1, 1, 1, (1, 0)),
    latent_space_dim=64,
)
autoencoder, _ = CheckpointManager.load_checkpoint(
    AUTOENCODER_PRETRAINED_PATH, autoencoder
)
model = SoundDesigner(encoder=autoencoder.encoder)
model, _ = CheckpointManager.load_checkpoint(MODEL_PATH, model)
model.eval()

# %%
val_dataset = PresetSamplesDataset(DATA_PATH, split="val", preprocessed=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# %%
count = 0
for batch in val_loader:
    inputs, sample_path, targets = batch
    outputs = model(inputs).detach()
    # Get the last 4 digits and find the index of the maximum value
    _, max_idx = torch.max(outputs[0, -5:], dim=0)
    # Create a new tensor of zeros and set the max index to 1
    one_hot = torch.zeros_like(outputs[0, -5:])
    one_hot[max_idx] = 1
    # Replace the last 4 values with one-hot encoded values
    outputs[0, -5:] = one_hot
    print("-" * 100)
    print("Sample path:", sample_path)
    print("Expected:", targets)
    print("Output:", outputs)
    # Create reconstructed preset with pedalboard
    setattr(instrument, "osc1_wavetable", IDX_TO_OSC1_WAVETABLE[int(max_idx.item())])
    setattr(instrument, "osc1_position", round(float(outputs[0, 0].item()), 2) * 100)
    setattr(instrument, "osc1_param2", round(float(outputs[0, 1].item()), 2) * 100)
    audio = instrument(
        [Message("note_on", note=60), Message("note_off", note=60, time=2)],
        duration=2,  # seconds
        sample_rate=SAMPLE_RATE,
        reset=False,
    )
    audio_int16 = np.int16(audio.transpose(1, 0) * 32767)
    write(f"reconstructed_output_{count}.wav", SAMPLE_RATE, audio_int16)
    count += 1
    if count > 5:
        break

# %%
