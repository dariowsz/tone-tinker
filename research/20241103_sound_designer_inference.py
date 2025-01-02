# %%
import start  # noqa isort:skip

# %%
import torch
from torch.utils.data import DataLoader

from src.dataset import PresetSamplesDataset
from src.ml_models import Autoencoder, SoundDesigner

# %%
# Constants
AUTOENCODER_PRETRAINED_PATH = (
    "checkpoints/autoencoder-val-loss-0.00305-log-specs-dim-64.pth"
)
MODEL_PATH = "checkpoints/sound-designer-val-loss-0.30391.pth"
DATA_PATH = "data"

# %%
autoencoder = Autoencoder(
    input_shape=(1, 256, 376),
    conv_filters=(32, 64, 128, 256),
    conv_kernels=(3, 3, 3, 3),
    conv_strides=(2, 2, 2, 2),
    decoder_output_padding=(1, 1, 1, (1, 0)),
    latent_space_dim=64,
)
autoencoder.load_state_dict(torch.load(AUTOENCODER_PRETRAINED_PATH))
model = SoundDesigner(encoder=autoencoder.encoder)
model.load_state_dict(torch.load(MODEL_PATH))
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
    count += 1
    if count > 10:
        break

# %%
