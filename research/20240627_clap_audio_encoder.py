# %%
import start  # noqa isort:skip

# %%
import laion_clap
import librosa
import numpy as np
import torch

# %%
# Quantization
quantize_data = True


def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1.0, a_max=1.0)
    return (x * 32767.0).astype(np.int16)


# %%
model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt("checkpoints/630k-audioset-best.pt")
model.eval()

# %%
audio_data, _ = librosa.load("output.wav", sr=48000)  # sample rate should be 48000
audio_data = audio_data.reshape(1, -1)  # Make it (1,T) or (N,T)
if quantize_data:
    audio_data = torch.from_numpy(
        int16_to_float32(float32_to_int16(audio_data))
    ).float()  # quantize before send it in to the model
audio_embed = model.get_audio_embedding_from_data(x=audio_data, use_tensor=True)
print(audio_embed[:, -20:])
print(audio_embed.shape)

# %%
with open("test.npy", "wb") as f:
    np.save(f, np.array([audio_embed.detach().numpy()] * 4))

# %%
