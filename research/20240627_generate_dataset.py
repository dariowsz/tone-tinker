# %%
import start  # noqa isort:skip

# %%
import random

import numpy as np
import pandas as pd
from mido import Message  # not part of Pedalboard, but convenient!
from pedalboard._pedalboard import load_plugin
from scipy.io.wavfile import write
from tqdm import tqdm

from src.schemas import SynthParameter

# %%
# Load a VST3 or Audio Unit plugin from a known path on disk:
instrument = load_plugin(
    path_to_plugin_file="/Library/Audio/Plug-Ins/VST3/Massive.vst3",
)

# %%
df = pd.read_csv("research/massive_params.csv")
df["change"] = df["change"] == 1
df[df["change"]].head()

# %%
split = "val"
num_examples = 200
osc1_wavetable_subset = ["Squ-Sw I", "Sin-Tri", "Plysaw II", "Esca II", "A.I."]

data = []
for i in tqdm(range(num_examples)):
    example = {}
    for param_name in df[df["change"]]["param_name"]:
        synth_param = SynthParameter.from_audio_processor_parameter(
            instrument.parameters[param_name]  # type: ignore
        )
        if param_name == "osc1_wavetable":
            sampled_param_value = random.choices(osc1_wavetable_subset, k=1)[0]
        else:
            sampled_param_value = random.choices(synth_param.valid_values, k=1)[0]
        setattr(instrument, param_name, sampled_param_value)
        example[param_name] = sampled_param_value

    sample_rate = 48000
    audio = instrument(
        [Message("note_on", note=60), Message("note_off", note=60, time=2)],
        duration=2,  # seconds
        sample_rate=sample_rate,
        reset=False,
    )

    audio_int16 = np.int16(audio.transpose(1, 0) * 32767)

    # write("output.wav", sample_rate, audio.transpose(1, 0))
    filename = f"output_{i}.wav"
    write(f"data/{split}/{filename}", sample_rate, audio_int16)

    example["audio_path"] = filename
    data.append(example)

df = pd.DataFrame(data)
df.to_csv(f"data/{split}.csv", index=False)

# %%
df = pd.read_csv("data/train.csv")
df["osc1_wavetable"].value_counts()
# %%
