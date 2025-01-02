# %%
import start  # noqa isort:skip

# %%
from mido import Message  # not part of Pedalboard, but convenient!
from pedalboard._pedalboard import load_plugin
from scipy.io.wavfile import write

# %%
# Load a VST3 or Audio Unit plugin from a known path on disk:
instrument = load_plugin(
    path_to_plugin_file="/Library/Audio/Plug-Ins/VST3/Massive.vst3",
)

# %%
# Change some parameters:
instrument.osc1_pitch = -12.0  # type: ignore

# Render some audio by passing MIDI to an instrument:
sample_rate = 44100
audio = instrument(
    [Message("note_on", note=60), Message("note_off", note=60, time=2)],
    duration=2,  # seconds
    sample_rate=sample_rate,
    reset=False,
)

# %%
write("output.wav", sample_rate, audio.transpose(1, 0))

# %%
instrument.parameters  # type: ignore

# %%
