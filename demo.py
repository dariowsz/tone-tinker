import io

import librosa
import streamlit as st
import yaml

# Add this as the first Streamlit command
st.set_page_config(page_title="Tone Tinker Demo")

from src.inference import SoundDesignerInference


@st.cache_resource
def load_model():
    with open("configs/sound_designer_training_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = SoundDesignerInference(
        model_path="checkpoints/sound_designer/64/2024-07-05_00:59:27/sound_designer_best.pth",
        config=config,
    )
    return model


def main():
    st.title("Massive Preset Generator v1")

    model = load_model()

    # File uploader - restrict to wav files only
    uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

    if uploaded_file is not None:
        # Display file details
        st.write("Filename:", uploaded_file.name)
        st.write("File size:", uploaded_file.size, "bytes")

        # Read WAV file
        try:
            audio_bytes = uploaded_file.read()
            audio_data = librosa.load(
                io.BytesIO(audio_bytes), sr=48000, mono=True, duration=2
            )[0]

            # Add audio player
            st.audio(audio_bytes, format="audio/wav")

            # Get model predictions
            osc1_position, osc1_param2, osc1_wavetable = model.predict(audio_data)

            # Display predictions
            st.write("## Predicted Parameters")
            st.write("OSC1 Position:", osc1_position)
            st.write("OSC1 Parameter 2:", osc1_param2)
            st.write("OSC1 Wavetable:", osc1_wavetable)

        except Exception as e:
            st.error(f"Error processing the audio file: {str(e)}")


if __name__ == "__main__":
    main()
