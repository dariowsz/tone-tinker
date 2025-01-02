# Tone Tinker

Tone Tinker is a machine learning project that predicts Native Instruments Massive VST parameters from audio samples. The project uses a two-stage model architecture to reverse-engineer synthesizer parameters from raw audio.

## Examples
### Original:
https://github.com/user-attachments/assets/effeb7bd-e8b6-403d-9823-f26381d72c5f


https://github.com/user-attachments/assets/90be5994-e3ca-42dd-a99d-966204039fd4


https://github.com/user-attachments/assets/c452f15b-81a9-45cd-be28-4d509a3c1538


### Reconstructed
https://github.com/user-attachments/assets/ce943f05-9496-4445-a50c-d9c5ec043602


https://github.com/user-attachments/assets/50b195a3-8485-4cd5-aad9-f441dbb4c3d5


https://github.com/user-attachments/assets/eaaf1203-caa3-460b-9cc9-9f6171863068


## Overview

The project consists of two main components:

1. **Autoencoder**: Compresses audio spectrograms into a lower-dimensional latent space
2. **Sound Designer**: Predicts synthesizer parameters from the encoded audio representation

## Architecture
<img width="1004" alt="tone-tinker-arch" src="https://github.com/user-attachments/assets/869050bc-5cc7-491a-b8ee-7bcec100beaf" />

### Data Pipeline
1. Raw audio (.wav) files → Log spectrograms
2. Spectrograms → Autoencoder → Latent representation
3. Latent representation → Sound Designer → Synthesizer parameters

### Models
- **Autoencoder**: Convolutional neural network that learns to compress and reconstruct audio spectrograms
- **Sound Designer**: Multi-Layer Perceptron that predicts:
  - 2 continuous parameters (regression)
  - 1 categorical parameter (5-class classification for wavetable selection)

## Dataset

The dataset was generated programmatically using:
- Spotify's Pedalboard library to interface with the Massive VST
- 1000 training presets with randomized parameters
- Parameters sampled:
  - 2 continuous parameters
  - Wavetable selection (5 options: "Squ-Sw I", "Sin-Tri", "Plysaw II", "Esca II", "A.I.")

## Getting Started

### Prerequisites
- Python 3.11+
- Poetry 1.8.3
- Native Instruments Massive VST3

### Installation

```bash
git clone https://github.com/dariowsz/tone-tinker.git
cd tone-tinker
poetry install
pip install -r requirements.macos.txt
```

### Training

1. Generate the dataset:
Follow the instructions in `research/20240627_generate_dataset.py`.

2. Preprocess the data (this will generate the spectrograms and save them to disk):
```bash
python src/preprocess.py
```

3. Train the autoencoder:
```bash
python src/autoencoder_trainer.py
```

4. Train the sound designer:
```bash
python src/sound_designer_trainer.py
```

## Project Status

This is an initial proof of concept with a limited parameter set. Future improvements may include:
- Expanding to more synthesizer parameters
- Exploring more complex model architectures
- Using other AI techniques like Reinforcement Learning to train the sound designer


## Acknowledgments

- Spotify's Pedalboard library for VST automation
- Native Instruments Massive VST
