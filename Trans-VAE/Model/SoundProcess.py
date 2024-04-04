import os

import librosa
import numpy as np
import torch
from matplotlib import pyplot as plt


def preprocess_audio_for_model(wav_file_path, sr=22050, n_mels=128, hop_length=512, scaler=None,
                               target_shape=(128, 100)):
    # , target_shape=(128, 201)
    audio, _ = librosa.load(wav_file_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Adjust Mel-spectrogram size
    if mel_spec_db.shape[1] < target_shape[1]:
        pad_width = target_shape[1] - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    elif mel_spec_db.shape[1] > target_shape[1]:
        mel_spec_db = mel_spec_db[:, :target_shape[1]]

    # Normalize
    if scaler is not None:
        mel_spec_db_flat = mel_spec_db.flatten().reshape(1, -1)
        mel_spec_db_flat = scaler.transform(mel_spec_db_flat)
        mel_spec_db = mel_spec_db_flat.reshape(target_shape)

    processed_mel_spec = np.expand_dims(mel_spec_db, axis=0)
    return processed_mel_spec


def mel_spectrogram_to_audio(mel_spec, sr=22050, n_fft=2048, hop_length=512, win_length=None, n_iter=32):
    """
    Enhanced conversion from Mel-spectrogram to audio waveform using Griffin-Lim.
    """
    mel_spec = mel_spec.squeeze(0).numpy()
    if win_length is None:
        win_length = n_fft
    # Convert dB to power
    mel_spec = librosa.db_to_power(mel_spec)
    # Invert Mel to STFT
    stft = librosa.feature.inverse.mel_to_stft(mel_spec, sr=sr, n_fft=n_fft, power=1.0)
    # Apply Griffin-Lim
    audio = librosa.griffinlim(stft, n_iter=n_iter, hop_length=hop_length, win_length=win_length)
    return audio

def plot_melspectrograms(input_mel_spec, output_mel_spec, sr=22050, hop_length=512, save_path=None):
    """
    Plots the input and output Mel-spectrograms before and after the model.

    Parameters:
    - input_mel_spec: The input Mel-spectrogram to the model.
    - output_mel_spec: The Mel-spectrogram generated by the model.
    - sr: The sampling rate used for the Mel-spectrograms.
    - hop_length: The hop length used for the Mel-spectrograms.
    - save_path: Optional; if provided, the plot will be saved to this path.
    """
    # Set up the matplotlib figure and axes
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot input Mel-spectrogram
    axs[0].set_title('Input Mel-spectrogram')
    librosa.display.specshow(librosa.power_to_db(input_mel_spec, ref=np.max),
                             sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', ax=axs[0])
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Mel frequency')

    # Plot output Mel-spectrogram
    axs[1].set_title('Output Mel-spectrogram')
    librosa.display.specshow(librosa.power_to_db(output_mel_spec, ref=np.max),
                             sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', ax=axs[1])
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Mel frequency')

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    # Display the plots
    plt.tight_layout()
    plt.show()


def get_wav_files(base_path):
    wav_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.wav'):
                # Extract the emotion from the directory structure
                wav_files.append(os.path.join(root, file))
    return wav_files