import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display
def compute_spectrum(y, sr):
    # Placeholder function for computing spectrum, you might have your own implementation
    D = np.abs(librosa.stft(y))**2
    S = librosa.feature.melspectrogram(S=D, sr=sr)
    DB = librosa.power_to_db(S, ref=np.max)
    return DB

def plot_spectrum(original_DB, modified_DB, sr):
    # Placeholder function for plotting, adjust according to your needs
    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    img1 = librosa.display.specshow(original_DB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax[0])
    ax[0].set(title='Original')
    ax[0].label_outer()
    img2 = librosa.display.specshow(modified_DB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax[1])
    ax[1].set(title='Modified')
    plt.colorbar(img1, ax=ax[0])
    plt.colorbar(img2, ax=ax[1])
    plt.show()

def emotional_prosody_modulation(input_file, output_file, segment_length_sec, sadness_intensity):
    """
    Apply emotional prosody modulation to segmented audio to convey sadness.

    Parameters:
    - input_file: Path to the input audio file.
    - output_file: Path where the processed audio will be saved.
    - segment_length_sec: Length of each segment in seconds.
    - sadness_intensity: Intensity of sadness emotion (0 to 1, where 0 is neutral and 1 is maximum sadness).
    """
    # Load the input audio file
    y, sr = librosa.load(input_file)

    # Calculate segment length in samples
    segment_length_samples = int(segment_length_sec * sr)

    # Total number of segments
    total_segments = int(np.ceil(len(y) / segment_length_samples))

    # Create a new list to hold processed audio
    processed_audio = np.array([])

    # Process each segment
    for segment in range(total_segments):
        start = segment * segment_length_samples
        end = min((segment + 1) * segment_length_samples, len(y))
        segment_audio = y[start:end]

        # Adjust pitch variation based on sadness intensity
        # Higher sadness intensity results in lower pitch variation
        pitch_variation = -sadness_intensity * 2  # Shift pitch down for more sadness

        # Slow down speech rate based on sadness intensity
        # Higher sadness intensity results in slower speech rate
        speech_rate_factor = 1.0 - sadness_intensity * 0.3

        # Apply emotional prosody modulation
        # Adjust pitch
        segment_audio = librosa.effects.pitch_shift(segment_audio, sr=sr, n_steps=int(pitch_variation))

        # Slow down speech rate
        segment_audio = librosa.effects.time_stretch(segment_audio, rate=speech_rate_factor)

        processed_audio = np.concatenate((processed_audio, segment_audio))

    return processed_audio,sr

# Paths to your input and output files
# input_wav_path = '0011_000002.wav'
# output_wav_path = 'output_sad.wav'

# Apply emotional prosody modulation to convey sadness
#emotional_prosody_modulation(input_wav_path, output_wav_path, segment_length_sec=0.1, sadness_intensity=1)
