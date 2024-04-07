import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display
from scipy.signal import butter, filtfilt


def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def modify_audio_for_anger(input_file, output_file, segment_length_sec, anger_intensity):
    """
    Apply various audio processing techniques to convey anger in speech.

    Parameters:
    - input_file: Path to the input audio file.
    - output_file: Path where the processed audio will be saved.
    - segment_length_sec: Length of each segment in seconds.
    - anger_intensity: Intensity of anger emotion (0 to 1, where 0 is neutral and 1 is maximum anger).
    """
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

        # Adjust pitch variation based on anger intensity
        # Higher anger intensity results in lower pitch variation
        pitch_variation = anger_intensity * 2  

        # Speed up speech rate based on anger intensity
        speech_rate_factor = 1.0 + anger_intensity * 0.3

        # Apply emotional prosody modulation
        # Adjust pitch
        segment_audio = librosa.effects.pitch_shift(segment_audio, sr=sr, n_steps=int(pitch_variation))

        # Slow down speech rate
        segment_audio = librosa.effects.time_stretch(segment_audio, rate=speech_rate_factor)
        
        # Increase volume
        segment_audio *= (1 + anger_intensity * 2)
        
        # Apply high-pass filter
        lowpass_cutoff = 500  # Hz
        segment_audio = highpass_filter(segment_audio, lowpass_cutoff, sr)

        # Distort
        segment_audio = np.tanh(segment_audio * 10)

        processed_audio = np.concatenate((processed_audio, segment_audio))

    return processed_audio
    # Write the processed audio to the output file
    #sf.write(output_file, processed_audio, sr)


# Paths to your input and output files
# input_wav_path = '0011_000002.wav'
# output_wav_path = 'output_angry.wav'

# Apply audio processing to convey anger
#modify_audio_for_anger(input_wav_path, output_wav_path, segment_length_sec=0.1, anger_intensity=0.8)
