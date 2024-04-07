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


def segmented_pitch_shift(input_file, output_file, segment_length_sec, steps):
    """
    Apply pitch shifting to segmented audio.
    
    Parameters:
    - input_file: Path to the input audio file.
    - output_file: Path where the processed audio will be saved.
    - segment_length_sec: Length of each segment in seconds.
    - steps: Number of semitones to shift the audio.
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
        
        # Apply pitch shift to this segment
        shifted_segment = librosa.effects.pitch_shift(segment_audio, sr, steps)
        
        # Concatenate the processed segment
        processed_audio = np.concatenate((processed_audio, shifted_segment))
    
    # Write the processed audio to the output file
    sf.write(output_file, processed_audio, sr)



def segmented_pitch_shift_with_variability_and_plot(input_file, output_file, segment_length_sec, steps, shift_probability=0.5):
    """
    Apply pitch shifting to segmented audio with variability and plot the spectrum.
    
    Parameters:
    - input_file: Path to the input audio file.
    - output_file: Path where the processed audio will be saved.
    - segment_length_sec: Length of each segment in seconds.
    - steps: Number of semitones to shift the audio.
    - shift_probability: Probability of a segment being pitch shifted.
    """
    # Load the input audio file
    y, sr = librosa.load(input_file)
    
    # Compute and plot spectrum of the original audio
    #original_DB = compute_spectrum(y, sr)

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
        
        # Randomly decide whether to apply pitch shift to this segment
        if np.random.rand() < shift_probability:
            segment_audio = librosa.effects.pitch_shift(segment_audio, sr, steps)
        
        processed_audio = np.concatenate((processed_audio, segment_audio))
    
    #  # Generate and add white noise
    # noise = np.random.normal(0, 0.001, processed_audio.shape)
    # noisy_audio = processed_audio + noise
    
    # # Normalize to prevent clipping
    # max_val = max(np.abs(noisy_audio))
    # if max_val > 1.0:
    #     noisy_audio /= max_val

    # Write the processed audio to the output file
    #sf.write(output_file, noisy_audio, sr)
    
    # Compute and plot spectrum of the modified audio
    #modified_DB = compute_spectrum(noisy_audio, sr)
    
    # Plot the original and modified spectra
    #plot_spectrum(original_DB, modified_DB, sr)

    return processed_audio, sr

def increase_pitch(y,sr):
    new_y = librosa.effects.pitch_shift(y, sr, 5)
    return new_y

def emotional_prosody_modulation_happy(input_file, output_file, segment_length_sec, happiness_intensity):
    """
    Apply emotional prosody modulation to segmented audio to convey happiness.

    Parameters:
    - input_file: Path to the input audio file.
    - output_file: Path where the processed audio will be saved.
    - segment_length_sec: Length of each segment in seconds.
    - happiness_intensity: Intensity of happiness emotion (0 to 1, where 0 is neutral and 1 is maximum happiness).
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

        # Adjust pitch variation based on happiness intensity
        # Higher happiness intensity results in higher pitch variation
        pitch_variation_factor = 3  # Shift pitch up for more happiness

        # Speed up speech rate based on happiness intensity
        # Higher happiness intensity may result in a faster speech rate
        speech_rate_factor = 1.0 + happiness_intensity * 0.25  # More pronounced speed increase

        # Apply emotional prosody modulation
        # Adjust pitch
        segment_audio = librosa.effects.pitch_shift(segment_audio, sr=sr, n_steps=int(pitch_variation_factor))

        # Increase speech rate
        segment_audio = librosa.effects.time_stretch(segment_audio, rate=speech_rate_factor)

        processed_audio = np.concatenate((processed_audio, segment_audio))

    return processed_audio, sr


# Paths to your input and output files
# input_wav_path = '0019_000018.wav'
# output_wav_path = 'output_random_pitch_noise.wav'
#segmented_pitch_shift_with_variability_and_plot(input_wav_path, output_wav_path,0.1,4.5)