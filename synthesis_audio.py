import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.models import Model
import numpy as np
from sklearn.preprocessing import StandardScaler
import soundfile as sf
import joblib

def load_transcriptions(transcription_path):
    transcriptions = {}
    counter = 0
    flag = 0
    with open(transcription_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                recording_id, text, emotion = parts
        
                transcriptions[recording_id] = (text, emotion)
    print(len(transcriptions))
    return transcriptions

def load_transcriptions(transcription_path):
    transcriptions = {}
    with open(transcription_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                recording_id, text, emotion = parts
                transcriptions[recording_id] = (text, emotion)
    return transcriptions

def derive_happy_id(neutral):
    latter = neutral.split("_")
    new_part = str(int(latter[-1]) + 700)
    new_latter = new_part.zfill(6)
    new_id = latter[0] + "_" + new_latter
    return new_id

def process_speaker_directory(speaker_dir, transcriptions):
    paired_mel_spectrograms = []

    # Filter recording IDs by emotion
    neutral_ids = {rec_id for rec_id, (_, emotion) in transcriptions.items() if emotion == 'Neutral'}
    happy_ids = {rec_id for rec_id, (_, emotion) in transcriptions.items() if emotion == 'Happy'}

    # Assuming a systematic way to derive happy IDs from neutral IDs
    for neutral_id in neutral_ids:
        happy_id = derive_happy_id(neutral_id)  # Implement this function based on your ID system
        if happy_id in happy_ids:
            try:
                neutral_mel_spec_db, happy_mel_spec_db = pair_mel_spectrograms(speaker_dir, neutral_id, happy_id)
                paired_mel_spectrograms.append((neutral_mel_spec_db, happy_mel_spec_db))
            except Exception as e:
                print(f"Error processing pair {neutral_id}, {happy_id}: {e}")

    return paired_mel_spectrograms

def pair_mel_spectrograms(speaker_dir, neutral_id, happy_id):
    neutral_path = os.path.join(speaker_dir, 'Neutral', neutral_id + '.wav')
    happy_path = os.path.join(speaker_dir, 'Happy', happy_id + '.wav')

    neutral_y, sr = librosa.load(neutral_path, sr=None)
    neutral_mel_spec = librosa.feature.melspectrogram(y=neutral_y, sr=sr)
    neutral_mel_spec_db = librosa.power_to_db(neutral_mel_spec, ref=np.max)

    happy_y, sr = librosa.load(happy_path, sr=None)
    happy_mel_spec = librosa.feature.melspectrogram(y=happy_y, sr=sr)
    happy_mel_spec_db = librosa.power_to_db(happy_mel_spec, ref=np.max)

    return neutral_mel_spec_db, happy_mel_spec_db

def normalize_mel_spectrograms(mel_spectrograms):
    if not mel_spectrograms or not all(isinstance(pair, tuple) and len(pair) == 2 for pair in mel_spectrograms):
        raise ValueError("Input mel_spectrograms must be a non-empty list of tuples, each containing two numpy arrays.")

    # Determine the maximum size for padding
    max_size = max(mel_spec.shape[1] for pair in mel_spectrograms for mel_spec in pair if isinstance(mel_spec, np.ndarray))

    # Pad and flatten the spectrograms
    flat_mels = np.vstack([np.pad(mel_spec, ((0, 0), (0, max_size - mel_spec.shape[1])), 'constant', constant_values=(0,)).flatten() 
                           for mel_spec_pair in mel_spectrograms 
                           for mel_spec in mel_spec_pair if isinstance(mel_spec, np.ndarray)])

    if flat_mels.size == 0:
        raise ValueError("No valid Mel-spectrogram arrays found in mel_spectrograms.")

    scaler = StandardScaler().fit(flat_mels)
    joblib.dump(scaler, 'mel_scaler.save')

    # Normalize each spectrogram and return them in their original (padded) shape
    normalized_mels = [(scaler.transform(np.pad(mel_spec_pair[0], ((0, 0), (0, max_size - mel_spec_pair[0].shape[1])), 'constant', constant_values=(0,)).flatten().reshape(1, -1)).reshape(-1, max_size), 
                        scaler.transform(np.pad(mel_spec_pair[1], ((0, 0), (0, max_size - mel_spec_pair[1].shape[1])), 'constant', constant_values=(0,)).flatten().reshape(1, -1)).reshape(-1, max_size)) 
                       for mel_spec_pair in mel_spectrograms if all(isinstance(mel_spec, np.ndarray) for mel_spec in mel_spec_pair)]

    return normalized_mels, scaler


# def normalize_mel_spectrograms(mel_spectrograms):
#     # Check if mel_spectrograms is not empty and contains numpy arrays
#     if not mel_spectrograms or not all(isinstance(pair, tuple) and len(pair) == 2 for pair in mel_spectrograms):
#         raise ValueError("Input mel_spectrograms must be a non-empty list of tuples, each containing two numpy arrays.")
    
#     # Flatten the spectrogram arrays for StandardScaler
#     flat_mels = np.vstack([mel_spec.flatten() for mel_spec_pair in mel_spectrograms for mel_spec in mel_spec_pair if isinstance(mel_spec, np.ndarray)])

#     # If flat_mels is empty, raise an error
#     if flat_mels.size == 0:
#         raise ValueError("No valid Mel-spectrogram arrays found in mel_spectrograms.")

#     # Create a StandardScaler, fit on the flattened spectrograms
#     scaler = StandardScaler().fit(flat_mels)
    
#     # Normalize each spectrogram and return them in their original shape
#     normalized_mels = [(scaler.transform(mel_spec_pair[0].flatten().reshape(1, -1)).reshape(mel_spec_pair[0].shape), 
#                         scaler.transform(mel_spec_pair[1].flatten().reshape(1, -1)).reshape(mel_spec_pair[1].shape)) 
#                        for mel_spec_pair in mel_spectrograms if all(isinstance(mel_spec, np.ndarray) for mel_spec in mel_spec_pair)]
    
#     return normalized_mels, scaler


def build_seq2seq_model(input_shape, latent_dim=256):
    # Encoder
    encoder_inputs = Input(shape=input_shape)
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = RepeatVector(input_shape[0])(encoder_outputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True)
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states) 
    decoder_dense = TimeDistributed(Dense(input_shape[1]))
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Model
    model = Model(encoder_inputs, decoder_outputs)
    return model

# Define the dataset directory
dataset_directory = 'Data'
paired_mel_spectrograms = []

# Iterate over all speaker directories 
for speaker_dir_name in sorted(os.listdir(dataset_directory)):
    #if speaker_dir_name == "0011" :
    speaker_dir = os.path.join(dataset_directory, speaker_dir_name) 
    txt_name = speaker_dir_name + ".txt"
    transcription_path = os.path.join(speaker_dir, txt_name) 
    #print(transcription_path)
    if os.path.exists(transcription_path): 
        #print("exter the if ")
        transcriptions = load_transcriptions(transcription_path) 
        #print(len(transcriptions))
        speaker_pairs = process_speaker_directory(speaker_dir, transcriptions) 
        paired_mel_spectrograms.extend(speaker_pairs)

# Apply normalization
#print(len(paired_mel_spectrograms))
paired_mel_spectrograms_normalized, scaler = normalize_mel_spectrograms(paired_mel_spectrograms)
# Split the data into training and validation sets
train_pairs, val_pairs = train_test_split(paired_mel_spectrograms_normalized, test_size=0.1, random_state=42)

# Separate the pairs into individual lists for the model
X_train, Y_train = zip(*train_pairs)
X_val, Y_val = zip(*val_pairs)

# Convert the lists to numpy arrays for training
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_val = np.array(X_val)
Y_val = np.array(Y_val)

# Assuming a shape for the Mel-spectrograms, replace with your actual shape
mel_shape = X_train.shape[1:]  # e.g., (time_steps, mel_bins)
seq2seq_model = build_seq2seq_model(mel_shape)

# Compile the model
seq2seq_model.compile(optimizer='adam', loss='mse')

def mel_spectrogram_to_audio(mel_spec, sr=22050):
    """
    Convert a Mel-spectrogram back to an audio waveform.
    
    Parameters:
    - mel_spec: numpy.ndarray, the Mel-spectrogram.
    - sr: int, the sampling rate.

    Returns:
    - audio: numpy.ndarray, the reconstructed audio waveform.
    """
    # Convert the Mel-spectrogram back to audio
    audio = librosa.feature.inverse.mel_to_audio(mel_spec, sr=sr)
    
    # Normalize the audio
    audio = librosa.util.normalize(audio)
    
    return audio

# Train the model
history = seq2seq_model.fit(X_train, Y_train, 
                            batch_size=32, 
                            epochs=2, 
                            validation_data=(X_val, Y_val))

def preprocess_audio_for_model(wav_file_path, sr=22050, n_mels=128, hop_length=512, scaler=None, target_shape=(128, 201)):
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

# def recover_wav(mel, wav_path, mel_mean_std, ismel=False, n_fft = 2048, win_length=800, hop_length=200):
#     if ismel:
#         mean, std = np.load(mel_mean_std)
#     else:
#         mean, std = np.load(mel_mean_std.replace('mel','spec'))
    
#     mean = mean[:,None]
#     std = std[:,None]
#     mel = 1.2 * mel * std + mean
#     mel = np.exp(mel)

#     if ismel:
#         filters = librosa.filters.mel(sr=16000, n_fft=2048, n_mels=80)
#         inv_filters = np.linalg.pinv(filters)
#         spec = np.dot(inv_filters, mel)
#     else:
#         spec = mel

#     def _griffin_lim(stftm_matrix, shape, max_iter=50):
#         y = np.random.random(shape)
#         for i in range(max_iter):
#             stft_matrix = librosa.core.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
#             stft_matrix = stftm_matrix * stft_matrix / np.abs(stft_matrix)
#             y = librosa.core.istft(stft_matrix, win_length=win_length, hop_length=hop_length)
#         return y

#     shape = spec.shape[1] * hop_length -  hop_length + 1

#     y = _griffin_lim(spec, shape)
#     #scipy.io.wavfile.write(wav_path, 16000, y)
#     sf.write(wav_path,y,16000)
#     return y


# Example usage
wav_file_path = 'joy6.wav'
std_scaler = joblib.load('mel_scaler.save')
X_test = preprocess_audio_for_model(wav_file_path) 
predicted_mel_spectrograms = seq2seq_model.predict(X_test)

# Convert the first predicted Mel-spectrogram to audio (as an example)
# Note: This assumes 'predicted_mel_spectrograms' is a 3D numpy array where the first dimension is the batch size
predicted_audio = mel_spectrogram_to_audio(predicted_mel_spectrograms[0], sr=22050)

# Save the reconstructed audio to a file
output_path = 'predicted_happy_emotion_2.wav'
sf.write(output_path, predicted_audio, 22050)

print("Yea")