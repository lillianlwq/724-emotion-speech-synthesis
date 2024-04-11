from sad_speech import emotional_prosody_modulation
from happy_speech import emotional_prosody_modulation_happy, compute_spectrum, plot_spectrum, increase_pitch
from angry_speech import modify_audio_for_anger
from surprise_speech import modify_audio_for_surprise
import librosa
import soundfile as sf
import argparse
import os
import random

# Main script that integrate four speech script and adjust the neutral audio with the passing parameter
def add_emotion_to_speech(emotion_type, input_wav, output_wav):
    print(f"Applying {emotion_type} emotion to {input_wav} and saving to {output_wav}")
    if (emotion_type == 'happy'):
        y, sr = librosa.load(input_wav)
        original_DB_happpy = compute_spectrum(y, sr)
        modified_audio_happy = increase_pitch(y,sr)
        modified_DB_happpy = compute_spectrum(modified_audio_happy, sr)
        plot_spectrum(original_DB_happpy, modified_DB_happpy, sr)
        sf.write(output_wav, modified_audio_happy, sr)
    elif (emotion_type == 'sad'):
        y, sr = librosa.load(input_wav)
        original_DB_sad = compute_spectrum(y, sr)
        modified_audio_sad, sr_sad = emotional_prosody_modulation(input_wav, output_wav, segment_length_sec=0.1, sadness_intensity=1)
        modified_DB_sad = compute_spectrum(modified_audio_sad, sr_sad)
        plot_spectrum(original_DB_sad, modified_DB_sad, sr_sad)
        sf.write(output_wav, modified_audio_sad, sr_sad)
    elif(emotion_type == 'anger'):
        y, sr = librosa.load(input_wav)
        original_DB_angry = compute_spectrum(y, sr)
        modified_audio_angry = modify_audio_for_anger(input_wav, output_wav, segment_length_sec=0.1, anger_intensity=0.8)
        modified_DB_angry = compute_spectrum(modified_audio_angry, sr)
        plot_spectrum(original_DB_angry, modified_DB_angry, sr)
        sf.write(output_wav, modified_audio_angry, sr)
    elif(emotion_type == 'surprise'):
        y, sr = librosa.load(input_wav)
        original_DB_surprise = compute_spectrum(y, sr)
        modified_audio_surprise = modify_audio_for_surprise(input_wav, output_wav, segment_length_sec=0.1, surprise_intensity=0.8)
        modified_DB_surprise = compute_spectrum(modified_audio_surprise, sr)
        plot_spectrum(original_DB_surprise, modified_DB_surprise, sr)
        sf.write(output_wav, modified_audio_surprise, sr)
    else:
        return
    
if __name__ == "__main__":

    # TWO WAY RUNNING THIS SCIPT

    # Option A: Using command line to work with one audio file
    # parser = argparse.ArgumentParser(description="Modify speech emotion.")
    # parser.add_argument("emotion_type", type=str, help="Type of emotion to apply.")
    # parser.add_argument("input_wav", type=str, help="Input WAV file.")
    # parser.add_argument("output_wav", type=str, help="Output WAV file with applied emotion.")
    # args = parser.parse_args()
    # add_emotion_to_speech(args.emotion_type, args.input_wav, args.output_wav)

    # Option B: Work with directory of files
    base_directory = "audio_files"
    folders = ['anger', 'happy', 'sad', 'surprise']
    for folder in folders:
        counter = 1
        folder_path = os.path.join(base_directory, folder)
        all_files = os.listdir(folder_path)
        selected_files = random.sample(all_files, 5)  # randomly choose 5 files to accelerate the program
        for file_name in selected_files:
            file_path = os.path.join(folder_path, file_name)
            output_wav = folder + str(counter) + ".wav"
            add_emotion_to_speech(folder, file_path, output_wav)
            counter += 1
