import os
import shutil
import time
import librosa
import soundfile as sf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import Transformer
import VQ_VAE
import Config
import SoundProcess

import pyttsx3


class MelSpectrogramDataset(Dataset):
    def __init__(self, file_paths, segment_length):
        self.file_paths = file_paths
        self.segment_length = segment_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load audio file
        processed_mel_spec = SoundProcess.preprocess_audio_for_model(self.file_paths[idx])
        return processed_mel_spec


class EmotionSpeechDataset(Dataset):
    def __init__(self, original_dir, tts_dir, label_file):
        self.samples = []
        with open(label_file, 'r') as f:
            for line in f:
                filename, emotion = line.strip().split('\t')
                original_wav = os.path.join(original_dir, filename)
                tts_wav = os.path.join(tts_dir, filename)
                self.samples.append((original_wav, tts_wav, emotion))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        original_wav, tts_wav, emotion = self.samples[idx]
        origin_mel = SoundProcess.preprocess_audio_for_model(original_wav)
        tts_mel = SoundProcess.preprocess_audio_for_model(tts_wav)
        return origin_mel, tts_mel, emotion


def train_VQVAE():
    config_audio = Config.get_Audio_Config()
    config_vqvae = Config.get_VQVAE_Config()

    # prepare the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQ_VAE.VQ_VAE(config_vqvae["num_hiddens"],
                          config_vqvae["num_residual_layers"],
                          config_vqvae["num_residual_hiddens"],
                          config_vqvae["num_embeddings"],
                          config_vqvae["embedding_dim"],
                          config_vqvae["commitment_cost"],
                          config_vqvae["decay"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config_vqvae['learning_rate'])
    file_paths = SoundProcess.get_wav_files(config_audio['audio_path'])
    dataset = MelSpectrogramDataset(file_paths, segment_length=config_audio['segments'])
    data_loader = DataLoader(dataset, batch_size=config_vqvae['batch_size'], shuffle=True)

    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        # Using tqdm for a progress bar
        data_loader_tqdm = tqdm(data_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for batch in data_loader_tqdm:
            batch_loss = 0
            optimizer.zero_grad()
            inputs = batch.to(device)
            vq_loss, data_recon, perplexity = model(inputs)
            recon_loss = VQ_VAE.vq_vae_loss(data_recon, inputs)
            loss = recon_loss + vq_loss
            batch_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_loss += batch_loss / len(batch)
            data_loader_tqdm.set_postfix(train_loss=train_loss / (len(data_loader)))

    ## save the path
    torch.save(model, 'VQVAE.pth')


def validation_Reconsturct():
    model = torch.load('VQVAE_50_Epoch.pth')
    model.eval()
    config_audio = Config.get_Audio_Config()
    file_paths = SoundProcess.get_wav_files(config_audio['audio_path'])
    dataset = MelSpectrogramDataset(file_paths, segment_length=config_audio['segments'])
    data_loader = DataLoader(dataset, 1, shuffle=True)

    valid_originals = next(iter(data_loader))
    valid_originals = valid_originals.to('cuda')
    vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
    _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
    valid_reconstructions = model._decoder(valid_quantize)

    y = SoundProcess.mel_spectrogram_to_audio(valid_reconstructions[0].cpu().data, config_audio['sample_rate'])
    sf.write('output1.wav', y, config_audio['sample_rate'])

    y = SoundProcess.mel_spectrogram_to_audio(valid_originals[0].cpu().data, config_audio['sample_rate'])
    sf.write('output2.wav', y, config_audio['sample_rate'])


def train_Transformer():
    # Transformer Setting
    config_transformer = Config.get_Transformer_Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # n_vocab = 512  # determined by VQ_VAE setting, num_embedding = 512
    # # seq_len = 51200  # determined by VQ_VAE setting, size of the quantized voice is: torch.Size([1, 64, 32, 25]), so the seq_len is 64*32*25 = 51200
    # d_model = 64  # determined by VQ_VAE token, which is torch.Size([1, 64, 32, 25]), the dimension is 64
    # model = Transformer.build_transformer(src_vocab_size=n_vocab,
    #                                       trg_vocab_size=n_vocab,
    #                                       src_seq_len=64 * 32 * 26,
    #                                       trg_seq_len=64 * 32 * 25,
    #                                       d_model=d_model).to(device)

    model = Transformer.MyTransformerModel(config_transformer['input_dim'],
                                           config_transformer['output_dim'],
                                           config_transformer['d_model'],
                                           config_transformer['num_heads'],
                                           config_transformer['num_layers'],
                                           config_transformer['dim_feedforward']).to(device)
    writer = SummaryWriter(config_transformer['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config_transformer['learning_rate'], eps=1e-9)
    loss_function = nn.MSELoss().to(device)

    # Import Trained VQ-VAE
    vqvae = torch.load('VQVAE_50_Epoch.pth')
    vqvae.eval().to(device)

    # Prepare Dataset
    config_DS = Config.get_TransDS_Config()
    dataset = EmotionSpeechDataset(config_DS["origin_path"], config_DS["TTS_path"], config_DS["label_path"])
    data_loader = DataLoader(dataset, config_transformer['batch_size'], shuffle=True)

    # Train Epoches
    seq_len = config_transformer['sequence_length']
    input_dim = config_transformer['input_dim']
    output_dim = config_transformer['output_dim']
    num_epochs = config_transformer['num_epochs']
    step = 1
    for epoch in range(num_epochs):
        model.train()
        data_loader_tqdm = tqdm(data_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        optimizer.zero_grad()
        for batch in data_loader_tqdm:
            origin, tts, emotion = batch
            batch_size, _, _, _ = origin.shape
            origin = origin.to(device)  # origin is the emotional talk
            tts = tts.to(device)  # tts is the non-emotion txt-to speach

            # calculate the token of each part
            vq_output = vqvae._pre_vq_conv(vqvae._encoder(origin))
            _, origin_quantize, _, _ = vqvae._vq_vae(vq_output)

            vq_output = vqvae._pre_vq_conv(vqvae._encoder(tts))
            _, tts_quantize, _, _ = vqvae._vq_vae(vq_output)

            # Concentrate emotion and the tts
            tts_quantize = concatenate_emotion_token(tts_quantize, emotion)

            src = tts_quantize.view(-1, batch_size * seq_len, input_dim)
            tgt = origin_quantize.view(-1, batch_size * seq_len, output_dim)
            out = model(src, tgt)

            target = tgt.view(-1, 25)
            output = out.view(-1, 25)
            loss = loss_function(output, target)

            data_loader_tqdm.set_postfix({"loss": f"{loss:6.3f}"})
            writer.add_scalar('train loss', loss.item(), step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
    torch.save(model, 'TransVAE.pth')

def audio_synesvalidation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqvae = torch.load('VQVAE_50_Epoch.pth')
    trans = torch.load('TransVAE.pth')

    vqvae.eval().to(device)
    trans.eval().to(device)

    config_DS = Config.get_TransDS_Config()
    config_transformer = Config.get_Transformer_Config()
    config_audio = Config.get_Audio_Config()

    dataset = EmotionSpeechDataset(config_DS["origin_path"], config_DS["TTS_path"], config_DS["label_path"])
    data_loader = DataLoader(dataset, 1, shuffle=True)
    with torch.no_grad():
        valid_originals, valid_tts, emotion = next(iter(data_loader))
        y = SoundProcess.mel_spectrogram_to_audio(valid_originals[0].cpu().data, config_audio['sample_rate'])
        sf.write('Original.wav', y, config_audio['sample_rate'])

        valid_originals = valid_originals.to(device)
        valid_tts = valid_tts.to(device)

        #apply vq-vae
        vq_output = vqvae._pre_vq_conv(vqvae._encoder(valid_originals))
        _, origin_quantize, _, _ = vqvae._vq_vae(vq_output)
        vq_output = vqvae._pre_vq_conv(vqvae._encoder(valid_tts))
        _, tts_quantize, _, _ = vqvae._vq_vae(vq_output)

        # Concentrate emotion and the tts
        tts_quantize = concatenate_emotion_token(tts_quantize, emotion)

        seq_len = config_transformer['sequence_length']
        input_dim = config_transformer['input_dim']
        output_dim = config_transformer['output_dim']
        batch_size = 1
        src = tts_quantize.view(-1, batch_size * seq_len, input_dim)
        tgt = origin_quantize.view(-1, batch_size * seq_len, output_dim)
        out = trans(src, tgt)

        # transform the size to fit for output
        out = out.unsqueeze(0)
        out = out.permute(0, 2, 1, 3)

        # reconstruct
        valid_reconstructions = vqvae._decoder(out)

        # write the output to file
        y = SoundProcess.mel_spectrogram_to_audio(valid_reconstructions[0].cpu().data, config_audio['sample_rate'])
        sf.write('Final_output.wav', y, config_audio['sample_rate'])

# Tool Function for tokenize emotion and concat it to the input parameter to Transformer
def concatenate_emotion_token(audio_tokens_batch, emotion_labels_batch, num_time_steps=32):
    # Convert the batch of emotion labels to a batch of emotion tokens
    emotion_tokens_batch = [Config.get_Emotion_Id(label) for label in emotion_labels_batch]
    emotion_tensor_batch = torch.tensor(emotion_tokens_batch, dtype=torch.int64).to('cuda')

    # Expand the emotion tensor to match the shape of the audio tokens
    emotion_tensor_batch = emotion_tensor_batch.view(-1, 1, 1, 1).expand(-1, 64, num_time_steps, 1)

    # Concatenate the emotion tensor to the audio tokens
    tokenized_input_batch = torch.cat((audio_tokens_batch, emotion_tensor_batch), dim=3)
    return tokenized_input_batch


def create_mask(input_tensor):
    batch_size, seq_length = input_tensor.shape
    mask = torch.ones((seq_length), dtype=torch.bool)
    return mask


def flatten_token(input_tensor):
    batch_size, embedding_dim, seq_length, num_tokens = input_tensor.shape
    flattened_seq_length = seq_length * num_tokens * embedding_dim
    flattened_tensor = input_tensor.view(batch_size, flattened_seq_length)
    return flattened_tensor


if __name__ == '__main__':
    # validation_Reconsturct()
    audio_synesvalidation()
    # train_Transformer()