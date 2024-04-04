import time

import librosa
import soundfile as sf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import Transformer
import VQ_VAE
import Config
import SoundProcess
from IPython.display import Audio


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
    torch.save(model,'VQVAE.pth')

def validation():
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

if __name__ == '__main__':
    validation()


