import torch
import torch.nn as nn

import Transformer
import VQ_VAE

# Config for VQ VAE
batch_size = 256
num_training_updates = 15000

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-3

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQ_VAE.VQ_VAE(num_hiddens,
                          num_residual_layers,
                          num_residual_hiddens,
                          num_embeddings,
                          embedding_dim,
                          commitment_cost,
                          decay).to(device)


